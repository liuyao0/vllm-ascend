"""Contiguous per-bucket copy stages for A2; imported by grouped_plan."""

from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit
def count_pack(
    sorted_ids,
    counts,
    packed,
    packed_physical,
    table,
    plan,
    C: tl.constexpr,
    NC: tl.constexpr,
    P2: tl.constexpr,
    S: tl.constexpr,
    PAGE_COLUMNS: tl.constexpr,
    PAGEP2: tl.constexpr,
):
    group, chunk = tl.program_id(0), tl.program_id(1)
    i = chunk * P2 + tl.arange(0, P2)
    value = tl.load(sorted_ids + group * C + 8 + i)
    following = tl.load(sorted_ids + group * C + 8 + i + 1)
    block = tl.floor(value * 0.0625)
    bits = value - block * 16
    for distance in tl.static_range(1, 5):
        previous = tl.load(sorted_ids + group * C + 8 + i - distance)
        # Valid candidates are unique per query. Their membership bits are
        # disjoint, so addition equals OR and avoids A2 scalar int32 bitwise ops.
        previous_block = tl.floor(previous * 0.0625)
        bits += tl.where(previous_block == block, previous - previous_block * 16, 0)
    valid = (value != 16777215) & (block != tl.floor(following * 0.0625))
    bucket = tl.where(block < 64, 0, bits)
    buckets = tl.arange(0, 16)
    histogram = tl.sum((valid[None, :] & (bucket[None, :] == buckets[:, None])).to(tl.float32), axis=1)
    tl.store(counts + (group * NC + chunk) * 16 + buckets, histogram)
    keys = tl.where(bucket == 0, block * 16 + bits, bucket * 131072 + block)
    keys = tl.where(valid, keys, 16777215).to(tl.float32)
    keys = tl.extra.cann.extension.sort(keys, dim=0, descending=False)
    tl.store(packed + (group * NC + chunk) * P2 + tl.arange(0, P2), keys)
    # Resolve each unique physical page once, before splitting it into buckets.
    # The copy stage then contains only contiguous GM transfers.
    live = keys != 16777215
    key_bucket = tl.floor(keys * (1.0 / 131072))
    block = tl.where(key_bucket == 0, tl.floor(keys * 0.0625), keys - key_bucket * 131072)
    request = tl.load(plan + group * S + 2)
    pages = tl.arange(0, PAGEP2)
    page_table = tl.load(table + request * PAGE_COLUMNS + pages, pages < PAGE_COLUMNS, 0)
    logical_page = tl.floor(block * 0.0625)
    page = tl.gather(page_table, tl.where(live, logical_page, 0).to(tl.int32), axis=0)
    physical = tl.where(live, page * 128 + ((block - logical_page * 16) * 8).to(tl.int32), -1)
    tl.store(packed_physical + (group * NC + chunk) * P2 + tl.arange(0, P2), physical)


@tr.jit
def layout(counts, plan, NC: tl.constexpr, NCP2: tl.constexpr, S: tl.constexpr):
    group = tl.program_id(0)
    chunks, buckets = tl.arange(0, NCP2), tl.arange(0, 16)
    histogram = tl.load(counts + (group * NC + chunks[:, None]) * 16 + buckets[None, :], chunks[:, None] < NC, 0)
    totals = tl.sum(histogram, axis=0)
    padded = tl.cdiv(totals, 64) * 64
    bases = tl.cumsum(padded) - padded
    tl.store(plan + group * S + 40 + buckets, totals)
    tl.store(plan + group * S + 4, tl.sum(padded))
    tl.store(plan + group * S + 5, tl.max(tl.where(totals > 0, buckets, 0)))
    tl.store(plan + group * S + 8 + 2 * buckets, bases)
    tl.store(plan + group * S + 9 + 2 * buckets, padded)


@tr.jit
def copy_bucket(
    packed,
    packed_physical,
    counts,
    plan,
    NC: tl.constexpr,
    NCP2: tl.constexpr,
    P2: tl.constexpr,
    S: tl.constexpr,
    OC: tl.constexpr,
):
    group, chunk, bucket = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    size = tl.load(counts + (group * NC + chunk) * 16 + bucket)
    if size > 0:
        buckets = tl.arange(0, 16)
        local_sizes = tl.load(counts + (group * NC + chunk) * 16 + buckets)
        first = tl.sum(tl.where(buckets < bucket, local_sizes, 0))
        chunks = tl.arange(0, NCP2)
        sizes = tl.load(counts + (group * NC + chunks) * 16 + bucket, chunks < chunk, 0)
        base = tl.load(plan + group * S + 8 + 2 * bucket) + tl.sum(sizes)
        lanes = tl.arange(0, P2)
        selected = tl.load(packed + (group * NC + chunk) * P2 + first + lanes, lanes < size, 0).to(tl.float32)
        block = tl.where(
            bucket == 0, tl.floor(selected * 0.0625), selected - tl.floor(selected * (1.0 / 131072)) * 131072
        )
        cap = tl.where(bucket == 0, (selected - tl.floor(selected * 0.0625) * 16).to(tl.int32), 2139095040)
        physical = tl.load(packed_physical + (group * NC + chunk) * P2 + first + lanes, lanes < size, -1)
        tl.store(plan + group * S + 64 + base + lanes, block * 8, lanes < size)
        tl.store(plan + group * S + 64 + OC + base + lanes, cap, lanes < size)
        tl.store(plan + group * S + 64 + 2 * OC + base + lanes, physical, lanes < size)
