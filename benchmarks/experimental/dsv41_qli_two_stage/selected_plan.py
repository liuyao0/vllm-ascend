"""Benchmark dispatch between dense masks and membership-packed K.

Selection uses host workload bounds before graph capture. This is not yet a
vLLM scheduler/graph-key integration. All plans retain correctness if graph
inputs change; a changed context range can change which plan is fastest.
"""

from local_cap import LocalCap
from mask_plan import MaskPlan
from packed_plan import PackedPlan


def allocate_plan(work, minimum_context):
    if minimum_context >= 98304:
        return "packed", PackedPlan.allocate(work)
    if work.max_length < 65536:
        return "cached_caps", MaskPlan.allocate(work)
    return "local_caps", LocalCap.allocate(work)
