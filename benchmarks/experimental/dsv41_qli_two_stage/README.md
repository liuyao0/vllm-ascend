# DeepSeek V4.1 candidate scoring on Ascend 910B

This experiment compares three implementations for the four QLI candidate
consumers. Each query has up to 2,048 blocks of eight positions and selects
512 positions. The source layer also uses a shorter token Top512 merge while
preserving its independent candidate Top2048. The benchmark owns the plan
and chooses its implementation before graph capture. The vLLM model runner
does not dispatch to these experimental modes.

| Request lengths in a step | Implementation |
| --- | --- |
| Minimum ≥ 98,304 | Group queries by candidate membership, pack K and scales once, reuse them across consumers. |
| Maximum < 65,536 | Build score caps once and reuse them across four dense QK scans. |
| Other lengths, including mixed batches | Build caps in each tile's local buffer during dense QK. |

## Measurements

These are device-event medians on one Ascend 910B4 with CANN 9.2.0, INT8 Q/K,
32 heads, head dimension 128 and randomized physical pages. Each ordinary
measurement includes one source, planning or packing where applicable, four
consumers, and output index preparation. The reference is a separate build
of the fast-mask + Top512 implementation.

| Context | Requests × queries | Reference (ms) | Experiment (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | 4 × 7 | 1.542 | 1.443 | 1.07× |
| 131,072 | 4 × 7 | 3.341 | 1.582 | 2.11× |
| 131,072 | 8 × 7 | 3.396 | 2.418 | 1.40× |
| 131,072 | 1 × 4,096 | 168.296 | 120.733 | 1.39× |

All 56 full-chain cases improved (1.025–2.838×). The other 22 cases inject
candidates to control overlap and time four consumers plus preparation.
Two of these regressed at 81,920 tokens and zero overlap, by 1.73% for
8 × 7 and 0.12% for 1 × 512. [results.json](results.json) contains all 78
cases, the selected path, the scope, and both latencies. These figures
describe a graph microbenchmark. TP8 serving and bounded-replay TTFT were
not measured for this version.

## Reproduce on A2

Use the Ascend checkout at this branch and an A2 environment with the matching
vLLM, CANN and torch-npu installed. From the repository root, build the plugin
and its custom kernels for the device. The following example targets 910B4.

```bash
export SOC_VERSION=ascend910b4 COMPILE_CUSTOM_KERNELS=1
python -m pip install --no-build-isolation --no-deps -e .
bench_dir="$PWD/benchmarks/experimental/dsv41_qli_two_stage"
python "$bench_dir/benchmark_selected.py" --mode graph --phase optimized
```

For an independent reference, use a fresh matching environment or container.
Keep this benchmark directory accessible there. From a clean checkout of
`7ec720c5f8a542e08729d3a785f365cd967da418`, apply its
[reference patch](reference-fast-mask.patch) and rebuild the reference plugin.
Set `bench_dir` to the absolute path of the experiment's benchmark directory.

```bash
git apply "$bench_dir/reference-fast-mask.patch"
export SOC_VERSION=ascend910b4 COMPILE_CUSTOM_KERNELS=1
python -m pip install --no-build-isolation --no-deps -e .
python "$bench_dir/benchmark_selected.py" --mode graph --phase reference
```

Run both phases with the same case range (`--start` inclusive, `--end`
exclusive). Compare the reference phase's `fast_mask_top512` and the optimized
phase's `grouped_total` device times for matching case IDs. The optimized
phase checks selected scores, valid positions, and padding against the
continuous-QK reference on its inputs.
