# Inference performance — 2026-10-05

How fast TimesFM 3.0 runs for this project on Apple silicon, where the time goes, and which
optimizations are worth it: PyTorch on MPS vs. the MLX backend, `torch.compile`, fp16, int8/int4
weight quantization, batch size and symmetric averaging.

Back to the [README](../README.md) · Accuracy results: [RESULTS.md](RESULTS.md) · CLI reference: [USAGE.md](USAGE.md)

## Summary

- **The live forecast is already fast enough.** `price-forecast tomorrow` spends ~1.6 s loading the
  checkpoint and ~1 s on inference. Nothing below changes that meaningfully.
- **The full backtest takes 7.85 min with PyTorch on MPS.** A batched MLX path with fused kernels
  brings it to **6.60 min (1.19×) with the same scores**. MLX in fp16 reaches 5.09 min (1.54×) but
  produced NaN forecasts for 8 of 90 days in one config.
- **The MLX backend as shipped is not faster** than PyTorch here. It runs one series at a time
  whenever covariates or symmetric averaging are used.
- **No help on an M1 Max:** fp16 on MPS (30% slower), `torch.compile` (crashes on Python 3.14),
  larger batches (only the 1-variate config gains) and int8/int4 weights (slower than fp16 and less
  accurate). The workload is compute-bound, not bandwidth-bound.
- **The biggest lever is symmetric averaging.** Turning it off (`--no-symmetric`) halves the runtime
  for ~3% more MAE. That trade is worth taking for exploratory sweeps, not for headline numbers.

## Setup

- **Machine:** MacBook Pro, Apple M1 Max (10-core CPU, 32-core GPU, ~10.4 TFLOPS fp32), 64 GB,
  macOS 26.5.2.
- **Software:** Python 3.14.1, PyTorch 2.14.1, `timesfm` 3.0.2 from PyPI (the project's locked
  version) for PyTorch. For MLX: `timesfm` from GitHub main
  ([`e51928e2`](https://github.com/google-research/timesfm/commit/e51928e2)) and `mlx` 0.32.3, in a
  separate environment.
- **Workload:** the backtest as in [RESULTS.md](RESULTS.md). 112 days of context (10,752 quarter
  hours = 336 input patches), a 96-slot horizon rounded up to 128 (4 more patches), symmetric
  averaging on, batch size 8. The five configs feed the model 1, 9, 15, 19 and 11 variates
  (targets + covariates).
- **Size of the job:** ~3.4 million patch tokens through the 20-layer, 1280-wide transformer, or
  about **2.3 PFLOP** for the full backtest.
- **Micro-benchmarks:** the 8 most recent 96-slot days, timed after a warm-up call. s/day is wall
  time divided by 8.
- **Full backtest:** wall time for all 5 configs × 90 days (2026-07-08 .. 2026-10-05), including
  model loading and MLX compilation, excluding data download and scoring.
- **Accuracy check:** every variant is compared with the current PyTorch fp32 forecasts. "Δ max" is
  the largest absolute difference over all quantiles, slots and days, in EUR/MWh.

The benchmarks were run with throwaway scripts outside this repository; the setup above is enough
to reproduce them.

## Where the time goes (PyTorch, MPS)

8 days, one `predict_batch` call per config, each part timed with a GPU sync around it.

| Config | Variates | Total | Transformer | Running stats loop | CPM refine loop | Other |
|---|---|---|---|---|---|---|
| `tfm_de` | 1 | 1.59 s | 0.73 s | 0.32 s | 0.42 s | 0.12 s |
| `tfm_de_wx_tso` | 11 | 8.40 s | 7.40 s | 0.32 s | 0.47 s | 0.21 s |
| `tfm_multi_wx_tso` | 19 | 14.44 s | 13.39 s | 0.33 s | 0.45 s | 0.26 s |

- **The transformer dominates** every config with covariates. It runs at about 5 TFLOPS, roughly
  half the GPU's fp32 peak.
- **Two Python loops cost a fixed ~0.38 s per forward pass.** The library computes the RevIN
  running statistics (`util.get_running_stats`) and the CPM refinement
  (`cpm_iterative_revin_refine`) patch by patch, 340 iterations of ~25 small tensor ops each. On MPS
  that is thousands of tiny kernel launches, which is half the runtime of the 1-variate config.
- **Loading the checkpoint** takes 1.6 s with PyTorch and 0.7–2.3 s with MLX.

## Full backtest

All 5 configs × 90 days:

| Backend | Wall time | Speedup | Accuracy vs. PyTorch fp32 |
|---|---|---|---|
| PyTorch, MPS, fp32 (current) | **7.85 min** | 1.00× | — |
| MLX fp32, batched + fused kernels | **6.60 min** | **1.19×** | every score within 0.001; Δ max 0.17 on the median |
| MLX fp16, batched + fused kernels | 5.09 min | 1.54× | ✗ `tfm_multi_wx`: 8 of 90 days entirely NaN; other configs Δ max ≤ 1.02, MAE within 0.005 |

Per config, in seconds for 90 days:

| Config | PyTorch fp32 | MLX fp32 | MLX fp16 |
|---|---|---|---|
| `tfm_de` | 14.1 | 7.1 (2.0×) | 5.6 |
| `tfm_multi` | 73.7 | 65.2 (1.13×) | 51.0 |
| `tfm_multi_wx` | 125.9 | 108.3 (1.16×) | 82.7 (NaN days) |
| `tfm_multi_wx_tso` | 157.7 | 137.7 (1.15×) | 103.8 |
| `tfm_de_wx_tso` | 97.9 | 77.1 (1.27×) | 61.5 |

The configs with many variates take most of the time, and they are limited by raw matrix-multiply
throughput, which is about the same in both frameworks. The fp16 NaNs hit 2026-08-23, 08-25 .. 08-27
and 08-29 .. 09-01, which looks like an overflow. Keeping the input block in fp32 or clamping the
normalized inputs might fix it, but that would need validating again.

## Everything tried (8-day micro-benchmark)

s/day per config; Δ max in EUR/MWh against the current PyTorch fp32 forecasts.

| Variant | `tfm_de` | `tfm_de_wx_tso` | `tfm_multi_wx_tso` | Δ max | Verdict |
|---|---|---|---|---|---|
| **PyTorch MPS fp32, batch 8 (current)** | 0.163 | 0.997 | 1.769 | — | baseline |
| PyTorch, batch 16 | 0.135 | 1.040 | 1.730 | 0 | helps the 1-variate config only |
| PyTorch, batch 32 | 0.131 | 1.037 | 1.730 | 0 | same |
| PyTorch, CPM refine loop skips the context patches | 0.143 | 0.973 | 1.744 | 0 | exact, small gain |
| PyTorch, running stats via cumulative sums + the above | 0.106 | 0.935 | 1.705 | 5.46 | ✗ constant covariates (e.g. `is_weekend`) get tiny non-zero σ |
| PyTorch, fp16 autocast | 0.210 | 1.316 | 2.275 | 0.43 | ✗ slower: no fast fp16 path on M1 |
| PyTorch, `torch.compile` | — | — | — | — | ✗ crashes while importing `torch._functorch` on Python 3.14 |
| MLX `predict_batch` as shipped (`mx.compile` on) | 0.208 | 1.095 | 1.758 | 16.31 | ✗ not faster; differs on `tfm_de` (see below) |
| MLX as shipped, `mx.compile` off | 0.429 | 1.365 | 2.162 | 16.31 | ✗ |
| MLX batched, batch 8 | 0.086 | 0.958 | 1.614 | 0.001 | |
| MLX batched, batch 16 | 0.072 | 0.897 | 1.516 | 0.001 | |
| **MLX batched + fused kernels, batch 16, fp32** | **0.064** | **0.807** | **1.365** | **0.002** | **best exact option** |
| MLX batched, batch 16, fp16 | 0.073 | 0.905 | 1.531 | 7.33 | ✗ |
| MLX batched + fused kernels, fp16 | 0.055 | 0.677 | 1.149 | 0.96 | fastest; NaNs in the full run |
| MLX + int8 weights (fp16 activations) | 0.059 | 0.745 | — | 4.29 | ✗ slower than fp16, larger drift |
| MLX + int4 weights (fp16 activations) | 0.060 | 0.756 | — | 46.95 | ✗ MAE 15.36 → 16.27 on `tfm_de_wx_tso` |
| MLX + int8 weights (fp32 activations) | 0.075 | 0.967 | — | 5.11 | ✗ slower than plain fp32 |

- **"Batched"** means the symmetric pairs of all days in a config go through `decode()` together in
  equal-sized batches of at most 16 (one compiled shape per config). This mirrors the PyTorch
  `predict_batch` semantics exactly: edge padding, decoding at the 128-step padded horizon, sorted
  quantiles, symmetric averaging.
- **"Fused kernels"** means swapping the hand-written attention, RMSNorm and RoPE in `timesfm3.mlx`
  for `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm` and `mx.fast.rope`.
- **Quantization** covered the transformer's linear layers (group size 64). It doesn't pay off
  because each batch pushes 50–100k patch tokens through every weight matrix. The job is
  compute-bound, so smaller weights save no time and dequantizing them adds work.

## Notes on the MLX backend

- **It is already in `timesfm` 3.0.2** (`timesfm3.mlx`); only the `mlx` package itself is missing
  from this project's environment. It reads the `google/timesfm-3.0-pytorch` checkpoint directly.
- **One series at a time.** With covariates, multivariate targets, symmetric averaging or edge padding, `predict_batch`
  calls `decode()` once per series and sign, so it gains nothing from batching.
- **Without covariates it does not match PyTorch.** It decodes the 96-step horizon directly, while
  PyTorch decodes the patch-rounded 128 steps. The patch stitching then differs for the last 32
  slots, up to 16 EUR/MWh (8-day MAE on `tfm_de`: 25.83 vs. 25.72).
- **The upstream fixes after 3.0.2 should not change these results.** The commits on main make MLX
  honour config flags (activation, stitching, CPM refine, variate RoPE, value norm) and zero out
  NaNs. The public checkpoint uses exactly the defaults that 3.0.2 hardcoded, and this project's
  inputs have no NaNs. This is from reading the diff, not from a 3.0.2 MLX run.
- **The upstream README's MLX latency numbers don't transfer.** They are for short univariate
  inputs (M4 Max, context 512, horizon 64). This workload is 11–19 variates × 340 patches.

## Symmetric averaging

Symmetric averaging forecasts each series and its negation and averages the two, which doubles the
compute. 90-day backtest, PyTorch on MPS:

| Config | Symmetric | MAE | RMSE | Pinball | 3h hit | s/day |
|---|---|---|---|---|---|---|
| `tfm_de` | on | 24.21 | 38.01 | 9.52 | 72.2% | 0.160 |
| `tfm_de` | off | 24.49 | 38.49 | 9.69 | 71.1% | 0.081 |
| `tfm_de_wx_tso` | on | 13.22 | 20.90 | 5.24 | 85.6% | 0.982 |
| `tfm_de_wx_tso` | off | 13.63 | 21.59 | 5.38 | 84.4% | 0.492 |

## Estimate: NVIDIA H100 / B200

These are estimates, not measurements. They take the ~2.3 PFLOP of the full backtest (consistent
with the 4.9 TFLOPS the M1 Max achieves in 471 s) and typical efficiencies for 1280-wide matrix
multiplies.

| Setup | H100 (SXM) | B200 |
|---|---|---|
| Code as is, plain fp32 (`--device cuda`) | ~1–1.5 min | ~1–1.3 min |
| TF32 on (`torch.set_float32_matmul_precision("high")`), batch 64 | ~25–40 s | ~15–25 s |
| bf16 autocast, batch 64 | ~20–30 s | ~15–20 s |

- **By default PyTorch doesn't use the tensor cores for fp32.** Plain fp32 runs on the CUDA cores
  (67 TFLOPS on an H100, ~75–80 on a B200), so the B200 barely helps there.
- **At these speeds, overheads dominate:** the per-patch loops (0.1–0.2 s of kernel launches per
  forward pass), CUDA start-up and checkpoint loading. A larger batch cuts the number of forward
  passes; going further would need the loops vectorized or compiled.
- **TF32 keeps fp32's range,** so the overflow seen with MLX fp16 can't happen. bf16 has the same
  range but would also need its drift checked against the current scores.

## Possible improvements

- **For many backtest sweeps:** a `--backend mlx` option with the batched, fused-kernel MLX path
  (1.19× overall, up to 2× for small configs, same scores).
- **For exploratory runs:** `--no-symmetric` (2×, ~3% more MAE).
- **Upstream (`google-research/timesfm`):**
  - batch the general path of `timesfm3.mlx` `predict_batch`;
  - use the `mx.fast` kernels in the MLX transformer;
  - skip the context patches in the CPM refine loop, which is exact in both backends;
  - align the MLX no-covariate horizon with PyTorch.
