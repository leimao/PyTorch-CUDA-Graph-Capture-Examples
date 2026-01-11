# PyTorch CUDA Graph Capture Examples

## Introduction

CUDA Graph is a useful feature to reduce CPU overhead for launching GPU kernels. This repository contains examples of how to capture CUDA Graphs using PyTorch CUDA semantics.

## Usages

### Run Docker Container

To run the NVIDIA NGC PyTorch Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus all -v $(pwd):/mnt -w /mnt nvcr.io/nvidia/pytorch:25.11-py3
```

### Run Examples

To capture CUDA Graph manually using PyTorch `torch.cuda.graph` API, please run the following command.

```bash
$ python torch_cuda_graph_manual_capture.py
CUDA Graph Whole Network Capture Example
======================================================================
Using device: NVIDIA GeForce RTX 5080

Model configuration:
  Batch size: 640
  Input dim: 4096
  Hidden dims: 2048 -> 1024 -> 512
  Output dim: 256

======================================================================
SCENARIO 1: Training WITHOUT CUDA Graph
======================================================================
Training WITHOUT CUDA graph...
  Completed 10 iterations.

Profiling trace saved to: traces/trace_without_manual_capture.json

Top 10 operations by CUDA time (without CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## forward_pass ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.624ms        59.56%       3.624ms     517.737us           0 B           0 B           0 B           0 B             7
                                          ProfilerStep*         3.22%     778.095us        74.15%      17.902ms       2.557ms       0.000us         0.00%       2.978ms     425.435us           0 B           0 B           0 B           0 B             7
    autograd::engine::evaluate_function: AddmmBackward0         1.91%     461.482us        14.53%       3.509ms     125.313us       0.000us         0.00%       2.971ms     106.111us           0 B           0 B     231.98 MB    -126.88 MB            28
                                         AddmmBackward0         1.31%     317.453us         9.84%       2.375ms      84.836us       0.000us         0.00%       2.717ms      97.051us           0 B           0 B     358.75 MB           0 B            28
                                               aten::mm         4.52%       1.091ms         6.09%       1.470ms      29.999us       2.717ms        44.66%       2.717ms      55.458us           0 B           0 B     358.75 MB     358.75 MB            49
                                     ## forward_pass ##         7.00%       1.691ms        18.60%       4.491ms     641.548us       0.000us         0.00%       2.069ms     295.579us           0 B           0 B     137.81 MB     -65.62 MB             7
                                           aten::linear         0.40%      96.651us         5.85%       1.413ms      50.447us       0.000us         0.00%       1.951ms      69.673us           0 B           0 B      65.62 MB           0 B            28
                                            aten::addmm         3.45%     832.270us         4.48%       1.080ms      38.587us       1.951ms        32.06%       1.951ms      69.673us           0 B           0 B      65.62 MB      65.62 MB            28
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.791ms        29.43%       1.791ms     127.895us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.593ms        26.19%       1.593ms     227.621us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 24.142ms
Self CUDA time total: 6.085ms


======================================================================
SCENARIO 2: Training WITH CUDA Graph
======================================================================
Preparing CUDA graph (warmup + capture)...
  Performing warmup iterations...
  Capturing CUDA graph...
CUDA graph ready.

  Training with graph replay...
  Completed 10 iterations.

Profiling trace saved to: traces/trace_with_manual_capture.json

Top 10 operations by CUDA time (with CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## graph.replay ##         0.00%       0.000us         0.00%       0.000us       0.000us       6.564ms        76.16%       6.564ms     937.756us           0 B           0 B             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       2.663ms        30.89%       2.663ms     295.836us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       2.061ms        23.91%       2.061ms     121.242us           0 B           0 B            17
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us       1.172ms        13.59%       1.172ms     130.176us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     573.472us         6.65%     573.472us      63.719us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     456.896us         5.30%     456.896us      50.766us           0 B           0 B             9
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     333.825us         3.87%     333.825us       9.273us           0 B           0 B            36
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     226.624us         2.63%     226.624us      25.180us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     205.058us         2.38%     205.058us      11.392us           0 B           0 B            18
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     169.183us         1.96%     169.183us      10.574us           0 B           0 B            16
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 8.690ms
Self CUDA time total: 8.619ms


======================================================================
Profiling completed successfully!
View traces in Chrome: chrome://tracing
  - traces/trace_without_manual_capture.json
  - traces/trace_with_manual_capture.json
======================================================================
```

To capture CUDA Graph more automatically using PyTorch `make_graphed_callables` API, please run the following command.

```bash
$ python torch_cuda_graph_make_graphed_callables.py
CUDA Graph Whole Network Capture Example
======================================================================
Using device: NVIDIA GeForce RTX 5080

Model configuration:
  Batch size: 640
  Input dim: 4096
  Hidden dims: 2048 -> 1024 -> 512
  Output dim: 256

======================================================================
SCENARIO 1: Training WITHOUT CUDA Graph
======================================================================
Training WITHOUT CUDA graph...
  Completed 10 iterations.

Profiling trace saved to: traces/trace_without_make_graphed_callables.json

Top 10 operations by CUDA time (without CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## forward_pass ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.786ms        62.30%       3.786ms     540.851us           0 B           0 B           0 B           0 B             7
                                          ProfilerStep*         3.20%     725.602us        75.51%      17.097ms       2.442ms       0.000us         0.00%       2.974ms     424.860us           0 B           0 B           0 B           0 B             7
    autograd::engine::evaluate_function: AddmmBackward0         1.84%     416.249us        13.71%       3.104ms     110.858us       0.000us         0.00%       2.973ms     106.185us           0 B           0 B     231.98 MB    -126.88 MB            28
                                         AddmmBackward0         1.25%     282.658us         9.21%       2.085ms      74.461us       0.000us         0.00%       2.722ms      97.223us           0 B           0 B     358.75 MB           0 B            28
                                               aten::mm         4.27%     967.644us         5.73%       1.297ms      26.465us       2.722ms        44.79%       2.722ms      55.556us           0 B           0 B     358.75 MB     358.75 MB            49
                                     ## forward_pass ##         7.58%       1.717ms        20.62%       4.669ms     667.061us       0.000us         0.00%       2.069ms     295.561us           0 B           0 B     137.81 MB     -65.62 MB             7
                                           aten::linear         0.46%     103.960us         6.63%       1.502ms      53.640us       0.000us         0.00%       1.956ms      69.849us           0 B           0 B      65.62 MB           0 B            28
                                            aten::addmm         3.89%     879.966us         5.10%       1.155ms      41.243us       1.956ms        32.18%       1.956ms      69.849us           0 B           0 B      65.62 MB      65.62 MB            28
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.795ms        29.53%       1.795ms     128.192us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.597ms        26.28%       1.597ms     228.169us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 22.641ms
Self CUDA time total: 6.077ms


======================================================================
SCENARIO 2: Training WITH CUDA Graph
======================================================================
Preparing CUDA graph (warmup + capture)...
  Creating graphed model...
  CUDA graph model ready.
CUDA graph ready.

  Training with graph replay...
  Completed 10 iterations.

Profiling trace saved to: traces/trace_with_make_graphed_callables.json

Top 10 operations by CUDA time (with CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         5.42%     620.940us        81.15%       9.292ms       1.327ms       0.000us         0.00%       3.128ms     446.854us           0 B           0 B           0 B           0 B             7
autograd::engine::evaluate_function: GraphedBackward...         1.73%     197.721us         9.66%       1.106ms     157.971us       0.000us         0.00%       3.063ms     437.572us           0 B           0 B      -4.38 MB      -4.38 MB             7
                                        GraphedBackward         4.24%     484.930us         7.65%     876.284us     125.183us       3.053ms        48.09%       3.063ms     437.572us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         0.00%       0.000us         0.00%       0.000us       0.000us       2.316ms        36.47%       2.316ms     330.794us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         4.92%     563.566us        11.65%       1.333ms     190.490us       0.000us         0.00%       2.192ms     313.103us           0 B           0 B           0 B           0 B             7
                                                Graphed         2.76%     315.584us         6.72%     769.863us     109.980us       2.045ms        32.21%       2.192ms     313.103us           0 B           0 B           0 B           0 B             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.759ms        27.70%       1.759ms     125.641us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.556ms        24.51%       1.556ms     222.281us           0 B           0 B           0 B           0 B             7
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us       1.017ms        16.02%       1.017ms     127.164us           0 B           0 B           0 B           0 B             8
                                   ## optimizer.step ##         2.78%     318.836us        13.08%       1.498ms     214.023us       0.000us         0.00%     883.811us     126.259us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.450ms
Self CUDA time total: 6.349ms


======================================================================
Profiling completed successfully!
View traces in Chrome: chrome://tracing
  - traces/trace_without_make_graphed_callables.json
  - traces/trace_with_make_graphed_callables.json
======================================================================

======================================================================
SCENARIO 3: Training WITH PARTIAL CUDA Graph (only block2)
======================================================================
Preparing CUDA graph for block2 only (warmup + capture)...
  Creating partially graphed model (only block2)...
  CUDA graph for block2 ready.
CUDA graph for block2 ready.

  Training with graph replay...
  Completed 10 iterations.

Profiling trace saved to: traces/trace_with_partial_make_graphed_callables.json

Top 10 operations by CUDA time (with partial CUDA graph - block2 only):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             ## forward_pass_graphed ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.626ms       109.36%       3.626ms     517.934us           0 B           0 B           0 B           0 B             7
                                          ProfilerStep*         3.68%     730.731us        78.39%      15.555ms       2.222ms       0.000us         0.00%       2.259ms     322.729us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         8.58%       1.703ms        22.93%       4.549ms     649.839us       0.000us         0.00%       2.106ms     300.809us           0 B           0 B      69.56 MB     -88.38 MB             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.793ms        54.08%       1.793ms     128.066us           0 B           0 B           0 B           0 B            14
                                           aten::linear         0.38%      74.894us         5.74%       1.138ms      54.190us       0.000us         0.00%       1.545ms      73.556us           0 B           0 B      53.38 MB           0 B            21
                                            aten::addmm         3.50%     694.782us         4.42%     876.082us      41.718us       1.545ms        46.59%       1.545ms      73.556us           0 B           0 B      53.38 MB      53.38 MB            21
autograd::engine::evaluate_function: GraphedBackward...         0.58%     115.669us         4.10%     813.511us     116.216us       0.000us         0.00%     561.567us      80.224us           0 B           0 B     -17.50 MB     -17.50 MB             7
                                        GraphedBackward         2.12%     420.204us         3.44%     681.689us      97.384us     546.366us        16.48%     561.567us      80.224us           0 B           0 B           0 B           0 B             7
                                                Graphed         1.65%     327.433us         3.81%     756.580us     108.083us     444.992us        13.42%     481.505us      68.786us           0 B           0 B           0 B           0 B             7
    autograd::engine::evaluate_function: AddmmBackward0         1.19%     235.448us         9.29%       1.844ms      87.803us       0.000us         0.00%     453.055us      21.574us           0 B           0 B      16.65 MB     -27.12 MB            21
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 19.842ms
Self CUDA time total: 3.315ms


======================================================================
All profiling completed successfully!
View traces in Chrome: chrome://tracing
  - traces/trace_without_make_graphed_callables.json
  - traces/trace_with_make_graphed_callables.json
  - traces/trace_with_partial_make_graphed_callables.json
======================================================================
```

The `torch.profiler` profiling results and traces will also be generated.
