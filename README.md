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

Profiling trace saved to: trace_without_manual_capture.json

Top 10 operations by CUDA time (without CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## forward_pass ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.758ms        61.90%       3.758ms     536.862us           0 B           0 B           0 B           0 B             7
    autograd::engine::evaluate_function: AddmmBackward0         1.87%     431.461us        14.04%       3.248ms     115.994us       0.000us         0.00%       2.966ms     105.914us           0 B           0 B     231.98 MB    -126.88 MB            28
                                          ProfilerStep*         3.03%     700.822us        74.69%      17.280ms       2.469ms       0.000us         0.00%       2.962ms     423.154us           0 B           0 B           0 B           0 B             7
                                         AddmmBackward0         1.30%     300.029us         9.45%       2.187ms      78.118us       0.000us         0.00%       2.713ms      96.892us           0 B           0 B     358.75 MB           0 B            28
                                               aten::mm         4.38%       1.012ms         5.90%       1.365ms      27.852us       2.713ms        44.68%       2.713ms      55.367us           0 B           0 B     358.75 MB     358.75 MB            49
                                     ## forward_pass ##         7.38%       1.708ms        19.97%       4.619ms     659.893us       0.000us         0.00%       2.059ms     294.211us           0 B           0 B     137.81 MB     -65.62 MB             7
                                           aten::linear         0.44%     102.551us         6.45%       1.492ms      53.268us       0.000us         0.00%       1.947ms      69.537us           0 B           0 B      65.62 MB           0 B            28
                                            aten::addmm         3.78%     873.558us         4.93%       1.142ms      40.771us       1.947ms        32.07%       1.947ms      69.537us           0 B           0 B      65.62 MB      65.62 MB            28
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.786ms        29.42%       1.786ms     127.583us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.590ms        26.18%       1.590ms     227.072us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 23.134ms
Self CUDA time total: 6.072ms


======================================================================
SCENARIO 2: Training WITH CUDA Graph
======================================================================
Preparing CUDA graph (warmup + capture)...
  Performing warmup iterations...
  Capturing CUDA graph...
CUDA graph ready.

  Training with graph replay...
  Completed 10 iterations.

Profiling trace saved to: trace_with_manual_capture.json

Top 10 operations by CUDA time (with CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## graph.replay ##         0.00%       0.000us         0.00%       0.000us       0.000us       6.075ms        79.14%       6.075ms     867.848us           0 B           0 B             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       2.048ms        26.69%       2.048ms     120.490us           0 B           0 B            17
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.986ms        25.87%       1.986ms     220.677us           0 B           0 B             9
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us       1.098ms        14.30%       1.098ms     121.974us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     571.057us         7.44%     571.057us      63.451us           0 B           0 B             9
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     454.614us         5.92%     454.614us      50.513us           0 B           0 B             9
void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     321.722us         4.19%     321.722us       8.937us           0 B           0 B            36
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     206.010us         2.68%     206.010us      11.445us           0 B           0 B            18
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     182.014us         2.37%     182.014us      11.376us           0 B           0 B            16
                                          ProfilerStep*         4.62%     354.888us        20.97%       1.610ms     230.054us       0.000us         0.00%     169.728us      24.247us           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 7.680ms
Self CUDA time total: 7.676ms


======================================================================
Profiling completed successfully!
View traces in Chrome: chrome://tracing
  - trace_without_manual_capture.json
  - trace_with_manual_capture.json
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

Profiling trace saved to: trace_without_make_graphed_callables.json

Top 10 operations by CUDA time (without CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     ## forward_pass ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.774ms        62.24%       3.774ms     539.116us           0 B           0 B           0 B           0 B             7
                                          ProfilerStep*         3.17%     782.500us        74.41%      18.366ms       2.624ms       0.000us         0.00%       2.964ms     423.444us           0 B           0 B           0 B           0 B             7
    autograd::engine::evaluate_function: AddmmBackward0         1.92%     474.130us        14.28%       3.525ms     125.904us       0.000us         0.00%       2.962ms     105.785us           0 B           0 B     231.98 MB    -126.88 MB            28
                                         AddmmBackward0         1.35%     334.266us         9.64%       2.378ms      84.944us       0.000us         0.00%       2.707ms      96.676us           0 B           0 B     358.75 MB           0 B            28
                                               aten::mm         4.45%       1.099ms         5.98%       1.475ms      30.105us       2.707ms        44.64%       2.707ms      55.243us           0 B           0 B     358.75 MB     358.75 MB            49
                                     ## forward_pass ##         6.93%       1.709ms        18.72%       4.621ms     660.119us       0.000us         0.00%       2.060ms     294.349us           0 B           0 B     137.81 MB     -65.62 MB             7
                                           aten::linear         0.42%     104.014us         5.96%       1.472ms      52.566us       0.000us         0.00%       1.947ms      69.545us           0 B           0 B      65.62 MB           0 B            28
                                            aten::addmm         3.51%     865.235us         4.55%       1.124ms      40.150us       1.947ms        32.11%       1.947ms      69.545us           0 B           0 B      65.62 MB      65.62 MB            28
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.786ms        29.46%       1.786ms     127.600us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.585ms        26.14%       1.585ms     226.432us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 24.681ms
Self CUDA time total: 6.064ms


======================================================================
SCENARIO 2: Training WITH CUDA Graph
======================================================================
Preparing CUDA graph (warmup + capture)...
  Creating graphed model...
  CUDA graph model ready.
CUDA graph ready.

  Training with graph replay...
  Completed 10 iterations.

Profiling trace saved to: trace_with_make_graphed_callables.json

Top 10 operations by CUDA time (with CUDA graph):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         6.10%     718.675us        82.79%       9.751ms       1.393ms       0.000us         0.00%       3.102ms     443.182us           0 B           0 B           0 B           0 B             7
autograd::engine::evaluate_function: GraphedBackward...         1.61%     189.637us         9.05%       1.066ms     152.244us       0.000us         0.00%       3.036ms     433.687us           0 B           0 B      -4.38 MB      -4.38 MB             7
                                        GraphedBackward         4.05%     476.912us         7.17%     844.867us     120.695us       3.027ms        48.18%       3.036ms     433.687us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         0.00%       0.000us         0.00%       0.000us       0.000us       2.401ms        38.20%       2.401ms     342.951us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         5.21%     613.647us        12.03%       1.416ms     202.357us       0.000us         0.00%       2.193ms     313.274us           0 B           0 B           0 B           0 B             7
                                                Graphed         2.90%     341.638us         6.82%     802.849us     114.693us       2.054ms        32.68%       2.193ms     313.274us           0 B           0 B           0 B           0 B             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.768ms        28.14%       1.768ms     126.290us           0 B           0 B           0 B           0 B            14
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.549ms        24.65%       1.549ms     221.285us           0 B           0 B           0 B           0 B             7
void at::native::(anonymous namespace)::multi_tensor...         0.00%       0.000us         0.00%       0.000us       0.000us     979.820us        15.59%     979.820us     122.477us           0 B           0 B           0 B           0 B             8
                                   ## optimizer.step ##         2.99%     351.664us        13.78%       1.623ms     231.871us       0.000us         0.00%     858.414us     122.631us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.778ms
Self CUDA time total: 6.284ms


======================================================================
Profiling completed successfully!
View traces in Chrome: chrome://tracing
  - trace_without_make_graphed_callables.json
  - trace_with_make_graphed_callables.json
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

Profiling trace saved to: trace_with_partial_make_graphed_callables.json

Top 10 operations by CUDA time (with partial CUDA graph - block2 only):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             ## forward_pass_graphed ##         0.00%       0.000us         0.00%       0.000us       0.000us       3.347ms       101.31%       3.347ms     478.112us           0 B           0 B           0 B           0 B             7
                                          ProfilerStep*         3.56%     709.723us        77.04%      15.376ms       2.197ms       0.000us         0.00%       2.271ms     324.414us           0 B           0 B           0 B           0 B             7
                             ## forward_pass_graphed ##         7.86%       1.569ms        21.09%       4.209ms     601.300us       0.000us         0.00%       2.105ms     300.647us           0 B           0 B      69.56 MB     -88.38 MB             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us       1.791ms        54.22%       1.791ms     127.938us           0 B           0 B           0 B           0 B            14
                                           aten::linear         0.33%      66.834us         5.15%       1.029ms      48.989us       0.000us         0.00%       1.544ms      73.506us           0 B           0 B      53.38 MB           0 B            21
                                            aten::addmm         3.10%     619.551us         3.95%     789.079us      37.575us       1.544ms        46.73%       1.544ms      73.506us           0 B           0 B      53.38 MB      53.38 MB            21
autograd::engine::evaluate_function: GraphedBackward...         0.57%     113.346us         4.18%     834.546us     119.221us       0.000us         0.00%     557.302us      79.615us           0 B           0 B     -17.50 MB     -17.50 MB             7
                                        GraphedBackward         2.19%     436.536us         3.53%     705.490us     100.784us     542.133us        16.41%     557.302us      79.615us           0 B           0 B           0 B           0 B             7
                                                Graphed         1.57%     313.169us         3.56%     711.016us     101.574us     444.694us        13.46%     480.916us      68.702us           0 B           0 B           0 B           0 B             7
void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_...         0.00%       0.000us         0.00%       0.000us       0.000us     448.441us        13.57%     448.441us      64.063us           0 B           0 B           0 B           0 B             7
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 19.958ms
Self CUDA time total: 3.304ms


======================================================================
All profiling completed successfully!
View traces in Chrome: chrome://tracing
  - trace_without_make_graphed_callables.json
  - trace_with_make_graphed_callables.json
  - trace_with_partial_make_graphed_callables.json
======================================================================
```

The `torch.profiler` profiling results and traces will also be generated.
