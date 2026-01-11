#!/usr/bin/env python3
"""
CUDA Graph Manual Capture Example

This script demonstrates how to manually capture and replay CUDA graphs for an
entire training iteration (forward pass, loss computation, backward pass, and
optimizer step). It profiles training with and without CUDA graphs for comparison.

Manual capture using torch.cuda.graph() provides full control over what operations
are included in the graph, allowing you to capture the complete training step
including loss computation and optimizer updates.
"""

import torch
import torch.nn as nn
from torch.profiler import record_function
from common import (train_without_cuda_graph, setup_model_and_data,
                    create_model, create_profiler, save_and_print_profile)


def prepare_cuda_graph(model, loss_fn, optimizer, static_input, static_target):
    """Warmup and capture CUDA graph (not profiled)."""
    print("  Performing warmup iterations...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    print("  Capturing CUDA graph...")
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    return g, static_loss


def train_with_cuda_graph(graph,
                          inputs,
                          targets,
                          static_input,
                          static_target,
                          static_loss,
                          profiler=None):
    """Train using CUDA graph for optimized replay (profiled part only)."""
    print("  Training with graph replay...")

    for i, (data, target) in enumerate(zip(inputs, targets)):
        with record_function("## copy_input_data ##"):
            static_input.copy_(data)
            static_target.copy_(target)

        with record_function("## graph.replay ##"):
            graph.replay()

        if profiler is not None:
            profiler.step()

        # NOTE: Avoid calling .item() in the training loop as it triggers device-to-host
        # memory copy and CPU-GPU synchronization, which damages performance.
        # if i % 2 == 0:
        #     print(f"  Iteration {i+1:2d}: Loss = {static_loss.item():.4f}")

    print(f"  Completed {len(inputs)} iterations.")
    print()


def main():
    print("CUDA Graph Whole Network Capture Example")
    print("=" * 70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. This example requires a CUDA-capable GPU."
        )
        return

    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print()

    # Configuration
    trace_dir = "traces"  # Directory for trace files

    # Model setup and data generation
    config, real_inputs, real_targets = setup_model_and_data(device)

    # Placeholders for graph capture
    static_input = torch.randn(config['N'], config['D_in'], device=device)
    static_target = torch.randn(config['N'], config['D_out'], device=device)

    # ========================================================================
    # Training WITHOUT CUDA Graph
    # ========================================================================
    print("=" * 70)
    print("SCENARIO 1: Training WITHOUT CUDA Graph")
    print("=" * 70)

    model_no_graph = create_model(config, device)
    loss_fn_no_graph = torch.nn.MSELoss()
    optimizer_no_graph = torch.optim.SGD(model_no_graph.parameters(), lr=0.1)

    with create_profiler() as prof_no_graph:
        train_without_cuda_graph(model_no_graph,
                                 loss_fn_no_graph,
                                 optimizer_no_graph,
                                 real_inputs,
                                 real_targets,
                                 profiler=prof_no_graph)

    # Save profiling trace and print summary
    trace_file_no_graph = trace_dir + "/" + "trace_without_manual_capture.json"
    save_and_print_profile(prof_no_graph, trace_file_no_graph,
                           "without CUDA graph")

    # ========================================================================
    # Training WITH CUDA Graph
    # ========================================================================
    print("=" * 70)
    print("SCENARIO 2: Training WITH CUDA Graph")
    print("=" * 70)

    model_with_graph = create_model(config, device)
    loss_fn_with_graph = torch.nn.MSELoss()
    optimizer_with_graph = torch.optim.SGD(model_with_graph.parameters(),
                                           lr=0.1)

    # Prepare graph (warmup + capture) - NOT profiled
    print("Preparing CUDA graph (warmup + capture)...")
    graph, static_loss = prepare_cuda_graph(model_with_graph,
                                            loss_fn_with_graph,
                                            optimizer_with_graph, static_input,
                                            static_target)
    print("CUDA graph ready.")
    print()

    # Profile only the training iterations
    with create_profiler() as prof_with_graph:
        train_with_cuda_graph(graph,
                              real_inputs,
                              real_targets,
                              static_input,
                              static_target,
                              static_loss,
                              profiler=prof_with_graph)

    # Save profiling trace and print summary
    trace_file_with_graph = trace_dir + "/" + "trace_with_manual_capture.json"
    save_and_print_profile(prof_with_graph, trace_file_with_graph,
                           "with CUDA graph")

    print("=" * 70)
    print("Profiling completed successfully!")
    print(f"View traces in Chrome: chrome://tracing")
    print(f"  - {trace_file_no_graph}")
    print(f"  - {trace_file_with_graph}")
    print("=" * 70)


if __name__ == "__main__":
    main()
