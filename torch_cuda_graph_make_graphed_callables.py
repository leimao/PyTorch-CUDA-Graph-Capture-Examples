#!/usr/bin/env python3
"""
CUDA Graph with make_graphed_callables Example

This script demonstrates how to use make_graphed_callables to automatically capture
and replay CUDA graphs for model forward and backward passes.
It profiles training with and without CUDA graphs for comparison.

make_graphed_callables simplifies CUDA graph usage by automatically handling warmup
and capture. It wraps individual callables (like models) and graphs their forward
and backward operations, while leaving loss computation and optimizer steps outside
the graph.

Three scenarios are demonstrated:
1. Training WITHOUT CUDA graph (baseline)
2. Training WITH full CUDA graph (entire model)
3. Training WITH partial CUDA graph (only block2 submodule)
"""

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.cuda import make_graphed_callables
from common import (MLPModel, train_without_cuda_graph, setup_model_and_data,
                    create_model, create_profiler, save_and_print_profile)


def prepare_cuda_graph(model, static_input):
    """Prepare CUDA graph using make_graphed_callables (not profiled)."""
    print("  Creating graphed model...")

    # Wrap the model with make_graphed_callables
    # This will graph the forward and backward passes of the model
    graphed_model = make_graphed_callables(model, (static_input, ))

    print("  CUDA graph model ready.")

    return graphed_model


def prepare_partial_cuda_graph(model, static_input):
    """Prepare CUDA graph for only block2 submodule (not profiled)."""
    print("  Creating partially graphed model (only block2)...")

    # First, do a forward pass to determine the input shape for block2
    with torch.no_grad():
        block1_output = model.block1(static_input)

    # Wrap only block2 with make_graphed_callables
    # This will graph only the forward and backward passes of block2
    # When passing a single callable, it returns the graphed callable directly
    graphed_block2 = make_graphed_callables(model.block2, (block1_output, ))
    model.block2 = graphed_block2

    print("  CUDA graph for block2 ready.")

    return model


def train_with_cuda_graph(graphed_model,
                          loss_fn,
                          optimizer,
                          inputs,
                          targets,
                          profiler=None):
    """Train using CUDA graph model for optimized replay (profiled part only)."""
    print("  Training with graph replay...")

    for i, (data, target) in enumerate(zip(inputs, targets)):
        with record_function("## optimizer.zero_grad ##"):
            optimizer.zero_grad(set_to_none=True)

        # Forward pass runs as a graph
        with record_function("## forward_pass_graphed ##"):
            y_pred = graphed_model(data)

        # NOTE: Loss computation is NOT part of the CUDA graph
        # Only the model's forward/backward passes are graphed
        with record_function("## loss_computation ##"):
            loss = loss_fn(y_pred, target)

        # Backward pass runs as a graph
        with record_function("## backward_pass_graphed ##"):
            loss.backward()

        # NOTE: Optimizer step is NOT part of the CUDA graph
        with record_function("## optimizer.step ##"):
            optimizer.step()

        if profiler is not None:
            profiler.step()

        # NOTE: Avoid calling .item() in the training loop as it triggers device-to-host
        # memory copy and CPU-GPU synchronization, which damages performance.
        # if i % 2 == 0:
        #     print(f"  Iteration {i+1:2d}: Loss = {loss.item():.4f}")

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
    trace_file_no_graph = trace_dir + "/" + "trace_without_make_graphed_callables.json"
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
    graphed_model = prepare_cuda_graph(model_with_graph, static_input)
    print("CUDA graph ready.")
    print()

    # Profile only the training iterations
    with create_profiler() as prof_with_graph:
        train_with_cuda_graph(graphed_model,
                              loss_fn_with_graph,
                              optimizer_with_graph,
                              real_inputs,
                              real_targets,
                              profiler=prof_with_graph)

    # Save profiling trace and print summary
    trace_file_with_graph = trace_dir + "/" + "trace_with_make_graphed_callables.json"
    save_and_print_profile(prof_with_graph, trace_file_with_graph,
                           "with CUDA graph")

    print("=" * 70)
    print("Profiling completed successfully!")
    print(f"View traces in Chrome: chrome://tracing")
    print(f"  - {trace_file_no_graph}")
    print(f"  - {trace_file_with_graph}")
    print("=" * 70)

    # ========================================================================
    # Training WITH PARTIAL CUDA Graph (only block2)
    # ========================================================================
    print()
    print("=" * 70)
    print("SCENARIO 3: Training WITH PARTIAL CUDA Graph (only block2)")
    print("=" * 70)

    model_partial_graph = create_model(config, device)
    loss_fn_partial_graph = torch.nn.MSELoss()
    optimizer_partial_graph = torch.optim.SGD(model_partial_graph.parameters(),
                                              lr=0.1)

    # Prepare partial graph (only block2) - NOT profiled
    print("Preparing CUDA graph for block2 only (warmup + capture)...")
    model_partial_graph = prepare_partial_cuda_graph(model_partial_graph,
                                                     static_input)
    print("CUDA graph for block2 ready.")
    print()

    # Profile only the training iterations
    with create_profiler() as prof_partial_graph:
        train_with_cuda_graph(model_partial_graph,
                              loss_fn_partial_graph,
                              optimizer_partial_graph,
                              real_inputs,
                              real_targets,
                              profiler=prof_partial_graph)

    # Save profiling trace and print summary
    trace_file_partial_graph = trace_dir + "/" + "trace_with_partial_make_graphed_callables.json"
    save_and_print_profile(prof_partial_graph, trace_file_partial_graph,
                           "with partial CUDA graph - block2 only")

    print("=" * 70)
    print("All profiling completed successfully!")
    print(f"View traces in Chrome: chrome://tracing")
    print(f"  - {trace_file_no_graph}")
    print(f"  - {trace_file_with_graph}")
    print(f"  - {trace_file_partial_graph}")
    print("=" * 70)


if __name__ == "__main__":
    main()
