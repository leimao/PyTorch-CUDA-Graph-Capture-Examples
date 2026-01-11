"""
Common utilities for CUDA graph examples.

This module contains shared model definitions and training functions used
across different CUDA graph demonstration scripts.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, record_function


class MLPBlock(nn.Module):
    """Single MLP block with Linear, ReLU, and Dropout."""

    def __init__(self, in_features, out_features, dropout_p=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MLPModel(nn.Module):
    """MLP model with three consecutive MLP blocks plus a final linear layer."""

    def __init__(self,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 hidden_dim3,
                 output_dim,
                 dropout_p1=0.2,
                 dropout_p2=0.1,
                 dropout_p3=0.1):
        super().__init__()
        self.block1 = MLPBlock(input_dim, hidden_dim1, dropout_p=dropout_p1)
        self.block2 = MLPBlock(hidden_dim1, hidden_dim2, dropout_p=dropout_p2)
        self.block3 = MLPBlock(hidden_dim2, hidden_dim3, dropout_p=dropout_p3)
        self.output = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x


def train_without_cuda_graph(model,
                             loss_fn,
                             optimizer,
                             inputs,
                             targets,
                             profiler=None):
    """Train without using CUDA graph (standard PyTorch training)."""
    print("Training WITHOUT CUDA graph...")

    for i, (data, target) in enumerate(zip(inputs, targets)):
        with record_function("## optimizer.zero_grad ##"):
            optimizer.zero_grad()

        with record_function("## forward_pass ##"):
            y_pred = model(data)

        with record_function("## loss_computation ##"):
            loss = loss_fn(y_pred, target)

        with record_function("## backward_pass ##"):
            loss.backward()

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


def setup_model_and_data(device):
    """Setup model configuration and generate training data."""
    # Model setup
    N, D_in, H1, H2, H3, D_out = 640, 4096, 2048, 1024, 512, 256
    print(f"Model configuration:")
    print(f"  Batch size: {N}")
    print(f"  Input dim: {D_in}")
    print(f"  Hidden dims: {H1} -> {H2} -> {H3}")
    print(f"  Output dim: {D_out}")
    print()

    # Generate training data
    num_iterations = 10
    real_inputs = [
        torch.randn(N, D_in, device=device) for _ in range(num_iterations)
    ]
    real_targets = [
        torch.randn(N, D_out, device=device) for _ in range(num_iterations)
    ]

    config = {
        'N': N,
        'D_in': D_in,
        'H1': H1,
        'H2': H2,
        'H3': H3,
        'D_out': D_out
    }

    return config, real_inputs, real_targets


def create_model(config, device):
    """Create a new MLPModel instance with the given configuration."""
    return MLPModel(input_dim=config['D_in'],
                    hidden_dim1=config['H1'],
                    hidden_dim2=config['H2'],
                    hidden_dim3=config['H3'],
                    output_dim=config['D_out'],
                    dropout_p1=0.2,
                    dropout_p2=0.1,
                    dropout_p3=0.1).to(device)


def create_profiler():
    """Create a profiler with standard configuration."""
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=1, warmup=2, active=7, repeat=1),
                   record_shapes=True,
                   profile_memory=True,
                   with_stack=True)


def save_and_print_profile(prof, trace_file, scenario_name):
    """Save profiling trace and print summary."""
    prof.export_chrome_trace(trace_file)
    print(f"Profiling trace saved to: {trace_file}")
    print()

    print(f"Top 10 operations by CUDA time ({scenario_name}):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
