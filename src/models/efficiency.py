"""FLOPs/param/latency tracking, built in alongside the first model rather
than bolted on later -- [redacted: private project notes]'s engineering priorities: "There's a
[redacted: detail about the data challenge this code was written for]
tracking in from day one." No external profiling dependency (fvcore/thop/
ptflops) -- MAC counting for Conv3d/Linear is a small, hand-verifiable
amount of arithmetic (see tests/test_efficiency.py's hand-computed cases),
consistent with this project's general preference for dependency-light,
directly-tested code over an unverified library.

MACs (multiply-accumulate operations), not "FLOPs", is what's actually
counted -- multiply by 2 for a FLOPs estimate under the common
1 MAC = 2 FLOPs convention, if that's the number wanted; MACs is reported as
the primal number since it maps directly to the arithmetic actually done.
"""

import time

import torch
import torch.nn as nn


def _conv3d_macs(module, output):
    out_channels = output.shape[1]
    spatial = 1
    for s in output.shape[2:]:
        spatial *= s
    batch = output.shape[0]
    kd, kh, kw = module.kernel_size
    in_per_group = module.in_channels // module.groups
    return batch * out_channels * spatial * in_per_group * kd * kh * kw


def _linear_macs(module, output):
    return output.numel() * module.in_features


_COUNTERS = {nn.Conv3d: _conv3d_macs, nn.Linear: _linear_macs}


def count_macs(model, forward_fn):
    """Total multiply-accumulate count for one call to `forward_fn()` (a
    zero-arg callable that runs `model` on example inputs, e.g.
    `lambda: model(image, plane_present)`), via forward hooks on every
    Conv3d/Linear submodule. Runs in eval mode with no_grad and restores the
    model's original training-mode flag afterward.

    Only Conv3d and Linear are counted -- BatchNorm/ReLU/pooling are
    negligible next to conv/linear MACs and are left out rather than adding
    unverified estimates for them.
    """
    was_training = model.training
    model.eval()
    total = {"macs": 0}
    handles = []

    def make_hook(counter_fn):
        def hook(module, inputs, output):
            total["macs"] += counter_fn(module, output)
        return hook

    for module in model.modules():
        counter_fn = _COUNTERS.get(type(module))
        if counter_fn is not None:
            handles.append(module.register_forward_hook(make_hook(counter_fn)))

    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    return total["macs"]


def measure_latency(forward_fn, device=None, n_warmup=3, n_repeats=10):
    """Wall-clock latency of repeated calls to `forward_fn()` (a zero-arg
    callable running one forward pass), in milliseconds. Runs `n_warmup`
    untimed calls first (first-call overhead -- cuDNN autotuning, lazy
    allocation -- would otherwise dominate a small n_repeats). Synchronizes
    the device before/after each timed call so async CUDA/MPS dispatch
    doesn't make the timing meaningless.

    Returns {"mean_ms", "std_ms", "min_ms", "max_ms", "n_repeats"}.
    """
    def _sync():
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            forward_fn()
            _sync()

        times_ms = []
        for _ in range(n_repeats):
            _sync()
            start = time.perf_counter()
            forward_fn()
            _sync()
            times_ms.append((time.perf_counter() - start) * 1000)

    times = torch.tensor(times_ms)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std(unbiased=False).item() if len(times) > 1 else 0.0,
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "n_repeats": n_repeats,
    }
