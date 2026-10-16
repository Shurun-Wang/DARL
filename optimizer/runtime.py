import time
import copy

import torch

from utils import get_model_complexity_info

_complexity_cache = {}
_cache_hits = 0
_cache_misses = 0


def reset_complexity_cache():
    global _cache_hits, _cache_misses
    _complexity_cache.clear()
    _cache_hits = _cache_misses = 0


def complexity_cache_info():
    return dict(hits=_cache_hits, misses=_cache_misses, entries=len(_complexity_cache),
                profiling_device='cpu', lifetime='one HPO run')


def synchronize_device(device):
    """Wait for queued accelerator work so wall-clock timings are accurate."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def start_timer(device):
    synchronize_device(device)
    return time.perf_counter()


def stop_timer(started_at, device):
    synchronize_device(device)
    return time.perf_counter() - started_at


def measure_model_complexity(model, ch_num, device):
    """Return raw model size/FLOPs data; preserve the trial if profiling alone fails."""
    global _cache_hits, _cache_misses
    architecture = getattr(model, '_complexity_key', None)
    key = (architecture, (1, ch_num, 2000), str(next(model.parameters()).dtype))
    if architecture is not None and key in _complexity_cache:
        _cache_hits += 1
        return _complexity_cache[key]
    _cache_misses += 1
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    try:
        # Profile a disposable CPU copy: THOP hooks/buffers (including failure
        # leftovers) cannot alter training. Cache hits must not change RNG state.
        with torch.random.fork_rng(devices=[]):
            profile_model = copy.deepcopy(model).cpu().eval()
            _, flops_g = get_model_complexity_info(
                profile_model, input_shape=(1, ch_num, 2000), device='cpu')
        result = float(params_m), float(flops_g), True
        if architecture is not None and len(_complexity_cache) < 4096:
            _complexity_cache[key] = result
        return result
    except Exception:
        return float(params_m), None, False
