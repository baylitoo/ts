from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional

import numpy as np
import torch

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


def _percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


class RuntimeMetrics:
    """Utility to capture latency and resource metrics during training/eval."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._timings: DefaultDict[str, List[float]] = defaultdict(list)
        self._values: DefaultDict[str, List[float]] = defaultdict(list)
        self.wall_clock_start: float = time.perf_counter()
        self._nvml_initialised: bool = False
        self._nvml_device_handle: Any = None
        self._nvml_failed: bool = False

    @contextmanager
    def track_time(self, name: str) -> Any:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._timings[name].append(duration)

    def record_value(self, name: str, value: float) -> None:
        if math.isnan(value) or math.isinf(value):
            return
        self._values[name].append(float(value))

    def snapshot_memory(self, tag: str = "gpu") -> None:
        if not torch.cuda.is_available():
            return
        device = torch.device(self.device) if self.device != "auto" else torch.device("cuda")
        try:
            allocated = torch.cuda.memory_allocated(device) / (1024**2)
            reserved = torch.cuda.memory_reserved(device) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        except RuntimeError:
            return
        self._values[f"{tag}_mem_allocated_mb"].append(float(allocated))
        self._values[f"{tag}_mem_reserved_mb"].append(float(reserved))
        self._values[f"{tag}_mem_max_allocated_mb"].append(float(max_allocated))

        nvml_stats = self._sample_nvml_utilisation()
        if nvml_stats:
            for key, value in nvml_stats.items():
                self._values[key].append(value)

    def summarize(self) -> Dict[str, Any]:
        total_wall_clock = time.perf_counter() - self.wall_clock_start
        summary: Dict[str, Any] = {"total_wall_clock_sec": total_wall_clock}

        timing_summary: Dict[str, Dict[str, float]] = {}
        for name, entries in self._timings.items():
            timing_summary[name] = {
                "count": len(entries),
                "mean_sec": float(np.mean(entries)) if entries else float("nan"),
                "p50_sec": _percentile(entries, 50.0),
                "p90_sec": _percentile(entries, 90.0),
                "p95_sec": _percentile(entries, 95.0),
                "p99_sec": _percentile(entries, 99.0),
            }
        summary["timings"] = timing_summary

        value_summary: Dict[str, Dict[str, float]] = {}
        for name, entries in self._values.items():
            value_summary[name] = {
                "count": len(entries),
                "mean": float(np.mean(entries)) if entries else float("nan"),
                "p50": _percentile(entries, 50.0),
                "p90": _percentile(entries, 90.0),
                "p95": _percentile(entries, 95.0),
                "p99": _percentile(entries, 99.0),
            }
        summary["values"] = value_summary
        return summary

    def dump(self, path: Path) -> None:
        payload = self.summarize()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_nvml_utilisation(self) -> Optional[Dict[str, float]]:
        """Fetch GPU utilisation/temperature metrics via NVML if available."""
        if pynvml is None:
            return None
        if self._nvml_failed:
            return None

        if not self._nvml_initialised:
            if not self._initialise_nvml():
                self._nvml_failed = True
                return None

        try:
            assert self._nvml_device_handle is not None
            utilisation = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_device_handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                self._nvml_device_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            clock = pynvml.nvmlDeviceGetClockInfo(
                self._nvml_device_handle, pynvml.NVML_CLOCK_SM
            )
        except Exception:
            self._nvml_failed = True
            return None

        return {
            "gpu_util_percent": float(utilisation.gpu),
            "gpu_mem_util_percent": float(utilisation.memory),
            "gpu_temperature_c": float(temperature),
            "gpu_sm_clock_mhz": float(clock),
        }

    def _initialise_nvml(self) -> bool:
        """Initialise NVML for the configured CUDA device."""
        if pynvml is None:
            return False
        if not torch.cuda.is_available():
            return False

        # Determine device index from configuration.
        device_index: int
        if self.device == "auto":
            device_index = torch.cuda.current_device()
        elif self.device.startswith("cuda"):
            try:
                device_index = int(self.device.split(":")[1])
            except (IndexError, ValueError):
                device_index = torch.cuda.current_device()
        else:
            device_index = torch.cuda.current_device()

        try:
            pynvml.nvmlInit()
            self._nvml_device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_initialised = True
        except Exception:
            self._nvml_device_handle = None
            self._nvml_initialised = False
            return False

        return True
