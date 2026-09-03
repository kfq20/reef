"""Runtime adapters for external model services.

Importing this package registers every bundled runtime kind. Each adapter
module must therefore stay importable without its execution dependencies:
``ray_runtime`` does not import Ray at module scope, and the MLX runtime does
not import MLX. The heavy import happens inside the factory, when a
deployment actually selects that kind.
"""

from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
    RayRuntime,
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)
from reef.train.mlx_backend.runtime import MLXRuntime, MLXRuntimeFactory

__all__ = [
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "MLXRuntime",
    "MLXRuntimeFactory",
    "RayRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "connect_ray_runtime",
]
