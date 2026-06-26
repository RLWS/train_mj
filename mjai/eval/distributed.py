from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from mjai.utils.device import normalize_device_name


T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class EvalDistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_eval_distributed_context(config: dict[str, Any]) -> EvalDistributedContext:
    eval_config = config.get("eval", {})
    requested = bool(eval_config.get("distributed", False))
    env_world_size = _env_int("WORLD_SIZE", 1)
    env_rank = _env_int("RANK", 0)
    env_local_rank = _env_int("LOCAL_RANK", 0)

    if not requested:
        if env_world_size > 1:
            raise RuntimeError(
                "torchrun launched multiple eval processes, but eval.distributed is false"
            )
        return EvalDistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend=None,
            device=_resolve_eval_device(config.get("train", {}).get("device")),
        )

    if env_world_size <= 1:
        raise RuntimeError(
            "eval.distributed=true requires torchrun with WORLD_SIZE > 1, "
            "for example: torchrun --standalone --nproc_per_node=2 -m mjai.cli eval ..."
        )
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    device = _resolve_eval_device(config.get("train", {}).get("device"), local_rank=env_local_rank)
    backend = _resolve_eval_backend(eval_config, device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return EvalDistributedContext(
        enabled=True,
        rank=dist.get_rank() if dist.is_initialized() else env_rank,
        local_rank=env_local_rank,
        world_size=dist.get_world_size() if dist.is_initialized() else env_world_size,
        backend=backend,
        device=device,
    )


def gather_eval_objects(value: T, context: EvalDistributedContext) -> list[T]:
    if not context.enabled or not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[T | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


def shutdown_eval_distributed_context(context: EvalDistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _resolve_eval_device(config_device: str | None, *, local_rank: int | None = None) -> torch.device:
    normalized = normalize_device_name(config_device)
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_eval_backend(eval_config: dict[str, Any], device: torch.device) -> str:
    backend = str(eval_config.get("backend", "auto")).lower()
    if backend in {"auto", "vendor"}:
        # Offline eval does not wrap models in DDP or use device collectives; it only
        # needs a control-plane group for small Python-object metric summaries.
        # Gloo is more reliable here than vendor GPU collectives for all_gather_object.
        return "gloo"
    return backend


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default
