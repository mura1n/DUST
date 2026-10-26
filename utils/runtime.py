import os
from typing import Mapping, MutableMapping, Optional


DRIVESTUDIO_THREAD_ENV = "DRIVESTUDIO_CPU_THREADS"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_INTEROP_THREADS_CONFIGURED = False


def _parse_thread_limit(raw_value: str, env_name: str) -> int:
    try:
        limit = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from error
    if limit < 1:
        raise ValueError(f"{env_name} must be >= 1, got {limit}")
    return limit


def get_requested_cpu_threads(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    env = os.environ if environ is None else environ
    for env_name in (DRIVESTUDIO_THREAD_ENV, *THREAD_ENV_VARS):
        raw_value = env.get(env_name)
        if raw_value is None or raw_value == "":
            continue
        return _parse_thread_limit(raw_value, env_name)
    return None


def apply_thread_env_defaults(environ: Optional[MutableMapping[str, str]] = None) -> Optional[int]:
    env = os.environ if environ is None else environ
    limit = get_requested_cpu_threads(env)
    if limit is None:
        return None

    limit_value = str(limit)
    for env_name in THREAD_ENV_VARS:
        env.setdefault(env_name, limit_value)
    return limit


def configure_runtime_threads(limit: Optional[int] = None) -> Optional[int]:
    global _INTEROP_THREADS_CONFIGURED

    resolved_limit = get_requested_cpu_threads() if limit is None else limit
    if resolved_limit is None:
        return None

    import torch

    torch.set_num_threads(resolved_limit)
    if not _INTEROP_THREADS_CONFIGURED:
        torch.set_num_interop_threads(max(1, min(resolved_limit, 4)))
        _INTEROP_THREADS_CONFIGURED = True

    try:
        import cv2

        cv2.setNumThreads(resolved_limit)
    except Exception:
        pass

    return resolved_limit
