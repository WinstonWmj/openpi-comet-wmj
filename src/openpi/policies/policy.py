from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
from pathlib import Path

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _to_cpu(obj):
    """Recursively move tensors/arrays to CPU numpy (safe to torch.save)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    # jax.Array / jnp.ndarray / numpy types
    if hasattr(obj, "__array__"):
        try:
            return np.asarray(obj)
        except Exception:
            pass
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    # last resort
    return repr(obj)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        debug_dump_dir: str = "/mnt/public/chenjiawei/projects/behavior_sft_dev/RLinf/obs",
        debug_dump_max: int = 5,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # ---- Debug: obs loading for offline replay ----
        self._obs_debug_enabled = True  # 设为 True 启用加载
        self._obs_debug_dir = "/mnt/public/chenjiawei/projects/behavior_sft_dev/RLinf/obs"
        self._obs_debug_files = []
        self._obs_debug_idx = 0
        self._obs_debug_initialized = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _init_obs_loader(self):
        """初始化 obs 加载器，扫描目录中 obs_inputs_*.pt 文件并按序号排序"""
        obs_path = Path(self._obs_debug_dir)
        # 文件名格式: obs_inputs_0000_torch_xxx.pt，按 inputs 后面的数字排序
        self._obs_debug_files = sorted(
            obs_path.glob("obs_inputs_*.pt"),
            key=lambda x: int(x.stem.split("_")[2])  # 取 0000, 0001, ...
        )
        self._obs_debug_idx = 0
        self._obs_debug_initialized = True
        print(f"[DEBUG] Found {len(self._obs_debug_files)} obs files in {self._obs_debug_dir}")

    def _load_next_processed_obs(self) -> dict | None:
        """加载下一个保存的 obs，文件用完返回 None"""
        if not self._obs_debug_initialized:
            self._init_obs_loader()
        
        if self._obs_debug_idx >= len(self._obs_debug_files):
            if self._obs_debug_idx == len(self._obs_debug_files):
                print(f"[DEBUG] All {len(self._obs_debug_files)} obs files used, switching to live obs")
                self._obs_debug_idx += 1
            return None
        
        file_path = self._obs_debug_files[self._obs_debug_idx]
        saved_obs = torch.load(file_path, map_location="cuda", weights_only=False)
        self._obs_debug_idx += 1
        print(f"[DEBUG] Loaded obs {self._obs_debug_idx}/{len(self._obs_debug_files)}: {file_path.name}")
        # 保存的格式是 {'inputs': {...}}，取出 inputs
        inputs = saved_obs['inputs']
        # 把 numpy array 转成 torch tensor 并移到 cuda
        return self._convert_numpy_to_tensor(inputs)
    
    def _convert_numpy_to_tensor(self, data, device="cuda"):
        """递归将 dict 中的 numpy array 转成 torch tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        elif isinstance(data, dict):
            return {k: self._convert_numpy_to_tensor(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_tensor(v, device) for v in data]
        else:
            return data

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        # ====== dump the inputs that will be fed into Observation.from_dict ======
        # if self._debug_dump_count < self._debug_dump_max:
        #     ts = int(time.time() * 1000)
        #     backend = "torch" if self._is_pytorch_model else "jax"
        #     path = self._debug_dump_dir / f"obs_inputs_{self._debug_dump_count:04d}_{backend}_{ts}.pt"
        #     torch.save({"inputs": _to_cpu(inputs)}, path)
        #     print(f"[DEBUG] saved: {path}")
        #     self._debug_dump_count += 1
        # =======================================================================
        # ---- Debug: 优先加载保存的 processed_obs，用完后切换到实时 obs ----
        if self._obs_debug_enabled:
            loaded_obs = self._load_next_processed_obs()
            if loaded_obs is not None:
                inputs = loaded_obs
        # ---- Debug end ----

        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        print(f"output_actions: {outputs['actions']}")
        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results