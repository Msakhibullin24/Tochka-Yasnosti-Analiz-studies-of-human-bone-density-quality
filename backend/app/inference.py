from __future__ import annotations

from pathlib import Path
from threading import Lock

from .keypoints import normalize_landmarks
from .settings import BACKEND_ROOT, settings


class ModelUnavailable(RuntimeError):
    pass


class DxaPointPlacementModel:
    def __init__(self) -> None:
        self._model = None
        self._lock = Lock()
        self._device = "cpu"

    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if not settings.checkpoint_path.is_file():
                raise ModelUnavailable(
                    f"Checkpoint not found at {settings.checkpoint_path}. Run scripts/fetch_dxa_checkpoint.py."
                )
            try:
                from mmengine import Config
                from mmpose.apis import init_model
            except ImportError as error:
                raise ModelUnavailable(
                    "MMPose runtime is not installed. Install the backend 'ml' dependencies."
                ) from error

            config_path = BACKEND_ROOT / "third_party" / "dxa_pointplacement" / "dxa_points_config.py"
            metainfo_path = BACKEND_ROOT / "third_party" / "dxa_pointplacement" / "dxa_points.py"
            config = Config.fromfile(str(config_path))
            config.model.init_cfg = None
            for loader_name in ("test_dataloader", "val_dataloader"):
                loader = config.get(loader_name)
                if loader and loader.get("dataset"):
                    loader.dataset.metainfo = {"from_file": str(metainfo_path)}
            self._device = self._resolve_device()
            self._model = init_model(config, str(settings.checkpoint_path), device=self._device)
            return self._model

    @property
    def device(self) -> str:
        return self._device

    def predict(self, image):
        try:
            import numpy as np
            from mmpose.apis import inference_topdown
        except ImportError as error:
            raise ModelUnavailable("MMPose inference dependencies are not installed") from error

        model = self._load()
        rgb = np.repeat(image[:, :, None], 3, axis=2)
        with self._lock:
            results = inference_topdown(model, rgb)
        if not results:
            raise RuntimeError("DXA point-placement model returned no prediction")
        instances = results[0].pred_instances
        points = instances.keypoints[0]
        scores = instances.keypoint_scores[0] if hasattr(instances, "keypoint_scores") else None
        return normalize_landmarks(points, scores, image.shape[1], image.shape[0])


model = DxaPointPlacementModel()
