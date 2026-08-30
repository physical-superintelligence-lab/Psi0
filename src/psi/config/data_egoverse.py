from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, TYPE_CHECKING
from psi.config.config import DataConfig
if TYPE_CHECKING:
    from psi.data.dataset import TransformableDataset

class EgoVerseDataConfig(DataConfig):
    root_dir: str
    upsample_rate: int = 3 # actually it means downsampling factor, i.e. stride 
    chunk_size: int = 16
    robots: List[str] = Field(default_factory=lambda: ["aria_bimanual", "mecka_bimanual", "scale_bimanual"])
    # transform_mode: str = "keypoints_headframe_quat"
    valid_ratio: float = 0.01
    use_delta_actions: bool = True

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> TransformableDataset:
        from psi.data.egoverse.egoverse_dataset import EgoVerseDataset
        from psi.data.dataset import Dataset as MapStyleDataset
        
        dataset = EgoVerseDataset(
            local_dir=self.root_dir,
            # transform_mode="keypoints_headframe_quat",
            robots=self.robots,
            split=split,
            chunk_size=self.chunk_size,
            valid_ratio=self.valid_ratio,
            stride=self.upsample_rate,
        )
        return MapStyleDataset(
            self, dataset, transform_kwargs=transform_kwargs, **kwargs
        )

    def mock(self, split: str = "train", **kwargs) -> Any:
        # NOT TESTED YET
        import base64, importlib.util, io
        from pathlib import Path
        import numpy as np
        import torch
        import cv2

        # _egoverse_mock_data.py lives at <repo_root>/data/mock/, outside the src tree.
        _mock_path = Path(__file__).resolve().parents[3] / "data" / "mock" / "_egoverse_mock_data.py"
        _spec = importlib.util.spec_from_file_location("_egoverse_mock_data", _mock_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        _MOCK_OBSERVATIONS_IMAGES_FRONT_IMG_1 = _mod._MOCK_OBSERVATIONS_IMAGES_FRONT_IMG_1
        _MOCK_ACTIONS_KEYPOINTS               = _mod._MOCK_ACTIONS_KEYPOINTS
        _MOCK_OBSERVATIONS_STATE_KEYPOINTS    = _mod._MOCK_OBSERVATIONS_STATE_KEYPOINTS
        _MOCK_METADATA_ROBOT_NAME             = _mod._MOCK_METADATA_ROBOT_NAME
        _MOCK_EMBODIMENT                      = _mod._MOCK_EMBODIMENT

        # decode image from JPEG base64
        jpg_bytes = base64.b64decode(_MOCK_OBSERVATIONS_IMAGES_FRONT_IMG_1)
        img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]

        # decode tensors from npy base64
        def _decode_npy(b64: str) -> torch.Tensor:
            arr = np.load(io.BytesIO(base64.b64decode(b64)))
            return torch.from_numpy(arr).unsqueeze(0)

        return {
            "observations.images.front_img_1": image,
            "actions_keypoints":               _decode_npy(_MOCK_ACTIONS_KEYPOINTS),
            "observations.state.keypoints":    _decode_npy(_MOCK_OBSERVATIONS_STATE_KEYPOINTS),
            "metadata.robot_name":             [_MOCK_METADATA_ROBOT_NAME],
            "embodiment":                      [_MOCK_EMBODIMENT],
        }
