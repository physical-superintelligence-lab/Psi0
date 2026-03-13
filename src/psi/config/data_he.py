from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from psi.config.config import DataConfig

class HERawDataConfig(DataConfig):
    root_dir: str
    robot_type: str = "both"
    episodes: Optional[List[int]] = None
    chunk_size: int = 1
    use_delta_actions: bool = True
    upsample_rate: int = 1

    def __call__(self, split: str = "train", transform_kwargs=None, **kwargs) -> Any:
        if transform_kwargs is None:
            transform_kwargs = {}
        from psi.data.humanoid.he_raw_dataset import HERawDataset
        from psi.data.dataset import Dataset as MapStyleDataset
        
        repack = self.transform.repack
        num_past_frames = getattr(repack, "num_past_frames", 0)
        action_chunk_size = getattr(repack, "action_chunk_size", 1)
        use_delta_actions = getattr(repack, "use_delta_actions", False)
        self.chunk_size = int(action_chunk_size)
        self.use_delta_actions = bool(use_delta_actions)
        dataset = HERawDataset(
            data_root=self.root_dir,
            num_past_frames=num_past_frames,
            action_chunk_size=action_chunk_size,
            upsample_rate=self.upsample_rate,
            use_delta_actions=use_delta_actions,
            robot_type=self.robot_type,
            episodes=self.episodes,
        )
        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs, **kwargs)