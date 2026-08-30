import copy
from abc import ABC
from enum import Enum
from typing import Literal

import numpy as np
import torch

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.utils.type_utils import _to_numpy


class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    EVA_RIGHT_ARM = 6
    EVA_LEFT_ARM = 7
    EVA_BIMANUAL = 8
    MECKA_BIMANUAL = 9
    MECKA_RIGHT_ARM = 10
    MECKA_LEFT_ARM = 11
    SCALE_BIMANUAL = 12
    SCALE_RIGHT_ARM = 13
    SCALE_LEFT_ARM = 14


EMBODIMENT_ID_TO_KEY = {
    member.value: key for key, member in EMBODIMENT.__members__.items()
}


def get_embodiment(index):
    return EMBODIMENT_ID_TO_KEY.get(index, None)


def get_embodiment_id(embodiment_name):
    embodiment_name = embodiment_name.upper()
    return EMBODIMENT[embodiment_name].value


class Embodiment(ABC):
    """Base embodiment class. An embodiment is responsible for defining the transform pipeline that converts between the raw data in the dataset and the canonical representation used by the model."""

    @staticmethod
    def get_transform_list() -> list[Transform]:
        """Returns the list of transforms that convert between the raw data in the dataset and the canonical representation used by the model."""
        raise NotImplementedError

    @staticmethod
    def viz_transformed_batch(batch):
        """Visualizes a batch of transformed data."""
        raise NotImplementedError

    @staticmethod
    def get_keymap():
        """Returns a dictionary mapping from the raw keys in the dataset to the canonical keys used by the model."""
        raise NotImplementedError

    @classmethod
    def viz_gt_preds(
        cls,
        predictions,
        batch,
        image_key,
        action_key,
        transform_list=None,
        mode=Literal["traj", "axes", "keypoints"],
        gt_alpha=1.0,
        pred_alpha=0.7,
        **kwargs,
    ):
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()

        pred_actions = predictions[
            f"{embodiment_name}_{action_key}"
        ]  # TODO: make this work with groundtruth, clone batch and replace actions_keypoints with pred_actions
        if transform_list is not None:
            pred_batch = copy.deepcopy(batch)
            pred_batch[action_key] = pred_actions
            batch = cls.apply_transform(batch, transform_list)
            pred_batch = cls.apply_transform(pred_batch, transform_list)
            pred_actions = pred_batch[action_key]

        images = batch[image_key]
        actions = batch[action_key]
        ims_list = []
        images = _to_numpy(images)
        actions = _to_numpy(actions)
        pred_actions = _to_numpy(pred_actions)
        for i in range(images.shape[0]):
            image = images[i]
            action = actions[i]
            pred_action = pred_actions[i]
            ims = cls.viz(
                image, action, mode=mode, color="Reds", alpha=gt_alpha, **kwargs
            )
            ims = cls.viz(
                ims, pred_action, mode=mode, color="Greens", alpha=pred_alpha, **kwargs
            )
            ims_list.append(ims)
        ims = np.stack(ims_list, axis=0)
        return ims

    @classmethod
    def apply_transform(cls, batch, transform_list: list[Transform]):
        if transform_list:
            batch_size = None
            for v in batch.values():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    batch_size = v.shape[0]
                    break

            if batch_size is not None:
                # Apply transforms per-sample (matching how ZarrDataset applies them)
                results = []
                for i in range(batch_size):
                    sample = {}
                    for k, v in batch.items():
                        if (
                            isinstance(v, (np.ndarray, torch.Tensor))
                            and v.shape[0] == batch_size
                        ):
                            sample[k] = (
                                v[i].cpu().numpy()
                                if isinstance(v, torch.Tensor)
                                else v[i]
                            )
                        else:
                            continue

                    for transform in transform_list:
                        sample = transform.transform(sample)
                    results.append(sample)

                batch = {}
                for k in results[0]:
                    vals = [r[k] for r in results]
                    if isinstance(vals[0], np.ndarray):
                        batch[k] = np.stack(vals, axis=0)
                    elif isinstance(vals[0], torch.Tensor):
                        batch[k] = torch.stack(vals, dim=0)
                    else:
                        batch[k] = vals
            else:
                for transform in transform_list:
                    batch = transform.transform(batch)

        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(torch.float32)

        return batch
