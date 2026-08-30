import numpy as np
from PIL import Image
from typing import (TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Tuple,
                    Union)
from pydantic import BaseModel, Field, model_validator

class ResizeImage(BaseModel):
    size: int | tuple[int, int] = (256, 480)  # H,W

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.Resize(self.size, interpolation=v2.InterpolationMode.NEAREST)

    @property
    def resolution(self) -> Tuple[int, int]:
        if isinstance(self.size, int):
            return (self.size, self.size)
        elif isinstance(self.size, list) and len(self.size) == 2:
            return (self.size[0], self.size[1])
        elif isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        else:
            raise ValueError("size should be int or list of two ints")


class ColorJitter(BaseModel):
    brightness: float | tuple[float, float] = Field(default_factory=lambda: 0.2)
    contrast: float | tuple[float, float] = Field(default_factory=lambda: (0.8, 1.2))
    saturation: float | tuple[float, float] = Field(default_factory=lambda: (0.8, 1.2))
    hue: float | tuple[float, float] = Field(default_factory=lambda: 0.05)

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class CenterCrop(BaseModel):
    size: int | tuple[int, int] = (224,224) # H,W

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.CenterCrop(self.size)


class RandomCrop(BaseModel):
    size: int | tuple[int, int] = (224, 224)  # H,W
    pad_if_needed: bool = True  # pad when the input is smaller than the crop

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.RandomCrop(self.size, pad_if_needed=self.pad_if_needed)


class Normalize(BaseModel):
    mean: float | List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: float | List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        mean = [self.mean] * 3 if isinstance(self.mean, float) else self.mean
        std = [self.std] * 3 if isinstance(self.std, float) else self.std
        return v2.Normalize(mean, std)  # type: ignore

class GaussianNoise(BaseModel):
    mean: float = 0
    std: float = 3
    prob_skip: float = 0.1


    def __call__(self, img: Image.Image):

        rnd = np.random.rand()
        if rnd < self.prob_skip:
            return img  # 10% chance to skip adding noise

        arr = np.array(img).astype(np.float32)

        # mean = int(np.random.uniform(-self.mean, self.mean)) # real mean
        noise = np.random.normal(self.mean, self.std, arr.shape)
        arr_noisy = arr + noise

        arr_noisy = np.clip(arr_noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_noisy)

class SaltPepperNoise:
    """
    Adds salt-and-pepper noise to a tensor image.
    - prob: total probability of altering a pixel
    - salt_vs_pepper: fraction of salt (white) vs pepper (black)
    """
    def __init__(self, prob=0.01, salt_vs_pepper=0.5):
        self.prob = prob
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, img):
        # img is PIL Image
        arr = np.array(img)  # H x W x C

        # Random mask
        rnd = np.random.rand(arr.shape[0], arr.shape[1])

        pepper_mask = rnd < (self.prob * (1 - self.salt_vs_pepper))
        salt_mask   = rnd > (1 - self.prob * self.salt_vs_pepper)

        arr = arr.copy()
        # Pepper = 0
        arr[pepper_mask] = 0
        # Salt = 255
        arr[salt_mask] = 255

        return Image.fromarray(arr)

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob}, salt_vs_pepper={self.salt_vs_pepper})"
