import json
import torch
import numpy as np

from base64 import b64decode, b64encode
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from typing import Union, Dict, Any, List
from pathlib import Path

def numpy_serialize(o):
    # an already-serialized blob passes through unchanged
    if isinstance(o, dict) and "__numpy__" in o:
        return o
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)

def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """
    Recursively processes a JSON-like dictionary, converting any NumPy arrays
    or lists of NumPy arrays into a serializable format using the provided function.

    Args:
        data: The JSON-like dictionary or object to process.
        func: A function to apply to each NumPy array to make it serializable.

    Returns:
        The processed dictionary or object with all NumPy arrays converted.
    """
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data
    
class Message(object):
    def __init__(self):
        pass
    
    def serialize(self):
        raise NotImplementedError
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        raise NotImplementedError

class RequestMessage(Message):
    def __init__(self, image: Dict[str, Any], 
                 instruction: str, 
                 history: Dict[str, Any], 
                 state: Dict[str, Any], 
                 condition: Dict[str, Any], 
                 gt_action: Union[np.ndarray, List], 
                 dataset_name: str, 
                 timestamp: str):
        self.image, self.instruction, self.history, self.state, self.gt_action, self.dataset_name, self.timestamp = \
            image, instruction, history, state, gt_action, dataset_name, timestamp
        self.condition = condition

    def serialize(self):
        msg = {
            "image": self.image,
            "instruction": self.instruction,
            "history": self.history,
            "state": self.state,
            "condition": self.condition,
            "gt_action": self.gt_action,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize) # type: ignore
        return cls(
            image=response["image"],
            instruction=response["instruction"],
            history=response["history"],
            state=response["state"],
            condition=response["condition"],
            gt_action=response["gt_action"],
            dataset_name=response["dataset_name"],
            timestamp=response["timestamp"]
        )

class ResponseMessage(Message):
    def __init__(self, action: np.ndarray, err: float, traj_image: np.ndarray = np.zeros((1,1,3), dtype=np.uint8)):
        self.action = action
        self.err = err
        self.traj_image = traj_image
    
    def serialize(self):
        msg = {
            "action": self.action,
            "err": self.err,
            "traj_image": self.traj_image
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize) # type: ignore
        return cls(action=response["action"], err=response["err"], traj_image=response["traj_image"])
    
