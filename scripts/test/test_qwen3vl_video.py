import os
import sys
import warnings
import logging
# os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"  # decord2 installs as `decord`; torchcodec needs FFmpeg which is unavailable

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*CUDA-fused xIELU.*")
logging.getLogger("xielu").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class _StderrFilter:
    _keywords = ("CUDA-fused xIELU", "pip install git+https://github.com/nickjbrowning/XIELU")
    def __init__(self, stream): self._stream = stream
    def write(self, s):
        if not any(k in s for k in self._keywords):
            self._stream.write(s)
    def flush(self): self._stream.flush()
    def __getattr__(self, a): return getattr(self._stream, a)

sys.stderr = _StderrFilter(sys.stderr)

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model, output_loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    output_loading_info=True,
    attn_implementation="flash_attention_2"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    video_load_backend="decord"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
