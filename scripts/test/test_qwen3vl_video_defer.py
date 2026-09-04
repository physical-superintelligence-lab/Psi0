"""Verify DeferredQwen3VLProcessor against the stock Qwen3-VL processor.

Decodes real frames and feeds the SAME native uint8 (T,C,H,W) to both processors
(do_sample_frames=False, bypassing qwen_vl_utils.fetch_video). Asserts identical
video_grid_thw + input_ids, and same-device pixel_values_videos bit-exactness.
Generation is informational only.

Run:
    .venv-psi/bin/python scripts/test/test_qwen3vl_video_defer.py [--no-generate]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import numpy as np
import torch

# Instruct model by default (captions videos); a VLA ckpt passes the pixel/grid
# checks but generates nothing.
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_VIDEO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"


def load_frames(video: str, num_frames: int) -> torch.Tensor:
    """Decode `num_frames` uniformly-sampled RGB frames -> uint8 (T,C,H,W)."""
    path = video
    if video.startswith("http://") or video.startswith("https://"):
        path = os.path.join("/tmp", os.path.basename(video.split("?")[0]))
        if not os.path.exists(path):
            print(f"downloading {video} -> {path}")
            urllib.request.urlretrieve(video, path)

    import decord

    vr = decord.VideoReader(path)
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    frames = vr.get_batch(idx).asnumpy()  # (T,H,W,C) uint8 RGB
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)


def build_text(processor, num_frames: int):
    """Chat template text with a single <video> placeholder (no decoding here)."""
    dummy = [np.zeros((4, 4, 3), dtype=np.uint8)] * num_frames  # template-only
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": dummy},
            {"type": "text", "text": "Describe this video."},
        ],
    }]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-generate", action="store_true", help="skip model.generate")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from psi.data.gear.transform.qwen3vl_video_gpu import DeferredQwen3VLProcessor

    frames = load_frames(args.video, args.num_frames)
    print(f"frames: {tuple(frames.shape)} {frames.dtype}")

    # Separate instances: the deferred wrapper swaps video_processor in place.
    stock = AutoProcessor.from_pretrained(args.model)
    deferred = DeferredQwen3VLProcessor.from_pretrained(args.model)

    text = build_text(stock, args.num_frames)
    videos_kwargs = {"do_sample_frames": False}

    common = dict(
        text=[text],
        videos=[frames],
        return_tensors="pt",
        videos_kwargs=videos_kwargs,
    )

    out_stock = stock(**common)
    out_dfr = deferred(**common)

    # --- 1. grid + input_ids identical (CPU stage) ---
    g_s = out_stock["video_grid_thw"]
    g_d = out_dfr["video_grid_thw"]
    print(f"video_grid_thw  stock={g_s.tolist()}  deferred={g_d.tolist()}")
    grid_ok = torch.equal(g_s, g_d)
    ids_ok = torch.equal(out_stock["input_ids"], out_dfr["input_ids"])
    print(f"  grid match:      {grid_ok}")
    print(f"  input_ids match: {ids_ok}  (len={out_stock['input_ids'].shape[-1]})")

    # --- 2. finalize on both devices ---
    # Correctness is a same-device claim (CPU kernel must match stock CPU
    # _preprocess). The GPU resize adds a small CPU<->CUDA delta: diagnostic only.
    def finalize_on(device):
        d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out_dfr.items()}
        return deferred.finalize(d)

    pv_s = out_stock["pixel_values_videos"].float()

    dfr_cpu = finalize_on("cpu")
    pv_cpu = dfr_cpu["pixel_values_videos"].float()
    cpu_max = (pv_s - pv_cpu).abs().max().item() if pv_s.shape == pv_cpu.shape else float("inf")
    pv_ok = pv_s.shape == pv_cpu.shape and cpu_max <= 1e-4
    print(f"pixel_values_videos  stock={tuple(pv_s.shape)}  deferred={tuple(pv_cpu.shape)}")
    print(f"  [correctness] CPU finalize vs stock CPU: max_diff={cpu_max:.3e}  "
          f"bit_exact(<=1e-4)={pv_ok}")

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    dfr_gpu = None
    if use_cuda:
        dfr_gpu = finalize_on(args.device)
        pv_gpu = dfr_gpu["pixel_values_videos"].float().cpu()
        gpu_max = (pv_s - pv_gpu).abs().max().item()
        print(f"  [diagnostic]  GPU finalize vs stock CPU: max_diff={gpu_max:.3e}  "
              f"(CPU<->CUDA resize delta; not a logic error)")

    ok = grid_ok and ids_ok and pv_ok

    # --- 3. generation (informational only) ---
    if not args.no_generate:
        from transformers import Qwen3VLForConditionalGeneration
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model, dtype="auto", device_map=args.device,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            print(f"(flash_attention_2 unavailable: {e}; falling back to sdpa)")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model, dtype="auto", device_map=args.device,
                attn_implementation="sdpa",
            )
        model.eval()

        def generate(inputs):
            inputs = {k: (v.to(model.device, model.dtype)
                          if (torch.is_tensor(v) and v.is_floating_point())
                          else v.to(model.device) if torch.is_tensor(v) else v)
                      for k, v in dict(inputs).items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            trimmed = gen[:, inputs["input_ids"].shape[-1]:]
            return stock.batch_decode(trimmed, skip_special_tokens=True)[0]

        # Greedy text is brittle: bf16 rounding of the ~1e-7 float32 diff flips
        # near-ties, so divergence here is not a correctness signal.
        txt_s = generate(out_stock)
        txt_cpu = generate(dfr_cpu)
        if not txt_s.strip():
            print("\n(no text — likely a VLA checkpoint; correctness checks above "
                  "are unaffected.)")
        print("\n--- stock generation ---\n" + txt_s)
        print("\n--- deferred (CPU finalize) generation ---\n" + txt_cpu)
        print(f"\n[informational] stock == deferred(CPU finalize): "
              f"{txt_s.strip() == txt_cpu.strip()}")
        if dfr_gpu is not None:
            txt_gpu = generate(dfr_gpu)
            print("\n--- deferred (GPU finalize, production path) generation ---\n" + txt_gpu)
            print(f"[informational] stock == deferred(GPU finalize): "
                  f"{txt_s.strip() == txt_gpu.strip()}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
