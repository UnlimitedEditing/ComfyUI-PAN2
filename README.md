# ComfyUI-PAN2

ComfyUI nodes for [IFM/PAN2](https://huggingface.co/IFM/PAN2) (29B MMDiT video/world model, t2v + i2v, Apache-2.0).
Pure PyTorch (SDPA attention): no flash-attn-4 / triton / quack, so it runs on Windows.

## Weights
- `models/diffusion_models/PAN2-fp8/` = [zombothe26/pan2-fp8](https://huggingface.co/zombothe26/pan2-fp8) (33.7 GB, made with `quantize_pan2.py`), or the original bf16 `IFM/PAN2/transformer` folder.
  Both are the diffusers layout and load directly, no conversion step.
- VAE: stock ComfyUI Wan 2.2 VAE (`vae/wan2.2_vae.safetensors`).
- Text encoder: `Qwen/Qwen3.5-9B` (HF id or local folder; needs a `transformers` that knows `qwen3_5`).

## Nodes
`PAN2 Model Loader` -> `PAN2 Text Encode (Qwen3.5)` -> `PAN2 Sampler` -> core `VAE Decode` (Wan2.2 VAE) -> `Create Video` @ 24 fps.
For i2v, give the sampler a `start_image` and the Wan2.2 `vae`. Defaults match the official ones
(50 steps, CFG 2.5, shift 9, 704x1248, 241 frames, seed 42).

## Memory
`placement = blocks_on_cpu` keeps the weights in RAM and moves each of the 44 blocks to the GPU only while it runs
(needs ~34 GB RAM for fp8). CFG runs as one batch of 2, so activations are the limit at full 704x1248x241: start small
(e.g. 448x768, 49 frames) and scale up.

## Status
- `tests/test_selfcontained.py` passes (shapes, block swap, padding mask, bf16, fp8 vs fp32).
- `tests/test_equivalence.py` (port vs OFFICIAL model class on tiny weights) is written but NOT yet run.
- Not yet run on the real 29B weights or inside a live ComfyUI. Sampling quality, speed and the Qwen3.5 loading path are unverified.
