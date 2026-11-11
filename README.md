# ComfyUI-PAN2

ComfyUI nodes for [IFM/PAN2](https://huggingface.co/IFM/PAN2) (29B MMDiT video/world model, t2v + i2v, Apache-2.0).
Pure PyTorch (SDPA attention): no flash-attn-4 / triton / quack, so it runs on Windows.

## Weights
- `models/diffusion_models/PAN2-fp8/` = [zombothe26/pan2-fp8](https://huggingface.co/zombothe26/pan2-fp8) (33.7 GB), or the original bf16 `IFM/PAN2/transformer` folder.
  Both are the diffusers layout and load directly, no conversion step. (Measured: fp8 and bf16 give the same latent statistics and frames.)
- VAE: stock ComfyUI Wan 2.2 VAE (`vae/wan2.2_vae.safetensors`).
- Text encoder: `Qwen/Qwen3.5-9B` (HF id or a local folder such as `llm/Qwen3.5-9B` under `models/`; needs a `transformers` that knows `qwen3_5`).

## Prompts matter (read this)
PAN 2 was trained on **structured JSON captions with time ranges**, and its default negative is a JSON template whose
`time_range_sec` fields are stamped with the clip duration. A one-line prompt with an empty negative gives blown-out,
mushy multi-frame clips; a structured prompt + the official negative gives clean ones.
`PAN2 Prompt Builder` builds both from a short description:

    scene ;; event ;; subject ;; lighting ;; viewpoint ;; style        (only the scene is required)

e.g. `A small red fishing boat floats on calm water at sunrise. ;; rocks gently on the swell ;; Fishing boat: red hull, white cabin ;; low golden sun`.
With a subject, write the event as what it *does*. `camera` picks one of the 16 official camera tags (0 = static).
Time ranges use `(frames - 1) / 24` seconds, as the official code does.

## Nodes
`PAN2 Model Loader` + `PAN2 Prompt Builder` -> `PAN2 Text Encode (Qwen3.5)` -> `PAN2 Sampler` -> core `VAE Decode` (Wan2.2 VAE) -> `Create Video` @ 24 fps.
For i2v, give the sampler a `start_image` and the Wan2.2 `vae`. Defaults match the official ones
(50 steps, CFG 2.5, shift 9, 704x1248, 241 frames); 16 steps already gives a clean clip at small sizes.

## Memory
`placement = blocks_on_cpu` (auto when the weights do not fit) keeps the weights in RAM and moves each of the 44 blocks to the GPU
only while it runs (needs ~34 GB RAM for fp8). CFG runs as one batch of 2 (the ~1480-token negative included), so activations
are the limit at full 704x1248x241: start small (e.g. 448x256, 13 frames) and scale up.

## Tests
- `tests/test_prompt_builder.py`: pure Python.
- `tests/test_selfcontained.py`: tiny random model; shapes, block swap, padding mask, bf16, fp8.
- `tests/test_equivalence.py`: port vs the OFFICIAL model class on tiny weights (needs `PAN2_SRC`, a checkout of ifm-ai/PAN2); passes to 1e-7 in fp32.

## Status
Verified on Graydient (fp8, 448x256, 13 frames, 16 steps): coherent clip in ~65 s (5090) / ~129 s (4090) of execution.
Not yet measured: 50 steps, native 704x1248, long clips within the 180 s budget; the i2v path is wired but lightly tested.
