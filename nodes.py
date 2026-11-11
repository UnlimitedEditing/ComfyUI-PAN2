"""
ComfyUI nodes for IFM/PAN2 (29B MMDiT world/video model, t2v + i2v).

Graph:
  PAN2 Model Loader --\
  PAN2 Text Encode ----> PAN2 Sampler --> (core) VAE Decode [Wan2.2 VAE] --> Create Video / Save
  (optional) Load Image + Wan2.2 VAE --/

The VAE is the stock Wan 2.2 TI2V-5B VAE already in ComfyUI (vae/wan2.2_vae.safetensors); the sampler
normalises/denormalises latents with comfy's Wan22 latent format, so core VAEDecode just works.
"""
import gc
import math
import os
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import pan2_model

QWEN_ID = "Qwen/Qwen3.5-9B"
SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."
USER_HEADER = "<|im_start|>user\n"
LATENT_CH = 48
VAE_STRIDE_S, VAE_STRIDE_T = 16, 4


# ----------------------------------------------------------------------------- helpers

_T0 = time.time()


def _log(msg: str):
    """Timestamped line for the Graydient Logs tab (seconds since this module was imported ~ ComfyUI startup)."""
    print(f"[PAN2 +{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _resolve_qwen(s: str) -> str:
    """Local folder (absolute, or relative to models/) wins over a hub id: hub ids download inside the job budget."""
    if os.path.isdir(s):
        return s
    local = os.path.join(folder_paths.models_dir, s)
    return local if os.path.isdir(local) else s


def _model_dirs() -> List[str]:
    found = []
    for base in folder_paths.get_folder_paths("diffusion_models"):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if os.path.isfile(os.path.join(d, "config.json")) and any(f.endswith(".safetensors") for f in os.listdir(d)):
                found.append(name)
    return found or ["(put a PAN2 transformer folder in models/diffusion_models)"]


def _resolve_dir(name: str) -> str:
    for base in folder_paths.get_folder_paths("diffusion_models"):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            return d
    if os.path.isdir(name):
        return name
    raise FileNotFoundError(f"PAN2 transformer folder not found: {name}")


class PAN2Handle:
    """Loaded transformer plus its placement policy."""

    def __init__(self, model, device, swap):
        self.model, self.device, self.swap = model, device, swap

    def forward(self, x, t, text, mask):
        hook = pan2_model.BlockSwap(self.device) if self.swap else None
        return self.model(x.to(self.device), t.to(self.device), text.to(self.device), mask.to(self.device), block_hook=hook)


# ----------------------------------------------------------------------------- loader

class PAN2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_dir": (_model_dirs(),),
            "placement": (["auto", "blocks_on_cpu (24GB cards)", "all_on_gpu"],),
        }}

    RETURN_TYPES = ("PAN2_MODEL",)
    FUNCTION = "load"
    CATEGORY = "PAN2"

    def load(self, model_dir, placement):
        mm.unload_all_models()
        mm.soft_empty_cache()
        folder = _resolve_dir(model_dir)
        _log(f"loader: reading shards from {folder}")
        t0 = time.time()
        model = pan2_model.load_pan2(folder)
        _log(f"loader: shards read into RAM in {time.time() - t0:.1f}s")
        dev = mm.get_torch_device()
        nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if placement.startswith("all"):
            swap = False
        elif placement.startswith("blocks"):
            swap = True
        else:  # auto: need weights + ~8 GB working room
            free = mm.get_free_memory(dev)
            swap = free < nbytes + 8 * 1024 ** 3
        t0 = time.time()
        if swap:
            pan2_model.move_nonblock_to(model, dev)
        else:
            model.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        _log(f"loader: {nbytes / 1e9:.1f} GB, {'block-swapping from CPU' if swap else 'fully on GPU'}, "
             f"placement took {time.time() - t0:.1f}s, free VRAM {mm.get_free_memory(dev) / 1e9:.1f} GB")
        return (PAN2Handle(model, dev, swap),)


# ----------------------------------------------------------------------------- text encoder

_QWEN = {}


def _load_qwen(path_or_id: str):
    if path_or_id in _QWEN:
        return _QWEN[path_or_id]
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    cfg = AutoConfig.from_pretrained(path_or_id)
    cfg.text_config.num_hidden_layers -= 2  # PAN2 uses hidden states of layer N-2, no final norm
    full = AutoModelForImageTextToText.from_pretrained(path_or_id, config=cfg, dtype=torch.bfloat16,
                                                       attn_implementation="sdpa")
    lm = full.model.language_model
    del full.model.visual, full.lm_head
    lm.norm = torch.nn.Identity()
    lm.requires_grad_(False).eval()
    tok = AutoTokenizer.from_pretrained(path_or_id, padding_side="right")
    _QWEN[path_or_id] = (lm, tok)
    return _QWEN[path_or_id]


def _chat(caption: str):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": caption or " "}]


def _prefix_len(tok) -> int:
    ids = tok.apply_chat_template(_chat("x"), add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    ids = list(ids[0]) if isinstance(ids[0], (list, tuple)) else list(ids)
    hdr = tok(USER_HEADER, add_special_tokens=False)["input_ids"]
    for p in range(len(ids) - len(hdr) + 1):
        if ids[p:p + len(hdr)] == hdr:
            return p + len(hdr)
    raise ValueError("user-turn header not found in templated prompt")


@torch.no_grad()
def _encode_one(lm, tok, text: str, device, max_caption_tokens: int):
    pre = _prefix_len(tok)
    enc = tok.apply_chat_template(_chat(text), add_generation_prompt=True, tokenize=True, return_dict=True,
                                  truncation=True, max_length=pre + max_caption_tokens, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    hidden = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    return hidden[:, pre:].to(torch.bfloat16).cpu()  # [1, L, 4096]


class PAN2TextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "qwen": ("STRING", {"default": QWEN_ID, "tooltip": "HF id or local folder of Qwen3.5-9B"}),
            "max_caption_tokens": ("INT", {"default": 1000, "min": 16, "max": 1536}),
            "free_after": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("PAN2_COND", "PAN2_COND")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "PAN2"

    def encode(self, prompt, negative_prompt, qwen, max_caption_tokens, free_after):
        dev = mm.get_torch_device()
        qwen = _resolve_qwen(qwen)
        t0 = time.time()
        lm, tok = _load_qwen(qwen)
        _log(f"qwen: loaded {qwen} to RAM in {time.time() - t0:.1f}s")
        t0 = time.time()
        lm.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        _log(f"qwen: moved to {dev} in {time.time() - t0:.1f}s")
        try:
            t0 = time.time()
            pos = _encode_one(lm, tok, prompt, dev, max_caption_tokens)
            neg = _encode_one(lm, tok, negative_prompt, dev, max_caption_tokens)
            _log(f"qwen: encoded prompts in {time.time() - t0:.1f}s ({pos.shape[1]} / {neg.shape[1]} tokens)")
        finally:
            lm.to("cpu")
            if free_after:
                _QWEN.pop(qwen, None)
                del lm
                gc.collect()
            mm.soft_empty_cache()
        return ({"embeds": pos}, {"embeds": neg})


# ----------------------------------------------------------------------------- sampler

def _shifted_sigmas(steps: int, shift: float) -> torch.Tensor:
    g = torch.linspace(1, 0, steps + 1)
    return g if shift == 1.0 else (shift * g) / (1 + (shift - 1) * g)


def _cover_crop(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """img [1,H,W,3] 0-1 -> [1,h,w,3], scaled just enough to cover then center-cropped."""
    _, sh, sw, _ = img.shape
    zoom = max(w / sw, h / sh)
    nh, nw = int(round(sh * zoom)), int(round(sw * zoom))
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    top, left = (nh - h) // 2, (nw - w) // 2
    return x[:, :, top:top + h, left:left + w].permute(0, 2, 3, 1).contiguous()


class PAN2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PAN2_MODEL",),
                "positive": ("PAN2_COND",),
                "negative": ("PAN2_COND",),
                "width": ("INT", {"default": 1248, "min": 32, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 704, "min": 32, "max": 2048, "step": 32}),
                "length": ("INT", {"default": 241, "min": 1, "max": 481, "step": 4,
                                   "tooltip": "pixel frames; use 4k+1 (e.g. 97, 241). 24 fps."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "i2v_skip_first_latent_cfg": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Wan2.2 VAE, only needed with start_image"}),
                "start_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "PAN2"

    def sample(self, model, positive, negative, width, height, length, steps, cfg, shift, seed,
               i2v_skip_first_latent_cfg, vae=None, start_image=None):
        from comfy.latent_formats import Wan22

        # platform may inject any default /size: snap to the VAE-stride x patch grid instead of rejecting
        width, height = max(64, int(round(width / 32)) * 32), max(64, int(round(height / 32)) * 32)
        lf = Wan22()
        dev = model.device
        T = (length - 1) // VAE_STRIDE_T + 1
        rows, cols = height // VAE_STRIDE_S, width // VAE_STRIDE_S
        use_cfg = cfg > 1.0

        # ---- text (pad pos/neg to a common length, mask the padding) ----
        pe, ne = positive["embeds"], negative["embeds"]
        L = max(pe.shape[1], ne.shape[1]) if use_cfg else pe.shape[1]

        def pad(e):
            m = torch.zeros(1, L, dtype=torch.long)
            m[:, :e.shape[1]] = 1
            out = torch.zeros(1, L, e.shape[2], dtype=e.dtype)
            out[:, :e.shape[1]] = e
            return out, m

        pt, pm = pad(pe)
        if use_cfg:
            nt, nm = pad(ne)
            text, tmask = torch.cat([nt, pt]), torch.cat([nm, pm])  # [negative; positive]
        else:
            text, tmask = pt, pm

        # ---- conditioning channels: [cond latent (48) | mask (1)] appended to the noisy latent ----
        cond = torch.zeros(1, LATENT_CH, T, rows, cols)
        cmask = torch.zeros(1, 1, T, rows, cols)
        if start_image is not None:
            if vae is None:
                raise ValueError("start_image needs the Wan2.2 VAE input")
            img = _cover_crop(start_image[:1, :, :, :3].cpu().float(), height, width)
            z = vae.encode(img)  # [1,48,1,h,w], raw VAE space
            z = lf.process_in(z.float())
            cond[:, :, :1] = z[:, :, :1].cpu()
            cmask[:, :, :1] = 1.0
        if use_cfg:
            cond, cmask = torch.cat([cond] * 2), torch.cat([cmask] * 2)

        # ---- noise & Euler flow-matching loop (same arithmetic as the official FlowEulerScheduler) ----
        gen = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn((1, LATENT_CH, T, rows, cols), generator=gen, dtype=torch.bfloat16).to(dev)
        sigmas = _shifted_sigmas(steps, shift)
        timesteps = sigmas[:-1] * 1000.0
        pbar = comfy.utils.ProgressBar(steps)
        cond, cmask = cond.to(dev), cmask.to(dev)

        _log(f"sampler: {width}x{height}x{length}f -> latent grid T={T} {rows}x{cols}, {steps} steps, cfg {cfg}")
        t_loop = time.time()
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_step = time.time()
                mm.throw_exception_if_processing_interrupted()
                x_in = torch.cat([latents] * 2) if use_cfg else latents
                dit_in = torch.cat([x_in, cond.to(x_in.dtype), cmask.to(x_in.dtype)], dim=1)
                v = model.forward(dit_in, t.expand(x_in.shape[0]), text, tmask)
                if use_cfg:
                    v_neg, v_pos = v.chunk(2)
                    v = v_neg + cfg * (v_pos - v_neg)
                    if i2v_skip_first_latent_cfg and start_image is not None:
                        v[:, :, :1] = v_pos[:, :, :1]
                latents = latents.float() + v.float() * (sigmas[i + 1] - sigmas[i])
                pbar.update(1)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i < 3 or i == steps - 1:
                    _log(f"sampler: step {i + 1}/{steps} took {time.time() - t_step:.1f}s")
        _log(f"sampler: loop done in {time.time() - t_loop:.1f}s")

        out = lf.process_out(latents.float().cpu())  # back to raw VAE space for core VAEDecode
        return ({"samples": out},)


NODE_CLASS_MAPPINGS = {
    "PAN2ModelLoader": PAN2ModelLoader,
    "PAN2TextEncode": PAN2TextEncode,
    "PAN2Sampler": PAN2Sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PAN2ModelLoader": "PAN2 Model Loader",
    "PAN2TextEncode": "PAN2 Text Encode (Qwen3.5)",
    "PAN2Sampler": "PAN2 Sampler",
}
