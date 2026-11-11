"""
Pure-PyTorch inference port of the PAN 2 video diffusion transformer (29B MMDiT).

Differences from the official `pan` package, on purpose:
  * no flash-attn-4 / triton / quack: attention is torch SDPA, everything else is plain ops
  * loads the DIFFUSERS-layout shards (IFM/PAN2/transformer, or the fp8 repo made by
    quantize_pan2.py) directly, module names == diffusers key names, so no conversion step
  * fp8 weights (`<name>.weight` float8_e4m3fn + `<name>.scale_weight` fp32 scalar) are
    dequantized per layer on the fly
  * optional block swapping: weights stay in CPU RAM and each of the 44 blocks is moved to the
    GPU only while it runs (fits a 24 GB card with the fp8 weights)

Only the dense inference path (`PanVideoDiffusionTransformer._forward_dense`) is ported.
"""
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

SCALE_SUFFIX = ".scale_weight"


# ----------------------------------------------------------------------------- layers


class QLinear(nn.Module):
    """Linear whose weight may be float8 with a per-tensor fp32 scale (dequantized in fp32, then cast)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.register_buffer("scale_weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.scale_weight is not None:
            w = (w.to(torch.float32) * self.scale_weight.to(torch.float32)).to(x.dtype)
        elif w.dtype != x.dtype:
            w = w.to(x.dtype)
        b = self.bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        return F.linear(x, w, b)


class RMSNorm(nn.Module):
    """fp32 normalisation, cast back to input dtype, THEN scaled (the order PAN 2 was trained with)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt)
        return x * self.weight.to(dt)


class ProjAct(nn.Module):
    """diffusers GELU wrapper: `net.0.proj` (GELU-tanh MLP up-projection)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = QLinear(dim_in, dim_out, bias=True)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class FF(nn.Module):
    """`net` = [up(+gelu_tanh), dropout-placeholder, down] -> keys net.0.proj / net.2"""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.net = nn.ModuleList([ProjAct(dim, inner), nn.Identity(), QLinear(inner, dim, bias=True)])

    def forward(self, x):
        return self.net[2](self.net[0](x))


class SwiGLUFF(nn.Module):
    """Refiner FFN (bias-free). `net.0.proj.weight` stores [w3; w1]; out = w2(w3(x) * silu(w1(x)))."""

    class _P(nn.Module):
        def __init__(self, dim, inner):
            super().__init__()
            self.proj = QLinear(dim, 2 * inner, bias=False)

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.net = nn.ModuleList([SwiGLUFF._P(dim, inner), nn.Identity(), QLinear(inner, dim, bias=False)])

    def forward(self, x):
        w3, w1 = self.net[0].proj(x).chunk(2, dim=-1)
        return self.net[2](w3 * F.silu(w1))


class _Attn(nn.Module):
    pass


def _mod(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _rotate_half(x):
    pairs = x.unflatten(-1, (-1, 2))
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """q,k: [B,S,H,D]; cos,sin: [S,D] fp32. Rotation in fp32, cast back."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rot(x):
        x32 = x.float()
        return (x32 * cos + _rotate_half(x32) * sin).type_as(x)

    return rot(q), rot(k)


def rope_tables(axes_dim, grid, theta, temporal_scale, device):
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in grid], indexing="ij")
    scales = [temporal_scale] + [1.0] * (len(axes_dim) - 1)
    cos_l, sin_l = [], []
    for ax, dim, sc in zip(axes, axes_dim, scales):
        pos = ax.reshape(-1)
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
        ang = torch.outer(pos * sc, inv)
        cos_l.append(ang.cos().repeat_interleave(2, dim=1))
        sin_l.append(ang.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos_l, 1).to(device), torch.cat(sin_l, 1).to(device)


def sdpa(q, k, v, key_mask: Optional[torch.Tensor]):
    """q,k,v [B,S,H,D] -> [B,S,H*D]; key_mask [B,S] bool (True = real token) or None."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    mask = None if key_mask is None else key_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


# ----------------------------------------------------------------------------- caption refiner


class RefinerAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        hd = dim // heads
        self.to_q = QLinear(dim, dim, bias=False)
        self.to_k = QLinear(dim, dim, bias=False)
        self.to_v = QLinear(dim, dim, bias=False)
        self.norm_q = RMSNorm(hd, 1e-5)
        self.norm_k = RMSNorm(hd, 1e-5)
        self.to_out = nn.ModuleList([QLinear(dim, dim, bias=False)])


class RefinerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(dim, 1e-5)
        self.attn = RefinerAttn(dim, heads)
        self.norm2 = RMSNorm(dim, 1e-5)
        self.ff = SwiGLUFF(dim, int(dim * mlp_ratio))

    def forward(self, x, mask):
        h = self.norm1(x)
        a = self.attn
        q = a.to_q(h).unflatten(-1, (self.heads, -1))
        k = a.to_k(h).unflatten(-1, (self.heads, -1))
        v = a.to_v(h).unflatten(-1, (self.heads, -1))
        q = a.norm_q(q).to(v.dtype)
        k = a.norm_k(k).to(v.dtype)
        x = x + a.to_out[0](sdpa(q, k, v, mask))
        return x + self.ff(self.norm2(x))


# ----------------------------------------------------------------------------- joint block


class JointAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        hd = dim // heads
        for n in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            setattr(self, n, QLinear(dim, dim, bias=True))
        for n in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            setattr(self, n, RMSNorm(hd, 1e-6))
        self.to_out = nn.ModuleList([QLinear(dim, dim, bias=True)])
        self.to_add_out = QLinear(dim, dim, bias=True)


class _Norm1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = QLinear(dim, 6 * dim, bias=True)


class JointBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, context_mlp_ratio):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.norm1 = _Norm1(dim)  # adaLN modulation (video tokens, from timestep)
        self.scale_shift_table_context = nn.Parameter(torch.empty(6 * dim), requires_grad=False)
        self.attn = JointAttn(dim, heads)
        self.ff = FF(dim, int(dim * mlp_ratio))
        self.ff_context = FF(dim, int(dim * context_mlp_ratio))

    def forward(self, x, temb, n_txt, rope, key_mask):
        B, S, C = x.shape
        n_vid = S - n_txt
        v_s1, v_c1, v_g1, v_s2, v_c2, v_g2 = self.norm1.linear(F.silu(temb)).chunk(6, dim=-1)
        t_s1, t_c1, t_g1, t_s2, t_c2, t_g2 = (
            self.scale_shift_table_context.to(x.dtype).unsqueeze(0).expand(B, -1).chunk(6, dim=-1)
        )
        img, txt = x[:, :n_vid], x[:, n_vid:]
        img_m = _mod(F.layer_norm(img, (C,), eps=1e-6), v_s1, v_c1)
        txt_m = _mod(F.layer_norm(txt, (C,), eps=1e-6), t_s1, t_c1)

        a, H = self.attn, self.heads
        un = lambda t: t.unflatten(-1, (H, -1))
        iq, ik, iv = un(a.to_q(img_m)), un(a.to_k(img_m)), un(a.to_v(img_m))
        tq, tk, tv = un(a.add_q_proj(txt_m)), un(a.add_k_proj(txt_m)), un(a.add_v_proj(txt_m))
        iq, ik = a.norm_q(iq).to(iv.dtype), a.norm_k(ik).to(iv.dtype)
        tq, tk = a.norm_added_q(tq).to(tv.dtype), a.norm_added_k(tk).to(tv.dtype)
        iq, ik = apply_rope(iq, ik, rope[0], rope[1])  # video only; caption q/k are not rotated

        q, k, v = (torch.cat(p, dim=1) for p in ((iq, tq), (ik, tk), (iv, tv)))
        att = sdpa(q, k, v, key_mask)
        att_i, att_t = att[:, :n_vid], att[:, n_vid:]

        img = img + a.to_out[0](att_i) * v_g1.unsqueeze(1)
        txt = txt + a.to_add_out(att_t) * t_g1.unsqueeze(1)
        img = img + self.ff(_mod(F.layer_norm(img, (C,), eps=1e-6), v_s2, v_c2)) * v_g2.unsqueeze(1)
        txt = txt + self.ff_context(_mod(F.layer_norm(txt, (C,), eps=1e-6), t_s2, t_c2)) * t_g2.unsqueeze(1)
        return torch.cat([img, txt], dim=1)


# ----------------------------------------------------------------------------- model


class _TimeEmb(nn.Module):
    def __init__(self, dim, channels=256):
        super().__init__()
        self.channels = channels
        self.linear_1 = QLinear(channels, dim, bias=True)
        self.linear_2 = QLinear(dim, dim, bias=True)

    def forward(self, t):
        half = self.channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        emb = t[:, None].float() * freqs[None]
        wdt = self.linear_1.weight.dtype
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1).to(torch.bfloat16 if wdt == torch.float8_e4m3fn else wdt)
        return self.linear_2(F.silu(self.linear_1(emb)))


class _NormOut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = QLinear(dim, 2 * dim, bias=True)  # stored [scale; shift]


class PAN2Transformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.dim, self.heads = dim, cfg["num_attention_heads"]
        self.patch = tuple(cfg["patch_size"])
        self.in_channels, self.out_channels = cfg["in_channels"], cfg["out_channels"]
        self.rope_axes = tuple(cfg["rope_axes_dim"])
        self.rope_theta = float(cfg["rope_theta"])
        self.rope_tscale = float(cfg["rope_temporal_scale"])

        self.x_embedder = nn.Conv3d(self.in_channels, dim, kernel_size=self.patch, stride=self.patch)
        self.time_embedder = _TimeEmb(dim)
        self.context_embedder = nn.Sequential(RMSNorm(cfg["text_embed_dim"], 1e-5), QLinear(cfg["text_embed_dim"], dim, bias=True))
        self.context_type_embedding = nn.Parameter(torch.empty(dim), requires_grad=False)
        self.context_refiner = nn.ModuleList(
            [RefinerBlock(dim, self.heads, cfg.get("refiner_mlp_ratio", 4.0)) for _ in range(cfg["num_refiner_layers"])]
        )
        self.transformer_blocks = nn.ModuleList(
            [JointBlock(dim, self.heads, cfg["mlp_ratio"], cfg["context_mlp_ratio"]) for _ in range(cfg["num_layers"])]
        )
        self.norm_out = _NormOut(dim)
        self.proj_out = QLinear(dim, math.prod(self.patch) * self.out_channels, bias=True)
        self.requires_grad_(False)

    # -- text path (small, always resident) -------------------------------------------------
    def encode_text(self, text_states, text_mask):
        t = self.context_embedder(text_states)
        mask = text_mask.bool() | (torch.arange(text_mask.shape[1], device=text_mask.device) == 0)
        for blk in self.context_refiner:
            t = blk(t, mask)
        return t + self.context_type_embedding.to(t.dtype)

    # -- forward ---------------------------------------------------------------------------
    def forward(self, latents, timestep, text_states, text_mask, block_hook=None):
        """latents [B,97,T,H,W] -> velocity [B,48,T,H,W]. `block_hook(i, block)` -> context manager wrapping each block."""
        dt = self.x_embedder.weight.dtype
        dev = latents.device
        grid = tuple(s // p for s, p in zip(latents.shape[2:], self.patch))
        cos, sin = rope_tables(self.rope_axes, grid, self.rope_theta, self.rope_tscale, dev)

        video = self.x_embedder(latents.to(dt)).flatten(2).transpose(1, 2)
        temb = self.time_embedder(timestep.to(dev))
        text = self.encode_text(text_states.to(dt), text_mask)
        n_vid, n_txt = video.shape[1], text.shape[1]
        key_mask = None
        if not bool(text_mask.all()):
            key_mask = torch.cat([torch.ones(video.shape[0], n_vid, dtype=torch.bool, device=dev), text_mask.bool()], dim=1)
        x = torch.cat([video, text], dim=1)
        for i, blk in enumerate(self.transformer_blocks):
            if block_hook is None:
                x = blk(x, temb, n_txt, (cos, sin), key_mask)
            else:
                with block_hook(i, blk):
                    x = blk(x, temb, n_txt, (cos, sin), key_mask)
        scale, shift = self.norm_out.linear(F.silu(temb)).chunk(2, dim=1)
        h = F.layer_norm(x[:, :n_vid], (self.dim,), eps=1e-6) * (1 + scale)[:, None] + shift[:, None]
        tokens = self.proj_out(h)
        t, hh, ww = grid
        pt, ph, pw = self.patch
        c = self.out_channels
        tokens = tokens.reshape(tokens.shape[0], t, hh, ww, c, pt, ph, pw)
        tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)  # n c t pt h ph w pw
        return tokens.reshape(tokens.shape[0], c, t * pt, hh * ph, ww * pw)


# ----------------------------------------------------------------------------- loading


def _resolve(root: nn.Module, parts: List[str]):
    obj = root
    for p in parts:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _set_tensor(model: nn.Module, key: str, t: torch.Tensor):
    parts = key.split(".")
    if key.endswith(SCALE_SUFFIX):
        owner = _resolve(model, parts[:-1])
        owner._buffers["scale_weight"] = t.to(torch.float32).reshape(())
        return
    owner = _resolve(model, parts[:-1])
    name = parts[-1]
    if name in owner._parameters:
        owner._parameters[name] = nn.Parameter(t, requires_grad=False)
    else:
        raise KeyError(f"{key}: no parameter '{name}' on {type(owner).__name__}")


def read_index(folder: str) -> Dict[str, str]:
    idx = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(folder, f), "pt") as h:
                wm.update({k: f for k in h.keys()})
    return wm


def load_pan2(folder: str, compute_dtype=torch.bfloat16) -> PAN2Transformer:
    cfg = json.load(open(os.path.join(folder, "config.json")))
    with torch.device("meta"):
        model = PAN2Transformer(cfg)
    wmap = read_index(folder)
    by_shard: Dict[str, List[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    seen = set()
    for shard, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(folder, shard), "pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                if k.endswith(SCALE_SUFFIX):
                    pass
                elif t.dtype != torch.float8_e4m3fn and t.is_floating_point():
                    t = t.to(compute_dtype)
                _set_tensor(model, k, t)
                seen.add(k)
    # sanity: nothing left on meta
    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError(f"{len(left)} parameters missing from checkpoint, e.g. {left[:5]}")
    return model


# ----------------------------------------------------------------------------- block swapping


class BlockSwap:
    """Context-manager factory: moves one block's tensors to `device` for its forward, then restores the CPU refs."""

    def __init__(self, device):
        self.device = device

    def __call__(self, i, block):
        return _Swap(block, self.device)


class _Swap:
    def __init__(self, block, device):
        self.block, self.device = block, device
        self.saved = []

    def __enter__(self):
        for m in self.block.modules():
            for name, p in list(m._parameters.items()):
                if p is not None:
                    self.saved.append((m, "p", name, p.data))
                    p.data = p.data.to(self.device, non_blocking=True)
            for name, b in list(m._buffers.items()):
                if b is not None:
                    self.saved.append((m, "b", name, b))
                    m._buffers[name] = b.to(self.device, non_blocking=True)

    def __exit__(self, *exc):
        for m, kind, name, orig in self.saved:
            if kind == "p":
                m._parameters[name].data = orig
            else:
                m._buffers[name] = orig
        self.saved = []
        return False


def move_nonblock_to(model: PAN2Transformer, device):
    """Everything except the 44 joint blocks, including bare parameters (context_type_embedding) that
    named_children() would miss (that omission was the cause of the first Graydient probe failure)."""
    for name, p in model.named_parameters():
        if not name.startswith("transformer_blocks."):
            p.data = p.data.to(device)
    for name, m in model.named_modules():
        if name.startswith("transformer_blocks"):
            continue
        for bn, b in list(m._buffers.items()):
            if b is not None:
                m._buffers[bn] = b.to(device)


def move_blocks_to(model: PAN2Transformer, device):
    model.transformer_blocks.to(device)
