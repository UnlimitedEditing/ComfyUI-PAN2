"""
Numerical equivalence test: our pure-PyTorch port vs the OFFICIAL PanVideoDiffusionTransformer (dense path).

Uses a tiny random-weight model (hidden 256, 2 blocks) so it runs on CPU in seconds. The chain tested:
  official native weights -> official convert_checkpoint.native_to_diffusers -> diffusers-layout folder
  -> our load_pan2() -> forward, compared against official._forward_dense.
Then the same folder is fp8-quantized with quantize_pan2.py and reloaded (tolerance check).

Needs the official repo source on disk (not installed): set PAN2_SRC to a checkout of github.com/ifm-ai/PAN2
(only pan/, scripts/ are read). flash-attn / triton / quack are stubbed; attention is replaced by torch SDPA.

    set PAN2_SRC=C:\\path\\to\\PAN2
    python tests/test_equivalence.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = Path(os.environ["PAN2_SRC"])
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO))

# ---- stub the heavy/GPU-only dependencies of the official package -------------------------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

stub("loguru", logger=types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None))
_unavail = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("packed/GPU path not available in test"))

def padded_flash_attention(q, k, v, q_mask, k_mask):
    """[B,S,H,D] -> [B,S,H,D]; key-masked SDPA (masked query rows are never read downstream)."""
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    m = None if k_mask is None else k_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(qt, kt, vt, attn_mask=m).transpose(1, 2)

stub("pan.utils.flash_attention", flash_attn_varlen=_unavail, padded_flash_attention=padded_flash_attention)
stub("pan.utils.ulysses", ulysses_flash_attn_varlen=_unavail)
stub("pan.commons.parallel_layout",
     current_parallel_layout=lambda: types.SimpleNamespace(cp_enabled=False, cp_group=None))
stub("pan.models.transformers.modules.triton_kernels",
     packed_simple_concat=_unavail, packed_gate=_unavail, packed_modulate=_unavail, packed_concat_mse_loss=_unavail)
stub("pan.models.transformers.modules.mlp_ops",
     grouped_out_proj=_unavail, grouped_qkv_proj=_unavail, grouped_ffn=_unavail, range_linear=_unavail)

from pan.models.transformers.pan_transformer import PanVideoDiffusionTransformer  # noqa: E402

spec = importlib.util.spec_from_file_location("convert_checkpoint", SRC / "scripts" / "convert_checkpoint.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

import pan2_model  # noqa: E402  (ours)
from safetensors.torch import save_file  # noqa: E402

# ---- tiny official model with non-trivial weights -------------------------------------------------
torch.manual_seed(0)
HID, HEADS, LAYERS, TXT = 256, 4, 2, 64
HD = HID // HEADS
AXES = (8, 28, 28)
official = PanVideoDiffusionTransformer(
    patch_size=(1, 2, 2), in_channels=97, out_channels=48, hidden_size=HID, num_attention_heads=HEADS,
    num_layers=LAYERS, rope_axes_dim=AXES, text_embed_dim=TXT, zero_init_modulation=False,
).eval()
with torch.no_grad():
    for n, p in official.named_parameters():
        if p.dim() >= 2 or "text_modulation" in n or "type_embedding" in n:
            p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5 if p.dim() >= 2 else 0.1))
        else:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
    for n, p in official.named_parameters():
        if "modulation.linear.weight" in n:
            p.mul_(0.5)

cfg = dict(attention_head_dim=HD, context_mlp_ratio=4.0, in_channels=97, mlp_ratio=8.0, num_attention_heads=HEADS,
           num_layers=LAYERS, num_refiner_layers=4, out_channels=48, patch_size=[1, 2, 2], refiner_mlp_ratio=4.0,
           rope_axes_dim=list(AXES), rope_temporal_scale=16 / 24, rope_theta=256.0, text_embed_dim=TXT)

native_sd = {k: v.detach().clone() for k, v in official.state_dict().items()}
dsd = conv.native_to_diffusers(native_sd.pop, LAYERS, 4, HD) if False else conv.native_to_diffusers(native_sd, LAYERS, 4, HD)

tmp = Path(tempfile.mkdtemp())
full = tmp / "diffusers_fp32"
full.mkdir()
save_file({k: v.contiguous() for k, v in dsd.items()}, str(full / "diffusion_pytorch_model.safetensors"))
json.dump(cfg, open(full / "config.json", "w"))

# ---- inputs ---------------------------------------------------------------------------------------
B, T, H, W, L = 2, 3, 8, 10, 9
latents = torch.randn(B, 97, T, H, W)
timestep = torch.tensor([700.0, 250.0])
text = torch.randn(B, L, TXT)
mask = torch.ones(B, L, dtype=torch.long)
mask[1, 5:] = 0  # second prompt is shorter

with torch.no_grad():
    ref, _ = official._forward_dense(latents, timestep, text, mask, None, None)


def compare(name, out, tol):
    err = (out.float() - ref.float()).abs().max().item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    ok = rel < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} max_abs={err:.3e} rel={rel:.3e} cos={cos:.6f} (tol rel<{tol})")
    return ok


results = []

# 1) exact path: fp32 weights, fp32 compute
m32 = pan2_model.load_pan2(str(full), compute_dtype=torch.float32)
with torch.no_grad():
    out = m32(latents, timestep, text, mask)
assert out.shape == ref.shape, (out.shape, ref.shape)
results.append(compare("fp32 port vs official", out, 1e-4))

# 2) block-swap path must be identical to the resident path
with torch.no_grad():
    out_swap = m32(latents, timestep, text, mask, block_hook=pan2_model.BlockSwap("cpu"))
results.append(compare("block-swap vs official", out_swap, 1e-4))

# 3) bf16 compute
mbf = pan2_model.load_pan2(str(full), compute_dtype=torch.bfloat16)
with torch.no_grad():
    out_bf = mbf(latents, timestep, text, mask)
results.append(compare("bf16 port vs official", out_bf, 5e-2))

# 4) fp8 quantized via quantize_pan2.py (index json required by that script)
json.dump({"weight_map": {k: "diffusion_pytorch_model.safetensors" for k in dsd}}, open(full / "diffusion_pytorch_model.safetensors.index.json", "w"))
q8 = tmp / "fp8"
r = subprocess.run([sys.executable, str(REPO.parent / "tripostl" / "quantize_pan2.py"), "--src", str(full), "--dst", str(q8),
                    "--device", "cpu"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout, r.stderr)
    sys.exit("quantize_pan2.py failed")
m8 = pan2_model.load_pan2(str(q8), compute_dtype=torch.bfloat16)
with torch.no_grad():
    out8 = m8(latents, timestep, text, mask)
results.append(compare("fp8 port vs official", out8, 1.5e-1))

print("\nALL PASS" if all(results) else "\nSOME FAILED")
sys.exit(0 if all(results) else 1)
