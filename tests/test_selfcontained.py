"""
Self-contained smoke test: runs only OUR code (no official PAN2 source needed).
Tiny random model -> diffusers-layout folder -> load (fp32 / bf16 / fp8) -> forward, block swap, shape/NaN checks.
For true numerical equivalence with the official model see test_equivalence.py.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import pan2_model as pm  # noqa: E402

HID, HEADS, LAYERS, TXT = 256, 4, 2, 64
HD = HID // HEADS
cfg = dict(attention_head_dim=HD, context_mlp_ratio=4.0, in_channels=97, mlp_ratio=8.0, num_attention_heads=HEADS,
           num_layers=LAYERS, num_refiner_layers=4, out_channels=48, patch_size=[1, 2, 2], refiner_mlp_ratio=4.0,
           rope_axes_dim=[8, 28, 28], rope_temporal_scale=16 / 24, rope_theta=256.0, text_embed_dim=TXT)

torch.manual_seed(0)
with torch.device("meta"):
    skeleton = pm.PAN2Transformer(cfg)
sd = {}
for name, p in skeleton.named_parameters():
    shape = tuple(p.shape)
    if len(shape) >= 2:
        sd[name] = torch.randn(shape) * (0.5 / shape[-1] ** 0.5)
    elif "norm" in name or name.endswith("norm_q.weight"):
        sd[name] = 1.0 + 0.1 * torch.randn(shape)
    else:
        sd[name] = 0.1 * torch.randn(shape)

tmp = Path(tempfile.mkdtemp())
d = tmp / "full"
d.mkdir()
save_file(sd, str(d / "diffusion_pytorch_model.safetensors"))
json.dump(cfg, open(d / "config.json", "w"))
json.dump({"weight_map": {k: "diffusion_pytorch_model.safetensors" for k in sd}},
          open(d / "diffusion_pytorch_model.safetensors.index.json", "w"))

B, T, H, W, L = 2, 3, 8, 10, 9
x = torch.randn(B, 97, T, H, W)
t = torch.tensor([700.0, 250.0])
txt = torch.randn(B, L, TXT)
mask = torch.ones(B, L, dtype=torch.long)
mask[1, 5:] = 0

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))


m32 = pm.load_pan2(str(d), torch.float32)
with torch.no_grad():
    ref = m32(x, t, txt, mask)
check("output shape [B,48,T,H,W]", ref.shape == (B, 48, T, H, W), str(tuple(ref.shape)))
check("finite", torch.isfinite(ref).all().item())

with torch.no_grad():
    sw = m32(x, t, txt, mask, block_hook=pm.BlockSwap("cpu"))
check("block-swap identical", torch.equal(ref, sw))

# padding must not leak: the masked tail of prompt 1 must not change its output
txt2 = txt.clone()
txt2[1, 5:] = torch.randn_like(txt2[1, 5:]) * 10
with torch.no_grad():
    o2 = m32(x, t, txt2, mask)
check("masked padding has no effect", torch.allclose(ref, o2, atol=1e-5), f"max diff {(ref - o2).abs().max():.2e}")

mbf = pm.load_pan2(str(d), torch.bfloat16)
with torch.no_grad():
    o_bf = mbf(x, t, txt, mask)
cos = F.cosine_similarity(o_bf.float().flatten(), ref.flatten(), dim=0).item()
check("bf16 close to fp32", cos > 0.995, f"cos={cos:.5f}")

q8 = tmp / "fp8"
r = subprocess.run([sys.executable, str(REPO.parent / "tripostl" / "quantize_pan2.py"), "--src", str(d), "--dst", str(q8),
                    "--device", "cpu"], capture_output=True, text=True)
check("quantize_pan2.py ran", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
if r.returncode == 0:
    m8 = pm.load_pan2(str(q8), torch.bfloat16)
    n_fp8 = sum(1 for p in m8.parameters() if p.dtype == torch.float8_e4m3fn)
    check("fp8 tensors loaded", n_fp8 > 0, f"{n_fp8} fp8 weights")
    with torch.no_grad():
        o8 = m8(x, t, txt, mask)
    cos8 = F.cosine_similarity(o8.float().flatten(), ref.flatten(), dim=0).item()
    check("fp8 close to fp32", cos8 > 0.98, f"cos={cos8:.5f}")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
