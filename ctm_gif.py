"""CTM-inspired analysis for GRAINet's ViT-Tiny: per-layer attention +
certainty-over-depth, adapted from SakanaAI's Continuous Thought Machine
(https://github.com/SakanaAI/continuous-thought-machines) plotting ideas
-- per-internal-tick attention overlays and a running certainty curve,
with this ViT's layer depth standing in for CTM's internal "tick" axis
(this feedforward ViT has no tick loop of its own).

Produces an animated GIF: one frame per transformer block, showing that
layer's attention-received overlay (this ViT has no CLS token -- it uses
global average pooling -- so "attention received" is the natural analog,
column-summed from each layer's attention matrix) next to the predicted
21-bin grain-size histogram accumulating one more layer's "if the model
stopped here" prediction each frame, against the real ground-truth
histogram, titled with that layer's certainty score
(`1 - normalized_entropy` of the softmax output).

Note: needs a msgpack trained with `output_scalar=False, bins=21` (the
full-histogram model from the 3-model x 7-loss comparison), NOT the
scalar-`dm` model's checkpoint from the base training pipeline (that one
has a 1-unit output head, which silently produces NaN certainties here --
a real mistake made and caught while testing this exact script).

Usage:
    python ctm_gif.py --msgpack grainet_vit_histogram.msgpack --data data_KLEmme_1bank.npz --out grainet_ctm_analysis.gif
"""
import argparse
import io
import os

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax import serialization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vit_flax_nnx import create_vit_model, center_crop

EDGES_M = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
                    0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0, 1.2, 1.5, 2.0], dtype=float)
BIN_CENTERS_CM = np.array([(EDGES_M[d + 1] + EDGES_M[d]) / 2.0 for d in range(len(EDGES_M) - 1)]) * 100


def cdf2pdf(cdf_22pt):
    return np.diff(cdf_22pt)


def forward_with_attention_and_probs(m, image):
    if image.shape[1] != m.image_size or image.shape[2] != m.image_size:
        image = center_crop(image, m.image_size)
    x = m.patch_embedding(image)
    x = m.pos_encoding(x)
    attn_maps, layer_probs = [], []
    for block in m.transformer_blocks:
        normed = block.norm1(x)
        bb, s, ed = normed.shape
        qkv = block.attention.qkv(normed)
        q, k, v = [t.reshape(bb, s, block.attention.num_heads, block.attention.head_dim).transpose(0, 2, 1, 3)
                   for t in jnp.split(qkv, 3, axis=-1)]
        scale = 1.0 / jnp.sqrt(block.attention.head_dim)
        weights = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale, axis=-1)
        attn_maps.append(np.asarray(weights[0].mean(axis=0)))
        out = jnp.einsum("bhqk,bhvd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(bb, s, ed)
        x = x + block.attention.proj(out)
        x = x + block.mlp(block.norm2(x), training=False)
        pooled = jnp.mean(m.final_norm(x), axis=1)
        layer_probs.append(np.asarray(jax.nn.softmax(m.output_head(pooled), axis=-1))[0])
    return attn_maps, layer_probs


def entropy_certainty(probs):
    p = np.clip(probs, 1e-9, 1.0)
    return 1.0 - (-np.sum(p * np.log(p))) / np.log(len(p))


def make_gif(msgpack_path, data_npz_path, out_path, sample_idx=0, duration_ms=900):
    model = create_vit_model(image_size=224, bins=21, output_scalar=False, rngs=nnx.Rngs(0))
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    with open(msgpack_path, "rb") as f:
        params.replace_by_pure_dict(serialization.msgpack_restore(f.read()))
    model = nnx.merge(graphdef, params, rest)

    data = np.load(data_npz_path, allow_pickle=True)
    img = jnp.asarray(data["images"][sample_idx:sample_idx + 1].astype(np.float32) / 255.0)
    true_pdf = cdf2pdf(data["histograms"][sample_idx])

    attn_maps, layer_probs = forward_with_attention_and_probs(model, img)
    certainties = [entropy_certainty(p) for p in layer_probs]
    print("certainty per layer:", [round(c, 3) for c in certainties])

    cropped_img = np.asarray(center_crop(img, model.image_size))[0]
    h_p, w_p = model.h_patches, model.w_patches

    frames = []
    cmap = plt.cm.viridis
    for i in range(len(layer_probs)):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        ax0 = axes[0]
        ax0.imshow(cropped_img)
        received = attn_maps[i].sum(axis=0).reshape(h_p, w_p)
        ax0.imshow(np.kron(received, np.ones((224 // h_p, 224 // w_p))), alpha=0.55, cmap="inferno")
        ax0.set_title(f"layer {i+1}/{len(layer_probs)} attention", fontsize=10)
        ax0.axis("off")

        ax1 = axes[1]
        for j in range(i + 1):
            alpha = 0.25 + 0.55 * (j / max(i, 1)) if i > 0 else 0.9
            ax1.plot(BIN_CENTERS_CM, layer_probs[j], "-", color=cmap(j / max(len(layer_probs) - 1, 1)), alpha=alpha, linewidth=1.3)
        ax1.plot(BIN_CENTERS_CM, layer_probs[i], "s-", color=cmap(i / max(len(layer_probs) - 1, 1)), linewidth=2.4, markersize=4, label=f"layer {i+1} (current)")
        ax1.plot(BIN_CENTERS_CM, true_pdf, "o-", color="black", linewidth=2, markersize=4, label="ground truth")
        ax1.set_xlabel("grain diameter (cm)", fontsize=8); ax1.set_ylabel("probability mass", fontsize=8)
        ax1.set_title(f"certainty={certainties[i]:.3f}", fontsize=10)
        ax1.legend(fontsize=6, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False, disposal=2)
    print(f"saved {out_path} ({os.path.getsize(out_path)/1e3:.1f} KB, {len(frames)} frames)")
    return certainties


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msgpack", default="grainet_vit.msgpack")
    parser.add_argument("--data", default=os.path.expanduser("~/GRAINet_work/data_GRAINet_demo/data_KLEmme_1bank.npz"))
    parser.add_argument("--out", default="grainet_ctm_analysis.gif")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()
    make_gif(args.msgpack, args.data, args.out, sample_idx=args.sample_idx)
