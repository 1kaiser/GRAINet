"""Convert a trained GRAINet ViT-Tiny (vit_flax_nnx.py) to LiteRT, in three
forms: float32, dynamic-range-quantized int8, and full-integer-quantized
int8 -- each verified against the live JAX model, not just reported by
file size.

Runs the conversion in-process (this script itself should be run under a
CPU-compatible TensorFlow environment, e.g. `moge_litert` in the sibling
test2026/grainet_tryout notebook) since a very-new GPU-enabled JAX build's
StableHLO serialization version can be newer than what jax2tf's
TensorFlow-based TFLiteConverter supports -- see that notebook's Section 3
("Bug #7") for the full story and a subprocess-isolation pattern if you
need to train on GPU and convert in the same run.

Usage:
    python litert_quantization.py --msgpack grainet_vit.msgpack --data data_KLEmme_1bank.npz
"""
import argparse
import os
import time

import numpy as np
import jax
jax.config.update("jax_use_shardy_partitioner", False)  # Bug #5: Shardy dialect not recognized by this TF's StableHLO deserializer
import jax.numpy as jnp
import flax.nnx as nnx
from flax import serialization
from einops import rearrange, reduce as einops_reduce

from vit_flax_nnx import create_vit_model, center_crop


def build_precomputed_apply_fn(model):
    """Bug #6 fix: STRING2DPositionalEncoding's Cayley transform calls
    jnp.linalg.solve on every forward pass -- no TFLite kernel exists for
    it. The inverse depends only on trained parameters, never the input,
    so precompute it once and substitute a plain broadcast-add."""
    pe = model.pos_encoding
    h_coords, w_coords = jnp.meshgrid(jnp.arange(pe.h_patches), jnp.arange(pe.w_patches), indexing="ij")
    positions = rearrange([h_coords, w_coords], "coord h w -> (h w) coord")
    x_pos_enc = pe.x_encoding[positions[:, 1]]
    y_pos_enc = pe.y_encoding[positions[:, 0]]
    combined_encoding = jnp.concatenate([x_pos_enc, y_pos_enc], axis=-1)
    S_antisym = pe.S_param[...] - pe.S_param[...].T
    I = jnp.eye(pe.embed_dim)
    P = jnp.linalg.solve(I + S_antisym, I - S_antisym)
    pos_encoding_const = combined_encoding @ P

    def apply_fn(x):
        if x.shape[1] != model.image_size or x.shape[2] != model.image_size:
            x = center_crop(x, model.image_size)
        x = model.patch_embedding(x)
        x = x + pos_encoding_const[None, :, :]
        for block in model.transformer_blocks:
            x = block(x, training=False)
        x = model.final_norm(x)
        x = einops_reduce(x, "b s d -> b d", "mean")
        return model.output_head(x)

    return apply_fn


def convert_and_verify(msgpack_path, data_npz_path, out_dir, n_verify_tiles=22):
    model = create_vit_model(image_size=224, bins=21, output_scalar=True, rngs=nnx.Rngs(0))
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    with open(msgpack_path, "rb") as f:
        params.replace_by_pure_dict(serialization.msgpack_restore(f.read()))
    model = nnx.merge(graphdef, params, rest)
    apply_fn = build_precomputed_apply_fn(model)

    sample = jnp.asarray(np.random.RandomState(0).rand(1, 500, 200, 3).astype(np.float32))
    _ = apply_fn(sample)

    from jax.experimental import jax2tf
    import tensorflow as tf

    tf_fn = jax2tf.convert(apply_fn, enable_xla=False, with_gradient=False)
    module = tf.Module()
    module.f = tf.function(tf_fn, input_signature=[tf.TensorSpec((1, 500, 200, 3), tf.float32)], autograph=False)
    concrete_fn = module.f.get_concrete_function()

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)
    f32_bytes = converter.convert()
    f32_path = os.path.join(out_dir, "grainet_vit.tflite")
    with open(f32_path, "wb") as f:
        f.write(f32_bytes)
    print(f"[float32]           {time.time()-t0:.1f}s -> {f32_path} ({len(f32_bytes)/1e6:.2f} MB)")

    converter_dynq = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)
    converter_dynq.optimizations = [tf.lite.Optimize.DEFAULT]
    dynq_bytes = converter_dynq.convert()
    dynq_path = os.path.join(out_dir, "grainet_vit_dynamic_int8.tflite")
    with open(dynq_path, "wb") as f:
        f.write(dynq_bytes)
    print(f"[dynamic-range int8] -> {dynq_path} ({len(dynq_bytes)/1e6:.2f} MB)")

    data = np.load(data_npz_path, allow_pickle=True)
    calib_images = (data["images"][:100].astype(np.float32)) / 255.0

    def representative_dataset():
        for i in range(len(calib_images)):
            yield [calib_images[i:i + 1]]

    converter_fi = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)
    converter_fi.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fi.representative_dataset = representative_dataset
    converter_fi.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    fi_bytes = converter_fi.convert()
    fi_path = os.path.join(out_dir, "grainet_vit_full_int8.tflite")
    with open(fi_path, "wb") as f:
        f.write(fi_bytes)
    print(f"[full-int8]          -> {fi_path} ({len(fi_bytes)/1e6:.2f} MB)")

    from ai_edge_litert.interpreter import Interpreter

    X_verify = (data["images"][:n_verify_tiles].astype(np.float32)) / 255.0

    def verify(tflite_path, label):
        interp = Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        inp, out = interp.get_input_details(), interp.get_output_details()
        diffs, times_ms = [], []
        for i in range(len(X_verify)):
            x = X_verify[i:i + 1]
            jax_out = float(np.asarray(apply_fn(jnp.asarray(x))).squeeze())
            in_data = x
            if inp[0]["dtype"] != np.float32:
                scale, zp = inp[0]["quantization"]
                in_data = np.round(x / scale + zp).astype(inp[0]["dtype"])
            interp.set_tensor(inp[0]["index"], in_data)
            t0 = time.time(); interp.invoke(); times_ms.append((time.time() - t0) * 1000)
            tfl_out = interp.get_tensor(out[0]["index"])
            if out[0]["dtype"] != np.float32:
                scale, zp = out[0]["quantization"]
                tfl_out = (tfl_out.astype(np.float32) - zp) * scale
            diffs.append(abs(float(tfl_out.squeeze()) - jax_out))
        print(f"[{label:19s}] size={os.path.getsize(tflite_path)/1e6:.3f}MB "
              f"mean_invoke_ms={np.mean(times_ms):.3f} max_abs_diff_vs_jax={max(diffs):.6f} "
              f"mean_abs_diff={np.mean(diffs):.6f}")

    verify(f32_path, "float32")
    verify(dynq_path, "dynamic-range int8")
    verify(fi_path, "full-int8")

    return f32_path, dynq_path, fi_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msgpack", default="grainet_vit.msgpack")
    parser.add_argument("--data", default=os.path.expanduser("~/GRAINet_work/data_GRAINet_demo/data_KLEmme_1bank.npz"))
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    convert_and_verify(args.msgpack, args.data, args.out_dir)
