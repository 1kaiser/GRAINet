"""Plain ViT-Tiny baseline for the 3-model comparison: identical
architecture to vit_flax_nnx.py's ViTTiny_STRING2D_Cayley_Flax (same
PatchEmbedding/MultiHeadAttention/MLP/TransformerBlock, reused directly),
except the positional encoding is a standard learned absolute embedding
instead of the STRING2D-Cayley rotary-style one -- built to answer
"does the Cayley encoding actually help over a standard ViT?"
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from einops import reduce

from vit_flax_nnx import PatchEmbedding, TransformerBlock, center_crop


class LearnedPositionalEncoding(nnx.Module):
    """Standard ViT absolute positional embedding -- one learned vector
    per patch position, added directly (no Cayley transform, no rotation)."""
    def __init__(self, num_patches: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (num_patches, embed_dim))
        )

    def __call__(self, inputs):
        return inputs + self.pos_embedding[...][None, :, :]


class ViTTiny_Plain_Flax(nnx.Module):
    """Same structure as ViTTiny_STRING2D_Cayley_Flax, standard positional
    encoding instead of STRING2D-Cayley."""
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 num_layers: int = 4,
                 mlp_dim: int = 768,
                 num_classes: int = 21,
                 dropout_rate: float = 0.1,
                 output_scalar: bool = False,
                 *, rngs: nnx.Rngs):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.output_scalar = output_scalar

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim, rngs=rngs)
        self.pos_encoding = LearnedPositionalEncoding(self.num_patches, embed_dim, rngs=rngs)
        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.output_head = nnx.Linear(embed_dim, 1 if output_scalar else num_classes, rngs=rngs)

    def __call__(self, x, training: bool = False):
        if x.shape[1] != self.image_size or x.shape[2] != self.image_size:
            x = center_crop(x, self.image_size)
        x = self.patch_embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        x = self.final_norm(x)
        x = reduce(x, "b s d -> b d", "mean")
        x = self.output_head(x)
        if not self.output_scalar:
            x = jax.nn.softmax(x, axis=-1)
        return x


def create_plain_vit_model(image_size: int = 224, bins: int = 21, output_scalar: bool = False, rngs=None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    return ViTTiny_Plain_Flax(image_size=image_size, patch_size=16, embed_dim=192, num_heads=3,
                               num_layers=4, mlp_dim=768, num_classes=bins, dropout_rate=0.1,
                               output_scalar=output_scalar, rngs=rngs)


if __name__ == "__main__":
    import numpy as np
    rngs = nnx.Rngs(42)
    model = create_plain_vit_model(bins=21, output_scalar=False, rngs=rngs)
    x = jnp.asarray(np.random.RandomState(0).rand(2, 500, 200, 3).astype(np.float32))
    out = model(x, training=False)
    print("output shape:", out.shape, "sums:", np.asarray(out).sum(axis=-1))
    params = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"params: {params:,}")
