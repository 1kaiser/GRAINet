"""JAX/Flax NNX port of langnico/GRAINet's original FCN_grainsize
(resnet_architecture.py, Keras) -- same conv_block/identity_block
structure and filter counts as the source (stage2: 64/64/256, stage3+4:
BOTH 128/128/512, not the canonical ResNet50 doubling), adapted to output
a 21-bin softmax histogram (matching the real data convention: 22-point
CDF -> 21-bin PDF via np.diff) instead of Keras's own 22-bin default.
"""
import jax.numpy as jnp
import flax.nnx as nnx


class ConvBlock(nnx.Module):
    def __init__(self, in_c: int, kernel_size: int, filters, strides: int, rngs: nnx.Rngs):
        F1, F2, F3 = filters
        self.conv2a = nnx.Conv(in_c, F1, (1, 1), strides=(strides, strides), rngs=rngs)
        self.bn2a = nnx.BatchNorm(F1, rngs=rngs)
        self.conv2b = nnx.Conv(F1, F2, (kernel_size, kernel_size), strides=(1, 1), padding="SAME", rngs=rngs)
        self.bn2b = nnx.BatchNorm(F2, rngs=rngs)
        self.conv2c = nnx.Conv(F2, F3, (1, 1), strides=(1, 1), padding="VALID", rngs=rngs)
        self.bn2c = nnx.BatchNorm(F3, rngs=rngs)
        self.conv_shortcut = nnx.Conv(in_c, F3, (1, 1), strides=(strides, strides), padding="VALID", rngs=rngs)
        self.bn_shortcut = nnx.BatchNorm(F3, rngs=rngs)

    def __call__(self, x, train: bool):
        h = nnx.relu(self.bn2a(self.conv2a(x), use_running_average=not train))
        h = nnx.relu(self.bn2b(self.conv2b(h), use_running_average=not train))
        h = self.bn2c(self.conv2c(h), use_running_average=not train)
        shortcut = self.bn_shortcut(self.conv_shortcut(x), use_running_average=not train)
        return nnx.relu(h + shortcut)


class IdentityBlock(nnx.Module):
    def __init__(self, in_c: int, kernel_size: int, filters, rngs: nnx.Rngs):
        F1, F2, F3 = filters
        assert F3 == in_c
        self.conv2a = nnx.Conv(in_c, F1, (1, 1), strides=(1, 1), rngs=rngs)
        self.bn2a = nnx.BatchNorm(F1, rngs=rngs)
        self.conv2b = nnx.Conv(F1, F2, (kernel_size, kernel_size), strides=(1, 1), padding="SAME", rngs=rngs)
        self.bn2b = nnx.BatchNorm(F2, rngs=rngs)
        self.conv2c = nnx.Conv(F2, F3, (1, 1), strides=(1, 1), padding="VALID", rngs=rngs)
        self.bn2c = nnx.BatchNorm(F3, rngs=rngs)

    def __call__(self, x, train: bool):
        h = nnx.relu(self.bn2a(self.conv2a(x), use_running_average=not train))
        h = nnx.relu(self.bn2b(self.conv2b(h), use_running_average=not train))
        h = self.bn2c(self.conv2c(h), use_running_average=not train)
        return nnx.relu(h + x)


class FCN_GRAINet(nnx.Module):
    def __init__(self, num_bins: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 64, (3, 3), strides=(1, 1), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)

        self.stage2_conv = ConvBlock(64, 3, [64, 64, 256], strides=2, rngs=rngs)
        self.stage2_id = IdentityBlock(256, 3, [64, 64, 256], rngs=rngs)

        self.stage3_conv = ConvBlock(256, 3, [128, 128, 512], strides=2, rngs=rngs)
        self.stage3_id = IdentityBlock(512, 3, [128, 128, 512], rngs=rngs)

        self.stage4_conv = ConvBlock(512, 3, [128, 128, 512], strides=2, rngs=rngs)
        self.stage4_id = IdentityBlock(512, 3, [128, 128, 512], rngs=rngs)

        self.conv_out = nnx.Conv(512, num_bins, (1, 1), strides=(1, 1), padding="SAME", rngs=rngs)

    def __call__(self, x, train: bool = True):
        h = nnx.relu(self.bn1(self.conv1(x), use_running_average=not train))
        h = self.stage2_id(self.stage2_conv(h, train), train)
        h = self.stage3_id(self.stage3_conv(h, train), train)
        h = self.stage4_id(self.stage4_conv(h, train), train)
        h = self.conv_out(h)
        h = h.mean(axis=(1, 2))          # GlobalAveragePooling2D
        return nnx.softmax(h, axis=-1)   # histogram_prediction, matches original's softmax exactly


def count_params(model):
    import jax
    return sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


if __name__ == "__main__":
    import numpy as np
    rngs = nnx.Rngs(42)
    model = FCN_GRAINet(num_bins=21, rngs=rngs)
    x = jnp.asarray(np.random.RandomState(0).rand(2, 500, 200, 3).astype(np.float32))
    out = model(x, train=False)
    print("output shape:", out.shape, "sums:", np.asarray(out).sum(axis=-1))
    print(f"params: {count_params(model):,}")
