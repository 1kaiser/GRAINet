"""JAX ports of langnico/GRAINet's original loss_functions.py, operating
on 21-bin PDFs (y_true, y_pred both non-negative, summing to ~1 -- model
outputs are already softmax-normalized). Same math as the original Keras
implementation, translated to jax.numpy.
"""
import jax.numpy as jnp

EPS = 1e-7


def mse_loss(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)


def mae_loss(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))


def kl_loss(y_true, y_pred):
    """Kullback-Leibler divergence KL(true || pred), per original KL()."""
    yt = jnp.clip(y_true, EPS, 1.0)
    yp = jnp.clip(y_pred, EPS, 1.0)
    return jnp.mean(jnp.sum(yt * jnp.log(yt / yp), axis=-1))


def reverse_kl_loss(y_true, y_pred):
    """Original reverseKL(y_true, y_pred) = keras KLD(y_pred, y_true) = KL(pred || true)."""
    yt = jnp.clip(y_true, EPS, 1.0)
    yp = jnp.clip(y_pred, EPS, 1.0)
    return jnp.mean(jnp.sum(yp * jnp.log(yp / yt), axis=-1))


def jsd_loss(y_true, y_pred):
    """Jensen-Shannon divergence, per original JSD()."""
    m = 0.5 * (y_true + y_pred)
    m = jnp.clip(m, EPS, 1.0)
    yt = jnp.clip(y_true, EPS, 1.0)
    yp = jnp.clip(y_pred, EPS, 1.0)
    kl_tm = jnp.sum(yt * jnp.log(yt / m), axis=-1)
    kl_pm = jnp.sum(yp * jnp.log(yp / m), axis=-1)
    return jnp.mean(0.5 * kl_tm + 0.5 * kl_pm)


def emd_loss(y_true, y_pred):
    """Earth Mover's Distance for 1D histograms via CDF L1, per original emd()."""
    cdf_true = jnp.cumsum(y_true, axis=-1)
    cdf_pred = jnp.cumsum(y_pred, axis=-1)
    return jnp.mean(jnp.sum(jnp.abs(cdf_true - cdf_pred), axis=-1))


def _get_mean_size_squared():
    """Exact port of loss_functions.py::get_mean_size_squared -- volume
    proxy (bin-center-diameter squared) used to weight larger grains more
    heavily, since they represent disproportionately more physical mass."""
    import numpy as np
    edges = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
                      0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0, 1.2, 1.5, 2.0], dtype=float)
    n_bins = len(edges) - 1
    dmi = np.array([(edges[d + 1] + edges[d]) / 2.0 for d in range(n_bins)])
    return np.power(dmi, 2)


_WEIGHTS = _get_mean_size_squared()
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def mae_weighted_loss(y_true, y_pred):
    w = jnp.asarray(_WEIGHTS, dtype=y_true.dtype)
    return jnp.mean(jnp.abs(w * (y_pred - y_true)))


def mse_weighted_loss(y_true, y_pred):
    w = jnp.asarray(_WEIGHTS, dtype=y_true.dtype)
    return jnp.mean((w * (y_pred - y_true)) ** 2)


LOSS_FNS = {
    "mse": mse_loss,
    "emd": emd_loss,
    "maew": mae_weighted_loss,
    "msew": mse_weighted_loss,
    "kld": kl_loss,
    "rkl": reverse_kl_loss,
    "jsd": jsd_loss,
}


# metrics computed at eval time regardless of training loss (numpy, per-sample)
def calculate_iou_np(y_true, y_pred):
    inter = np.minimum(y_true, y_pred).sum()
    union = np.maximum(y_true, y_pred).sum()
    return inter / union


import numpy as np  # noqa: E402  (used by calculate_iou_np above)


if __name__ == "__main__":
    import numpy as np
    rng = np.random.RandomState(0)
    a = rng.dirichlet(np.ones(21))
    b = rng.dirichlet(np.ones(21))
    a_j, b_j = jnp.asarray(a), jnp.asarray(b)
    for name, fn in LOSS_FNS.items():
        print(f"{name}: {float(fn(a_j, b_j)):.6f}")
    print(f"iou: {calculate_iou_np(a, b):.6f}")
    print(f"self-KL (should be ~0): {float(kl_loss(a_j, a_j)):.8f}")
