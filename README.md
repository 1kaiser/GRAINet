# GRAINet: Mapping grain size distributions in river beds from UAV images with convolutional neural networks

<a href="https://colab.research.google.com/github/1kaiser/GRAINet/blob/main/GRAINet_ViT_comparison.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
     

This is a demonstration to show how to run the training and testing code used to analyze the grain size distributions in river beds over entire gravel bars ([Lang et al., 2020]( https://doi.org/10.5194/hess-2020-196)).

Along with the code, we release a small subset of the dataset used in the study, which is a single gravel bar (orthophoto) with 212 manually annotated image tiles. The full dataset cannot be published for commercial reasons, as it is owned by a private company (who also created it at their own cost).

**Important note**: The trained model resulting from this demo will not generalize as described in the paper.

**On-device / in-browser deployment**: the ViT-Tiny (`vit_flax_nnx.py`)
model has been trained, converted to LiteRT (`.tflite`) via `jax2tf`, and
verified end-to-end (correlation 0.99999965 vs. JAX on every real test
tile), including a self-contained single-page web demo running the model
directly in-browser via LiteRT.js. It's also grounded in standard
[soil gradation](https://en.wikipedia.org/wiki/Soil_gradation) theory
(D10/D30/D60, Cu, Cc, USCS classification), computed from this repo's own
real demo-dataset gradation curves — including a genuine, non-obvious
finding that `dm` (Fehr 1987) is not the same metric as the more familiar
D50. Full writeup, notebook, and web demo:
[1kaiser/test2026/grainet_tryout](https://github.com/1kaiser/test2026/tree/main/grainet_tryout)
(6 real version-drift/conversion bugs found and fixed along the way, one
of them — `nnx.List` for module lists — already applied to
`vit_flax_nnx.py` in this repo).

**3-model x 7-loss comparison** (added alongside): `vit_plain_flax_nnx.py`
(a plain ViT-Tiny with standard positional encoding, built as an ablation
against the Cayley-STRING encoding), `resnet_architecture_flax_nnx.py` (a
JAX/Flax NNX port of this repo's own `resnet_architecture.py`),
`loss_functions_jax.py` (JAX ports of all 5 losses in
`loss_functions.py` — KL/reverse-KL/JSD, EMD, volume-weighted MAE/MSE —
plus plain MSE), and `helper_jax.py` (JAX-compatible CDF→PDF conversion
and an exact port of `helper.py::get_dm`, verified bit-exact against this
repo's own stored `dm` on real ground-truth data). Real, verified
findings from training all 21 combinations: plain MSE is the worst loss
for every model; ResNet/FCN wins on quality despite training 6-7x slower
per step on GPU than either ViT; **both ViT variants show a genuine mode
collapse** (near-identical predictions across all test tiles) on this
212-tile dataset; and `get_dm`'s Fuller-curve derivation is itself
unstable on imperfect predictions (a 0.01 L1 perturbation can swing it by
30+ cm) — full tables, plots, and discussion in the
[test2026 writeup](https://github.com/1kaiser/test2026/tree/main/grainet_tryout#section-7-3-model-x-7-loss-comparison-matching-the-original-papers-methodology).

**CTM-inspired attention/certainty analysis**: adapted plotting ideas from
[SakanaAI's Continuous Thought Machine](https://github.com/SakanaAI/continuous-thought-machines)
(per-tick attention overlays, a certainty-over-time curve) to this ViT's
layer depth as the natural analog of CTM's internal "tick" axis. Real
finding: attention starts sparse/localized and spreads by the final
layer, while early-exit certainty genuinely *decreases* with depth — the
opposite of CTM's own curves, because the output head was only ever
trained on the last layer's features. Full discussion and plots in
[test2026's Section 8](https://github.com/1kaiser/test2026/tree/main/grainet_tryout#section-8-ctm-inspired-analysis--attention-and-certainty-over-depth).

**GPU training, quantized LiteRT, and an animated GIF** (latest update):
found a real gap — the notebook's own committed `_executed.ipynb` had
only ever run 13 of its cells (stale, from before the sections above
existed), so the 21-combo sweep and CTM analysis had only ever been
verified via separate scratch scripts. Fixed with the first true
end-to-end run: training moved to GPU, LiteRT conversion isolated into
its own CPU subprocess (a real cross-environment serialization-version
mismatch, not fixable by a config flag), plus a fixed missing `cv2`
import and two release-verification bugs (a hard assert that doesn't
account for cross-run training non-determinism, and a msgpack-clobbering
download path). Added dynamic-range + full-integer LiteRT quantization
(dynamic-range wins clearly: 2.7x smaller, 1.8x faster, <0.01cm error)
and a CTM-style animated GIF (one frame per layer: attention overlay +
accumulating histogram prediction + certainty score). Full writeup:
[test2026's Performance section](https://github.com/1kaiser/test2026/tree/main/grainet_tryout#performance).


## Getting Started

1) Clone this repository to your local machine. In your terminal type:
    ```
    git clone URL
    ```
   
2) Download the data from [here](https://share.phys.ethz.ch/~pf/nlangdata/GRAINet_demo_data.zip).

    Move the data folder into the GRAINet directory. The directory tree should look like this: `GRAINet/data_GRAINet_demo/`

## Prerequisites
This code uses keras with a tensorflow backend. GDAL is used to predict for georeferenced orthophotos.
The following instructions will guide you to install:

* python3
* jupyter
* tensorflow
* keras
* gdal

## Installing
We recommend to install python via anaconda and to create a new conda environment.

1) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) and read the [Anaconda tutorial](https://conda.io/docs/user-guide/getting-started.html)

2) Create a new environment: ```conda create --name GRAINenv python=3.7.1```

3) Activate the new conda environment (for conda 4.6 and later versions)
    * Windows: ```conda activate GRAINenv```
    * Linux and macOS: ```conda activate GRAINenv```
    
    For versions prior to conda 4.6, use:
    * Windows: ```activate GRAINenv```
    * Linux, macOS: ```source activate GRAINenv```
    
    --> now your terminal prompt should start with ***(GRAINenv)*** 
    
4) Install the following packages in your activated GRAINenv:
    ```
    conda install jupyter
    conda install matplotlib
    conda install keras=2.2.4
    conda install h5py=2.9.0
    conda install scikit-image
    ```
    
5) Install ***tensorflow*** with anaconda or follow the [official tensorflow installation instructions (e.g. with pip)](https://www.tensorflow.org/install/pip)
    or check the [anaconda installation instructions](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/).
    ```
    conda install tensorflow-gpu=1.13.1
    ```
      
6) Install ***GDAL*** with anaconda or follow the [official GDAL installation instructions](https://gdal.org/download.html)
   ```
   conda install -c conda-forge gdal
   ``` 

## Verify your installation

Run `python` in your conda GRAINenv. Then:
```
import keras
import tensorflow
import h5py
from osgeo import gdal
```

The code has been tested with the following versions:
```
keras.__version__
'2.2.4'
tensorflow.__version__
'1.13.1'
h5py.__version__
'2.9.0'
```

## Run the notebook
Open the jupyter notebook:
```
jupyter notebook GRAINet_demo_dm_regression.ipynb
```

## Citation

If you use this code please cite our paper: 

*Lang, Nico, Andrea Irniger, Agnieszka Rozniak, Roni Hunziker, Jan Dirk Wegner, and Konrad Schindler. "GRAINet: mapping grain size distributions in river beds from UAV images with convolutional neural networks." Hydrology and Earth System Sciences 25, no. 5 (2021): 2567-2597.*

BibTex:

```
@article{lang2021grainet,
  title={GRAINet: mapping grain size distributions in river beds from UAV images with convolutional neural networks},
  author={Lang, Nico and Irniger, Andrea and Rozniak, Agnieszka and Hunziker, Roni and Wegner, Jan Dirk and Schindler, Konrad},
  journal={Hydrology and Earth System Sciences},
  volume={25},
  number={5},
  pages={2567--2597},
  year={2021},
  publisher={Copernicus GmbH}
}
```





