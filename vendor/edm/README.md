# EDM source used by the ERA5-Land experiments

This directory contains the parts of [NVlabs/edm](https://github.com/NVlabs/edm) needed by the ERA5-Land temperature downscaling scripts. The upstream source was taken from commit `008a4e5316c8e3bfe61a62f874bddba254295afb`. The original NVIDIA license is preserved in [LICENSE.txt](LICENSE.txt): Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

The upstream files here are `training/networks.py`, `torch_utils/persistence.py`, `dnnlib/util.py`, and their package initializers. The `training/climate_*` and `training/flow_*` files are project-specific conditional-model extensions. The training and evaluation entry points are in [`scripts/`](../../scripts/).

This is a source snapshot, not a Git submodule. It allows the scripts to import `training`, `torch_utils`, and `dnnlib` without a separate checkout of EDM. The upstream license terms apply to the NVIDIA code and any derivatives of it; consult the included license before reuse.
