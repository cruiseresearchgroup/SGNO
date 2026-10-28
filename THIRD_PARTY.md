# Third-party software

SGNO's source is released under the MIT license in `LICENSE`.
Data generation uses upstream APEBench, Exponax, JAX, Equinox, Optax, PDEquinox, and Trainax.
These dependencies are not vendored; their original licenses and notices apply.

- APEBench: https://github.com/Ceyron/apebench (version 0.1.1; commit
  `dca24a00eeacbe5524a99542f6d5c2bc2deb8da8`)
- Exponax: https://github.com/Ceyron/exponax (version 0.1.0)
- PyTorch: https://github.com/pytorch/pytorch

The included FNO control follows the Fourier neural operator construction and the
APEBench-style spectral block. It is the matched corrector control, not a replacement
for the separately recorded external FNO/CNO/KNO/IS-FNO experiments.
The paper cites those methods and specifies their comparison objectives and capacities.
