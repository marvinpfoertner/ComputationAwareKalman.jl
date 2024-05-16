# ComputationAwareKalman.jl

**ComputationAwareKalman.jl** implements the *computation-aware Kalman filter* (CAKF) and the *computation-aware RTS smoother* (CAKS), novel approximate, probabilistic numerical versions of the Kalman filter and RTS smoother that are

1. *matrix-free* and *iterative*, and can fully leverage modern parallel hardware (i.e. GPUs);
2. *more efficient* than their standard versions, with **quadratic time** (worst-case) and **linear memory** complexities; and
3. computation-aware, i.e. they come with theoretical guarantees for their uncertainty estimates which capture the inevitable approximation error.

In our paper we have demonstrated the scalability of the approach by applying it to a state-space model with $\approx 230\mathrm{k}$ dimensions in the context of spatiotemporal GP regression of climate/weather data with about $4$ million data points.
The code for the experiments from the paper can be found in [ComputationAwareKalmanExperiments.jl](https://github.com/marvinpfoertner/ComputationAwareKalmanExperiments.jl).

## Citation

If you use this library, please cite our paper

```bibtex
@misc{Pfoertner2024CAKF,
  author = {Pf\"ortner, Marvin and Wenger, Jonathan and Cockayne, Jon and Hennig, Philipp},
  title = {Computation-Aware {K}alman Filtering and Smoothing},
  year = {2024},
  publisher = {arXiv},
  doi = {10.48550/arxiv.2405.08971},
  url = {https://arxiv.org/abs/2405.08971}
}
```
