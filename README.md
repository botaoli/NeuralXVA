# CVA Sensitivities, Hedging and Risk

This repository contains codes accompanying the paper **"CVA Sensitivities, Hedging and Risk"** by Stéphane Crépey, Botao Li, Hoang Nguyen, Bouazza Saadeddine (see citation below, paper available: <https://www.risk.net/cutting-edge/7959752/cva-sensitivities-hedging-and-risk>). It is based on the repository [NeuralXVA](https://github.com/BouazzaSE/NeuralXVA), and will be integrated in the future versions of NeuralXVA. This repository provides:

* the generation via Monte Carlo of paths of diffusion risk factors, default indicators and mark-to-markets in a multi-dimensional setup with multiple economies and counterparties on GPU, with the possibility of inserted pathwise model parameters (i.e. exogenous model parameters and initial conditions of model risk factors);
* the learning of a path-wise CVA using a Neural Network regression based on the generated Monte Carlo samples of the payoffs, with tweaks for run-on mode ([`CVA_learning_risk_mode.ipynb`](CVA_learning_risk_mode.ipynb)) and run-off mode ([`CVA_learning_sensis_mode.ipynb`](CVA_learning_sensis_mode.ipynb)).
* the computation of the $\Delta$ and $\Gamma$ bump sensitivities for CVA (demonstrated in [`CVA_bump_sensis.ipynb`](CVA_bump_sensis.ipynb)).
* Run-on [`CVA_hedging_run-on.ipynb`](CVA_hedging_run-on.ipynb) and run-off [`CVA_hedging_run-off.ipynb`](CVA_hedging_run-off.ipynb) CVA risk and hedging with market products.  
* implementing and benchmarking all fast bump sensitivities methods in the paper for multi-asset call option in the Black-Scholes model ([`BS_demo.ipynb`](BS_demo.ipynb))

## New features compared with NeuralXVA

This repository is derived from [NeuralXVA](https://github.com/BouazzaSE/NeuralXVA), retaining the key technical aspects of the original project. We encourage the interested user to check <https://github.com/BouazzaSE/NeuralXVA/blob/main/README.md> for technical details and simulation conventions. To adapt the framework for CVA risk and hedging, we augmented the CVA simulation and learning with the following new features:

* off-grid pricing: we add `early_pricing_date` paramter which enables pricing at a finer step (e.g. 0.01yr) compared to the current regular timestep (e.g. 0.1yr) without the need for a denser grid. This reduces simulation costs by avoiding unnecessary pricing computations at later dates;
* pathwise diffusion parameters: by passing a matrix of shocked parameters (with parameters as column and shocked scenarios as row) into the `DiffusionEngine` either at initialization or using the method `_gen_diff_params()`, simulations can run with different parameters on each path;
* reinitializing without redefining diffusion engine: by using the method `_reinitialize()` in `DiffusionEngine`, one can reset the initial values of the simulation and reapply the pathwise shock. It allows reusing the compiled CUDA kernels, thus reducing the overhead when running diffusions repeatedly. Notice that we need to toggle off `set_irs_at_par` in the `generate_batch()` method to avoid changing interest rate swap specifications after `_reinitialize()`;
* resetting the RNG state without redefining diffusion engine: the method `reset_rng_states()` in `DiffusionEngine` allows user to specify the seed for random numbers appears in simulation;
* changing the random seed during the simulation: the random number sequence for risk factors and default indicators simulation is changed to `seed_to_change` after `time_to_change_seed` when running the `generate_batch()` method. This allows to compute the nested CVA in the sensis mode. Notice that the nested computation integrated inside `DiffusionEngine` can only be used to computed the nested CVA at date $t$ in the baseline mode.

Also, we implement the following features that are independent of CVA learning and simulation:

* methods of derivative calculation for prices: we implemented the benchmark, smart and linear bump sensitivities of CVA;
* mapping sensitivities with respect to parameters to sensitivities with respect to products: using the implicit function theorem, we calculate the jacobian matrix of sensitivities. To support these features, we have included (analytical) pricing functions of zero coupon bounds and credit default swaps (CDS), along with their pseudo calibration losses under the assumption that model paramters are perfectly calibrated by these market products;
* expected shortfall regression: we implemented the linear regression using expected shortfall as loss and Adam as optimizer in [`tool.py`](tool.py).

## Running the notebooks

We keep most of the utilities for CVA hedging in notebooks independent from CVA simulation and learning. Some notebook requires the results of other notebooks to run. The dependencies are illustrated as follows:

```text
learning_risk_mode   bump_sensis   learning_sensis_mode
     | _______________/      \_____________ |
     |/                                    \|
hedging_run-off                    hedging_run-on 
```

To be more specific:

* The [`CVA_bump_sensis.ipynb`](CVA_bump_sensis.ipynb) generates the bump sensitivities using benchmark, smart and linear bump method, and store them in the file `{number of economies}_{number of spreads}_{seed}.pickle`. By default, there are 10 economy and 9 default spreads, and the output will be `10_9_0.pickle` with the seed of 0;
* The [`CVA_learning_risk_mode.ipynb`](CVA_learning_risk_mode.ipynb) generate features and labels for CVA run off analysis, storing them in `data_for_run-off_analysis.pickle`;
* The [`CVA_learning_sensis_mode.ipynb`](CVA_learning_sensis_mode.ipynb) generate features and labels for CVA run off analysis, storing them in `data_for_run-on_analysis.pickle`;
* The [`CVA_hedging_run-off.ipynb`](CVA_hedging_run-off.ipynb) requires `{number of economies}_{number of spreads}_{seed}.pickle` and `data_for_run-off_analysis.pickle` as input;
* The [`CVA_hedging_run-on.ipynb`](CVA_hedging_run-on.ipynb) requires `{number of economies}_{number of spreads}_{seed}.pickle` and `data_for_run-on_analysis.pickle` as input.

In general, run the sensis and learning notebooks before proceeding to hedging notebooks.

## Citing

If you use this code in your work, we strongly encourage you to both cite this Github repository (with the corresponding identifier for the commit you are looking at) and the papers describing our learning schemes:

```latex
@article{crepey2024cva,
  title={CVA Sensitivities, Hedging and Risk},
  author={Cr{\'e}pey, St{\'e}phane and Li, Botao and Nguyen, Hoang and Saadeddine, Bouazza},
  journal={arXiv preprint arXiv:2407.18583},
  year={2024}
}
```

## Working versions of packages

The code has been tested with the following package versions:

* `numpy = 1.26.4`;
* `torch = 2.2.2`;
* `matplotlib = 3.8.0`
* `pandas = 2.2.1`;

and Python version `3.11.8`.
