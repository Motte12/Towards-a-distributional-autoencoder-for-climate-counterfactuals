# Towards-a-distributional-autoencoder-for-climate-counterfactuals

This repository contains code to reproduce the results for [Probabilistic storyline attribution using machine learning](https://arxiv.org/abs/2606.02550).

## Repository Structure

```
Towards-a-distributional-autoencoder-for-climate-counterfactuals/
├── README.md
├── LICENSE
├── environment.yml                 # Conda environment file
├── settings.json                   # Global settings
├── src/
│   ├── modeling/                   # Core modeling code
│   │   ├── __init__.py
│   │   ├── create_ensemble.sh      # Bash script to start create_test_ensemble.py
│   │   ├── create_test_ensemble.py # Create an ensemble from a trained model
│   │   ├── pca_encoder.py          # PCA encoder implementation
│   │   ├── start_joint_training.sh # Launch training script for DAE
│   │   └── train_joint_dae.py      # Train the model
│   ├── analysis/                   # Model output analysis
│   │   ├── __init__.py
│   │   ├── extended_abstract_figure.ipynb # Figure for extended abstract
│   │   ├──Figure03.ipynb
│   │   ├──Figure03_CF.ipynb
│   │   ├──2028_2053_ERA5A_attribution_analysis.ipynb
│   │   └── quantile_regression/                        # contains all baseline quantile regression related code
│   │       ├── evaluate_pytorch_quantile_regression.py # compare DAE and baseline in regional domain
│   │       ├── pytorch_quantile_regression.py          # train baseline quantile regression
│   │       ├── run_baseline_evaluation.sh              # start baseline comparison
│   │       └── submit_pytorch_quantile_regression.sh   # start quantile regression training
│   └── utils/                      # Shared helper functions
│       ├── __init__.py
│       ├── utils.py                # Data processing and visualization utilities
│       ├── dpa_ensemble.py         # DPA ensemble utilities
│       └── evaluation.py           # Evaluation metrics
└── _devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001_pene0 # Pre-trained model
```


## Workflow to reproduce the manuscript figures

0. Code and data setup
    - clone this repository
    - get the data from ...
    - in `settings.json` adjust the paths to the data folder 
2. Create a conda environement using the environment.yaml file ([explained here](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-a-file))
3. Either reproduce the exact results from the paper (directly jump to step 7) or train the models from scratch (continue with step 4, note that results will slightly differ from the paper due to stochasticity)
---
(trainig from scratch)

4. Train the DAE model from scratch
   - start model training by running start_joint_training.sh (this trains the model specified in settings.json)
   - (optional: set your own model and training parameters, then also adjust `["current_model"]` and `["epochs"]` in `settings.json`) 
5. Generate factual and counterfactual ensembles
   - in `src/modeling/create_ensemble.sh`, adjust
       - the conda envrionment name in line 5 to the name of your conda environment
   - optional
       - adjust location for saving the generated ensemble `save_path=` (default is in the model directory)
   - execute `create_ensemble.sh` and `create_ERA5_test_ensemble.sh` to create the test and ERA5 ensembles (potentially need to make it executable before `chmod +x create_ensemble.sh`) (use slurm depending on your resources)
6. Train the baseline quantile regression models by submitting  `src/analysis/quantile_regression/submit_pytorch_quantile_regression.sh`, this executes `src/analysis/quantile_regression/pytorch_quantile_regression.py`
---
7. Manuscript figures
    - **Figure 2**
        + run `src/analysis/DAE_evaluation/evaluate_v5_ETH_DAE_ensemble.sh` to produce data (change line 5 to your environment)
        + run `src/analysis/DAE_evaluation/plot_data.ipynb` to create Figure 2
    - **Figure 3**
        + for subplots a) and c), run `src/analysis/Figure03.ipynb` (`src/analysis/Figure03_CF.ipynb` respectively)
        + for subplots b) and d)
            + run `src/analysis/quantile_regression/run_baseline_evaluation.sh` (runs `src/analysis/quantile_regression/evaluate_pytorch_quantile_regression.py`) (again change line 5 to your environment and check that path in `src/analysis/quantile_regression/run_baseline_evaluation.sh` line 43 corresponds to the paths where the QR baseline is saved)
            + this produces calibration curves in the subfolder '/src/analysis/quantile_regression/qr_baseline_eval_results'
    - **Figure 4**
        + run `src/analysis/2028_2053_ERA5_attribution_analysis.ipynb`


#### License
This project uses a small portion of code from [this framework](https://github.com/xwshen51/engression) by Xinwei Shen and Nicolai Meinshausen, which is licensed under the BSD 3-Clause License.
