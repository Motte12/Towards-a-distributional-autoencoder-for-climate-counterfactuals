# Towards-a-distributional-autoencoder-for-climate-counterfactuals

This repository contains code to reproduce the results for the extended abstract *Towards a distributional autoencoder for climate counterfactuals* submitted to the Climate Informatics 2026 conference.

## Repository Structure

```
Towards-a-distributional-autoencoder-for-climate-counterfactuals/
├── README.md
├── LICENSE
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
│   │   └── extended_abstract_figure.ipynb # Figure for extended abstract
│   └── utils/                      # Shared helper functions
│       ├── __init__.py
│       ├── utils.py
│       ├── dpa_ensemble.py
│       └── evaluation.py
├── environment.yml                 # Conda environment file
└── _devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0.0001_pene0 # Pre-trained model
```


## Instructions


### Data setup
- get the training and test data from **zenodo link**
- create a data directory (arbitrary name) and put the data there (don't change names of the datasets)
- insert the data directory name into `settings.json` in ['paths']['data']
- adjust the paths in settings.json

### Workflow to reproduce the extended abstract figure

1. create a conda environement using the environment.yaml file ([explained here](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file))
2. Train the model (or skip this, directly go to step 2 and use the pretrained model in _devicecuda100_6_100_100_1001_20_2_50_encoderislearnable_lambda0.5_alpha1.5_bs128_bnisFalse_lr0)
   - start model training by executing start_joint_training.sh
3. Create an ensemble
   - in `create_ensemble.sh`, adjust
       - `MODEL=` and `MODEL_PATH=` accordingly
       - the conda envrionment name in line 5 to the name of your conda environment
   - optional
       - adjust location for saving the generated ensemble `save_path=` (default is in the model directory)
       - adjust the last command (around line 56) if you want to use slurm
   - execute `create_ensemble.sh` to create the ensemble (potentially need to make it executable before `chmod +x create_ensemble.sh`)
4. Analysis with `extended_abstract_figure.ipynb`
   - potentially adjust `dae_ensemble_fact` and `dae_ensemble_cf`
   - run the notebook


This project is **work in progress**. If you encounter any issues or have suggestions, please reach out.

#### License
This project uses a small portion of code from [this framework](https://github.com/xwshen51/engression) by Xinwei Shen and Nicolai Meinshausen, which is licensed under the BSD 3-Clause License. No changes were made to the original code. The rest of the project is my own original work, licensed under the MIT License.