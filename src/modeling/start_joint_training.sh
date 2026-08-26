#!/bin/bash
# use slurm if required

# start training
# add arguments for model settings
~/.conda/envs/dpa/bin/python train_joint_dae.py \
    --settings_file "/home/sc.uni-leipzig.de/fl53wumy/llaae_new/TowardsDistributionalAutoencoderClimateCounterfactuals/settings.json" \
    --encoder "learnable" \
    --in_dim 648 \
    --latent_dim 50 \
    --num_layer 6 \
    --hidden_dim 100 \
    --noise_dim_dec 5 \
    --resblock 1 \
    --in_dim_lm 1001 \
    --noise_dim_lm 20 \
    --num_layer_lm 2 \
    --hidden_dim_lm 50 \
    --lr 0.0001 \
    --batch_size 128 \
    --epochs 100 \
    --batch_norm 0 \
    --lam 1.0 \
    --alpha 1.5 \
    --include_pen_e 0 \