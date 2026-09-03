#!/bin/bash
#SBATCH --job-name=joint_dae_train
#SBATCH --partition=paula                 # adjust to your cluster's GPU partition name
#SBATCH --gpus=a30:1                    # request 1 GPU
#SBATCH --exclusive                     # exclusive node access (see note below)
#SBATCH --cpus-per-task=8               # adjust based on data loading needs
#SBATCH --mem=200G                       # adjust based on your model/data size
#SBATCH --time=0-5:00                 # walltime limit, adjust as needed
#SBATCH --output=logs/train_%j.out      # stdout log, %j = job ID
#SBATCH --error=logs/train_%j.err       # stderr log

# Make sure log directory exists
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
START_TIME=$(date +%s)

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

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Job finished at $(date)"
echo "Total training time: ${ELAPSED} seconds ($(($ELAPSED / 3600))h $((($ELAPSED % 3600) / 60))m $(($ELAPSED % 60))s)"