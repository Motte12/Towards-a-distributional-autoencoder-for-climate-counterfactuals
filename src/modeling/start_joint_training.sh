#!/bin/bash

# configuration
global_settings="../../settings.json"
MODEL_PATH=$(jq -r '.paths.output_dir' "$global_settings") # path of the trained model
NO_EPOCHS=$(jq -r '.epochs' "$global_settings") # specify the model you want to use in terms of its training epochs (in settings.json)
ENS_MEMBERS=$(jq -r '.no_ens_members' "$global_settings") # number of ensemble members to generate, code is not robust to any changes of this number


MODEL=$(jq -r '.current_model' "$global_settings")
echo "using model: $MODEL $NO_EPOCHS $ENS_MEMBERS"

# specify path to save generated ensemble
save_path="${MODEL_PATH}/${MODEL}" 
echo "savepath: $save_path"
ensemble_save_path_eth="${save_path}/dae_ensemble_after_${NO_EPOCHS}_epochs/"
echo "ensemble save path: $ensemble_save_path_eth"


# specify model
ENCODER="model_enc_${NO_EPOCHS}.pt"
DECODER="model_dec_${NO_EPOCHS}.pt"
LATENT_MAP="model_pred_${NO_EPOCHS}.pt"


### load model configs ###
cfg="${MODEL_PATH}/${MODEL}/model_and_train_settings.json"

alpha=$(jq -r '.alpha' "$cfg")
batch_norm=$(jq -r '.batch_norm' "$cfg")
batch_size=$(jq -r '.batch_size' "$cfg")
encoder=$(jq -r '.encoder' "$cfg")
epochs=$(jq -r '.epochs' "$cfg")
hidden_dim=$(jq -r '.hidden_dim' "$cfg")
hidden_dim_lm=$(jq -r '.hidden_dim_lm' "$cfg")
in_dim=$(jq -r '.in_dim' "$cfg")
in_dim_lm=$(jq -r '.in_dim_lm' "$cfg")
lam=$(jq -r '.lam' "$cfg")
latent_dim=$(jq -r '.latent_dim' "$cfg")
lr=$(jq -r '.lr' "$cfg")
noise_dim_dec=$(jq -r '.noise_dim_dec' "$cfg")
noise_dim_lm=$(jq -r '.noise_dim_lm' "$cfg")
num_layer=$(jq -r '.num_layer' "$cfg")
num_layer_lm=$(jq -r '.num_layer_lm' "$cfg")
out_activation=$(jq -r '.out_activation // empty' "$cfg")
resblock=$(jq -r '.resblock' "$cfg")
settings_file=$(jq -r '.settings_file' "$cfg")


# start training
# add arguments for model settings
~/.conda/envs/dpa/bin/python train_joint_dae.py \
    --settings_file "$global_settings" \
    --encoder $encoder \
    --in_dim $in_dim \
    --latent_dim $latent_dim \
    --num_layer $num_layer \
    --hidden_dim $hidden_dim \
    --noise_dim_dec $noise_dim_dec \
    --resblock $resblock \
    --in_dim_lm $in_dim_lm \
    --noise_dim_lm $noise_dim_lm \
    --num_layer_lm $num_layer_lm \
    --hidden_dim_lm $hidden_dim_lm \
    --lr $lr \
    --batch_size $batch_size \
    --epochs $epochs \
    --batch_norm $batch_norm \
    --lam $lam \
    --alpha $alpha \
    --include_pen_e 0 \