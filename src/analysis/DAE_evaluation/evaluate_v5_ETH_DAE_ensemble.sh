#!/bin/bash



source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dpa # insert the name of your own conda environment here

# === Shared configuration ===
# configuration
global_settings="../../../settings.json"
MODEL_PATH=$(jq -r '.paths.output_dir' "$global_settings") # path of the trained model
NO_EPOCHS=$(jq -r '.epochs' "$global_settings") # specify the model you want to use in terms of its training epochs
ENS_MEMBERS=$(jq -r '.no_ens_members' "$global_settings") # number of ensemble members to generate, code is not robust to any changes of this number



MODEL=$(jq -r '.current_model' "$global_settings")
echo "using model: $MODEL $NO_EPOCHS $ENS_MEMBERS"


results_save_comment_eth=""
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

##########################
# could insert creation of ensembles here





# run the analysis
period_start_years=(1850)
period_end_years=(2100)

for i in "${!period_start_years[@]}"; do
    start=${period_start_years[$i]}
    end=${period_end_years[$i]}

    echo "Running analysis for period ${start}-${end}"
    echo "Epochs: ${NO_EPOCHS}"

    # eth test set
    python analysis_results_sheet_ETH_master_slim.py \
        --period_start $start \
        --period_end $end \
        --ensemble_path $ensemble_save_path_eth \
        --no_epochs $NO_EPOCHS \
        --ens_members $ENS_MEMBERS \
        --calculate_e_loss_per_ti 0 \
        --StoNet_ensemble 0 \
        --save_path_eth "ETH_analysis_results/final_analysis_eth_test_set/model_${MODEL}/trained_for_${NO_EPOCHS}_epochs_${results_save_comment_eth}" \
        --save_path_le "ETH_analysis_results/final_analysis_train_LE/model_${MODEL}/model_trained_for_${NO_EPOCHS}_epochs_${results_save_comment_eth}" \
        --settings_file_path $global_settings \
        --no_test_members 3 \
        --include_train_analysis 0 

done

