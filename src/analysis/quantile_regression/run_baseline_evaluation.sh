#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dpa # insert the name of your own conda environment here


domain_list=("GER" "FR" "SP")
climate_list=("gen" "cf_gen")

global_settings="../../../settings.json"
MODEL_PATH=$(jq -r '.paths.output_dir' "$global_settings") # path of the trained model
DAE_MODEL=$(jq -r '.current_model' "$global_settings")
epochs=$(jq -r '.epochs' "$global_settings") #60 # specify the model you want to use in terms of its training epochs
echo $DAE_MODEL

for domain in "${domain_list[@]}"; do
    for climate in "${climate_list[@]}"; do

        # set climate
        if [[ "$climate" == "gen" ]]; then
            eval_cf=0
        elif [[ "$climate" == "cf_gen" ]]; then
            eval_cf=1
        else
            echo "Choose a climate to evaluate!"
            exit
        fi

        python evaluate_pytorch_quantile_regression.py \
            --model_path "${MODEL_PATH}/QR_baseline/v5_quantile_regression_${domain}/" \
            --qr_epoch 100 \
            --results_save_path qr_baseline_eval_results/ \
            --compare_model "${MODEL_PATH}/${DAE_MODEL}/dae_ensemble_after_${epochs}_epochs/ETH_ensemble_after_${epochs}_epochs/ETH_${climate}_dpa_ens_${epochs}_dataset_restored.nc" \
            --data_version $global_settings \
            --one_dimensional_ger 0 \
            --standardize_predictors 1 \
            --eval_validation_set 0 \
            --eval_era5 0 \
            --eval_counterfactuals $eval_cf \
            --domain "$domain" \
            --eval_epochs $epochs \
            --dae_model "${DAE_MODEL}"
    done
done

