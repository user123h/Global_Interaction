#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --model 'dnn' \
    --winit_initialisation 'awpn' \
    --winit_first_layer_interpolation_scheduler 'linear' \
    --winit_first_layer_interpolation_end_iteration 200 \
    --winit_first_layer_interpolation 1 \
    --logger 'csv' \
    --experiment_name 'allaml_1' \
    --lr 0.0001 \
    --dropout_rate 0.15 \
    --dataset 'allaml'
    --enable_ofi_overfitting_detection \
    --ofi_patience 5 \
    --ofi_threshold 0.1 \
    --ofi_min_train_acc 0.95\
    --ofi_verbose 1 \
    --ofi_min_epochs 50 \
	--ofi_max_acc_threshold 0.15 \
	--min_epochs_to_save 50 
	--metric_model_selection 'balanced_accuracy'  
	--run_repeats_and_cv     
