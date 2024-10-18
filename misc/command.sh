#!/bin/bash

python3 /home/mahd/Label-Augmentation-Folder/main.py \
    --result_directory "./results" \
    --wandb_name "training_with_pipeline" \
    --wandb_project "my_new_project" \
    --seed 2022 \
    --input_channel 3 \
    --output_channel 11 \
    --encoder_depth 5 \
    --decoder_channel 256 128 64 32 16 \
    --lr 1e-4 \
    --epochs 400 \
    --batch_size 5 \
    --image_resize 512 \
    --dilate 72 \
    --dilation_decrease 10 \
    --dilation_epoch 50  \
    
