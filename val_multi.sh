data_folder=$HOME"/data"
checkpoint_folder=$HOME"/checkpoints/FDA"

# Model weight files (omit the .pth extension at the end)
# model_1_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"
# model_2_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"
# model_3_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"

model_1_weights=$checkpoint_folder"/pretrained/gta2city_LB_0_01"
model_2_weights=$checkpoint_folder"/pretrained/gta2city_LB_0_05"
model_3_weights=$checkpoint_folder"/pretrained/gta2city_LB_0_09"

val_target_data_directory=$data_folder"/Cityscapes/"
val_target_data_list=$data_folder"/Cityscapes/val-all.txt"

results_directory=$HOME"/validation_results"

# Run training
python3 FDA/evaluation_multi.py \
    --data-dir-target=$val_target_data_directory \
    --data-list-target=$val_target_data_list \
    --gt_dir=$val_target_data_directory"/gtFine/val" \
    --devkit_dir=$HOME"/FDA/dataset/cityscapes_list" \
    --num-classes=19 \
    --restore-opt1=$model_1_weights \
    --restore-opt2=$model_2_weights \
    --restore-opt3=$model_3_weights \
    --save=$results_directory \
    # --downsample=2 \
    # >> $HOME'/checkpoints/FDA/'$experiment_name'/output.log'