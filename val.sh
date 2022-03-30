data_folder="/home/cornehaasjes/data"
checkpoint_folder="/home/cornehaasjes/checkpoints/FDA"

# Model weight files (omit the .pth extension at the end)
model_1_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"
model_2_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"
model_3_weights=$checkpoint_folder"/train_fullsize_batch1_it1000_2022-03-28_14-20/gta5_1000"

val_target_data_directory=$data_folder"/Cityscapes/"
val_target_data_list=$data_folder"/Cityscapes/train-10.txt"

results_directory="/home/cornehaasjes/validation_results"

# Run training
python3 FDA/evaluation_multi.py \
    --data-dir-target=$val_target_data_directory \
    --data-list-target=$val_target_data_list \
    --gt_dir=$val_target_data_directory"/gtFine/val" \
    --devkit_dir="/home/cornehaasjes/FDA/dataset/cityscapes_list" \
    --num-classes=19 \
    --restore-opt1=$model_1_weights \
    --restore-opt2=$model_2_weights \
    --restore-opt3=$model_3_weights \
    --save=$results_directory \
    # --downsample=2 \
    # >> '/home/cornehaasjes/checkpoints/FDA/'$experiment_name'/output.log'