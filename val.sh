data_folder="/home/cornehaasjes/data"
checkpoint_folder="/home/cornehaasjes/checkpoints/FDA"

experiment_name="train_fullsize_batch1_it38000_beta01_gaussian_2022-04-06_14-41"

# Model weight files (omit the .pth extension at the end)
model_1_weights=$checkpoint_folder"/"$experiment_name"/gta5_38000"

val_target_data_directory=$data_folder"/Cityscapes/"
val_target_data_list=$data_folder"/Cityscapes/val-all.txt"

results_directory="/home/cornehaasjes/validation_results/"$experiment_name

# Create results directory
mkdir $results_directory

# Run training
python3 FDA/evaluation_single.py \
    --data-dir-target=$val_target_data_directory \
    --data-list-target=$val_target_data_list \
    --gt_dir=$val_target_data_directory"/gtFine/val" \
    --devkit_dir="/home/cornehaasjes/FDA/dataset/cityscapes_list" \
    --num-classes=19 \
    --restore-opt1=$model_1_weights \
    --save=$results_directory