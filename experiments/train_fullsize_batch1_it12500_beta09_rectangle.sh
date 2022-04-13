data_folder=$HOME"/data"

train_data_directory=$data_folder"/GTAV"
train_data_list=$data_folder"/GTAV/train-12500.txt"

train_target_data_directory=$data_folder"/Cityscapes/"
train_target_data_list=$data_folder"/Cityscapes/train-12500.txt"

# Change this for every different experiment!
experiment_name="train_fullsize_batch1_it12500_beta09_rectangle"

num_steps=12500
print_freq=100
save_freq=500
fda_shape="rectangle"
beta=0.09
seed=3407

# Create output directory
mkdir 'output/'$experiment_name

# Run training
python3 FDA/train.py \
    --experiment-name=$experiment_name \
    --snapshot-dir=$HOME'/checkpoints/FDA/' \
    --init-weights=$HOME'/checkpoints/FDA/DeepLab_init.pth' \
    --LB=$beta \
    --shape=$fda_shape \
    --entW=0.005 \
    --ita=2.0 \
    --switch2entropy=0 \
    --seed=$seed \
    --data-dir=$train_data_directory \
    --data-list=$train_data_list \
    --data-dir-target=$train_target_data_directory \
    --data-list-target=$train_target_data_list \
    --num-steps=$num_steps \
    --print-freq=$print_freq \
    --save-pred-every=$save_freq \
    --tempdata='output/'$experiment_name