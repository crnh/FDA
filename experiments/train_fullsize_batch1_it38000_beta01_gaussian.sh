data_folder="/home/cornehaasjes/data"

train_data_directory=$data_folder"/GTAV"
train_data_list=$data_folder"/GTAV/train-all.txt"

train_target_data_directory=$data_folder"/Cityscapes/"
train_target_data_list=$data_folder"/Cityscapes/train-all.txt"

# Change this for every different experiment!
experiment_name="train_fullsize_batch1_it38000_beta01_gaussian"

num_steps=38000
print_freq=100
save_freq=2000
fda_shape="gaussian"
beta=0.01

# Create output directory
mkdir 'output/'$experiment_name

# Run training
python3 FDA/train.py \
    --experiment-name=$experiment_name \
    --snapshot-dir='/home/cornehaasjes/checkpoints/FDA/' \
    --init-weights='/home/cornehaasjes/checkpoints/FDA/DeepLab_init.pth' \
    --LB=$beta \
    --shape=$fda_shape \
    --entW=0.005 \
    --ita=2.0 \
    --switch2entropy=0 \
    --data-dir=$train_data_directory \
    --data-list=$train_data_list \
    --data-dir-target=$train_target_data_directory \
    --data-list-target=$train_target_data_list \
    --num-steps=$num_steps \
    --print-freq=$print_freq \
    --save-pred-every=$save_freq \
    --tempdata='output/'$experiment_name