data_folder=$HOME"/data"

train_data_directory=$data_folder"/GTAV"
train_data_list=$data_folder"/GTAV/train-4000.txt"

train_target_data_directory=$data_folder"/Cityscapes/"
train_target_data_list=$data_folder"/Cityscapes/train-4000.txt"

# Change this for every different experiment!
experiment_name="train_downsample2_batch1_it4000"

num_steps=4000
print_freq=10
save_freq=100

# Run training
python3 FDA/train.py \
    --experiment-name=$experiment_name \
    --snapshot-dir=$HOME'/checkpoints/FDA/' \
    --init-weights=$HOME'/checkpoints/FDA/DeepLab_init.pth' \
    --LB=0.01 \
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
    --downsample=2 \
    --tempdata='output'