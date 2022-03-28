data_folder="/home/cornehaasjes/data"

train_data_directory=$data_folder"/GTAV"
train_data_list=$data_folder"/GTAV/train-4000.txt"

train_target_data_directory=$data_folder"/Cityscapes/"
train_target_data_list=$data_folder"/Cityscapes/train-4000.txt"

# Change this for every different experiment!
experiment_name="train_downsample2_batch4_it1000"

num_steps=1000
print_freq=10
save_freq=100

batch_size=4
lr=5e-4

# Run training
python3 FDA/train.py \
    --experiment-name=$experiment_name \
    --snapshot-dir='/home/cornehaasjes/checkpoints/FDA/' \
    --init-weights='/home/cornehaasjes/checkpoints/FDA/DeepLab_init.pth' \
    --LB=0.01 \
    --entW=0.005 \
    --ita=2.0 \
    --batch-size=$batch_size \
    --learning-rate=$lr \
    --switch2entropy=0 \
    --data-dir=$train_data_directory \
    --data-list=$train_data_list \
    --data-dir-target=$train_target_data_directory \
    --data-list-target=$train_target_data_list \
    --num-steps=$num_steps \
    --print-freq=$print_freq \
    --save-pred-every=$save_freq \
    --downsample=2