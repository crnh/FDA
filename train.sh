data_folder="/home/cornehaasjes/data"

train_data_directory=$data_folder"/GTAV"
train_data_list=$data_folder"/GTAV/train-10.txt"

train_target_data_directory=$data_folder"/Cityscapes/"
train_target_data_list=$data_folder"/Cityscapes/train-10.txt"

num_steps=10
print_freq=1

# Run training
python3 FDA/train.py \
    --snapshot-dir='/home/cornehaasjes/checkpoints/FDA' \
    --init-weights='/home/cornehaasjes/checkpoints/FDA/DeepLab_init.pth' \
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
    --tempdata='output'