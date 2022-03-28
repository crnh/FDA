#!/bin/bash

# Start new detached training session
tmux new-session -d -s run_training

# Start the issued command
tmux send-keys -t run_training "$1" Enter

# Attach to the training session
tmux a -t run_training