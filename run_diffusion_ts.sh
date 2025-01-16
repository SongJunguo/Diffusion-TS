#!/bin/bash

# 切换到工作目录
cd /workspace/Diffusion-TS

#conda init

# 激活conda环境
#conda activate Diffusion-TS

# 训练命令
echo "Starting training process..."
nohup python main.py --name trajectory_10_20 --config_file ./Config/trajectory_10_20.yaml --gpu 0 --train > training.log 2>&1 &

# 等待训练完成
echo "Training has started. Waiting for completion..."
wait $!

# 预测命令
echo "Training completed. Starting prediction process..."
nohup python main.py --name trajectory_10_20 --config_file ./Config/trajectory_10_20.yaml --gpu 0 --sample 1 --milestone 50 --mode predict --pred_len 20 > predict.log 2>&1 &

echo "Prediction has started in the background."
