```
python main.py --name etth --config_file ./Config/etth.yaml --gpu 0 --train
```
测试预测
```
python main.py --name etth --config_file  ./Config/etth.yaml --gpu 0 --sample 1 --milestone 10 --mode predict --pred_len 24
```
# trajectory 10 20 训练
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
python main.py --name trajectory_10_20 --config_file ./Config/trajectory_10_20.yaml --gpu 0 --train
```
# trajectory 10 20 训练 后台执行
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20 --config_file ./Config/trajectory_10_20.yaml --gpu 0 --train --milestone 70 > training90.log 2>&1 &

```
# trajectory 10 20 预测 20个未来点 
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20 --config_file  ./Config/trajectory_10_20.yaml --gpu 0 --sample 1 --milestone 70 --mode predict --pred_len 20  > predict90.log 2>&1 &

```

todo 
1 test 步长  删除dataset末尾的数据，防止数据不全带来问题 
2 保存 根据epoch 而不是iter 调节学习率 ？  把归一化数据加100倍？  
3 测试mse
4 查看opensky数据集格式 导入训练  有没有基于opensky数据集的论文

# Big model trajectory 10 20 训练 后台执行
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_big_model --config_file ./Config/trajectory_10_20_big_model.yaml --gpu 0 --train  > training_big_model.log 2>&1 &

```
# Big model trajectory 10 20 预测 20个未来点
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_big_model --config_file  ./Config/trajectory_10_20_big_model.yaml --gpu 0 --sample 1 --milestone 90 --mode predict --pred_len 20  > predict_big_model.log 2>&1 &

```

 # 计算指标
```
python main.py --name trajectory_10_20 --load_dir /workspace/Diffusion-TS/OUTPUT/trajectory_10_20/2025-0103-1330

```

# Super big trajectory 10 20 训练 后台执行
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_super_big --config_file ./Config/trajectory_10_20_super_big.yaml --gpu 0 --train  > training_super_big.log 2>&1 &

```
# Super big trajectory 10 20 预测 20个未来点
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_super_big --config_file  ./Config/trajectory_10_20_super_big.yaml --gpu 0 --sample 1 --milestone 37 --mode predict --pred_len 20  > predict_super_big_2_continue.log 2>&1 &

```

 # Super big 计算指标
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
python test_metrics.py --load_dir /root/autodl-tmp/trajectory_10_20_super_big/2025-0104-1114 --name trajectory_10_20_super_big 

```

# Super big trajectory 10 20 预测 20个未来点
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_super_big --config_file  ./Config/trajectory_10_20_super_big.yaml --gpu 0 --sample 1 --milestone 37 --mode predict --pred_len 20  > predict_super_big2continue.log 2>&1 &

```

 # Super big 2 计算指标
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
python test_metrics.py --load_dir /root/autodl-tmp/trajectory_10_20_super_big/2025-0104-1637 --name trajectory_10_20_super_big 

```

# Super big 2 trajectory 10 20 继续训练
```
cd /workspace/Diffusion-TS
conda activate Diffusion-TS
nohup python main.py --name trajectory_10_20_super_big --config_file  ./Config/trajectory_10_20_super_big.yaml --gpu 0 --train  --milestone 6    > training_super_big2-continue.log 2>&1 &

```