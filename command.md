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
conda activate diffusion-TS
python main.py --name trajectory_10_20 --config_file ./Config/trajectory_10_20.yaml --gpu 0 --train
```
# trajectory 10 20 预测 10个未来点
```
cd /workspace/Diffusion-TS
conda activate diffusion-TS
python main.py --name trajectory_10_20 --config_file  ./Config/trajectory_10_20.yaml --gpu 0 --sample 1 --milestone 10 --mode predict --pred_len 10

```