# MAGRL


dataset generation instructions： #旧版，暂时不需要，直接跑 rollout，见后面
生成dataset，命令行参数：
# --mode：
# base：正常场景（无风险事件）
# risk：开启风险事件（断线/降额/overload）
# --out_dir：输出文件夹名（不写就用配置默认值）
# --days：仿真天数（不写就用配置默认 days=30）
# --seed：随机种子（保证可复现）
# python -m scenario.run_generate --mode risk --seed 1
# python -m scenario.run_generate --mode base --seed 1 
举例：python -m scenario.run_generate --mode base --days 1 --seed 1 
目前使用的是：
python -m scenario.run_generate --mode risk --seed 1
ython -m scenario.run_generate --mode base --seed 1 


online点火测试：
python -m scenario.smoke_test_online

# 1) 生成离线图数据（每个时间步一个 PyG Data）
python -m scenario.data_rollout \
  --root data/offline_case33bw --episodes 200 --horizon 48 --policy mix

# 2) 主方法：多任务监督预训练（默认：backbone=sage, pos_weight=1, thr=tuned）
python -m train.supervised_pretrain \
  --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps

# 3) Ablation: backbone 对比（mlp/gcn/gat/sage）
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone mlp --save_dir checkpoints/abl_mlp
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone gcn --save_dir checkpoints/abl_gcn
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device cpu --backbone gat --save_dir checkpoints/abl_gat_cpu
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone sage --save_dir checkpoints/abl_sage

# 4) Ablation: edge_attr 开/关
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --use_edge_attr 1 --save_dir checkpoints/abl_edge1
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --use_edge_attr 0 --save_dir checkpoints/abl_edge0

# 5) Ablation: pos_weight 开/关（类别不均衡）
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --use_pos_weight 1 --save_dir checkpoints/abl_posw1
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --use_pos_weight 0 --save_dir checkpoints/abl_posw0

# 6) Ablation: tuned threshold vs fixed(0.5)
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --thr_mode tuned --save_dir checkpoints/abl_thrtuned
python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --thr_mode fixed --thr_fixed 0.5 --save_dir checkpoints/abl_thrfixed


RL训练：
# 5个worker + 大batch（每iter采样更多数据，质量更高）
python -m rl.train_rl \
  --ckpt checkpoints/gnn_pretrain/best.pt \
  --num_iters 200 \
  --num_workers 5 \
  --train_batch 2000 \
  2>/dev/null
