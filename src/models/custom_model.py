import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.base import ENCODER_OUT, ACTOR, CRITIC

class CustomStorageRLModule(PPOTorchRLModule):
    """
    基于最新 RLModule 架构的自定义配电网 PPO 策略网络
    """
    def setup(self):
        # 1. 调用父类 setup，它会根据 config 自动构建默认的基础组件
        super().setup()
        
        # 2. 获取环境的空间维度信息
        obs_dim = self.config.observation_space.shape[0]
        
        # 3. 在这里，你可以完全接管并重写特征提取器 (Encoder)
        # 目前先用多层感知机 (MLP) 作为示例。
        # 后续如果你需要针对电网拓扑加入图神经网络 (GNN)，只需替换此处的 self.encoder 即可。
        hidden_dims = self.config.model_config_dict.get("fcnet_hiddens", [256, 256])
        
        class CustomEncoder(nn.Module):
            def __init__(self, in_dim, h_dims):
                super().__init__()
                layers = []
                d = in_dim
                for hd in h_dims:
                    layers.append(nn.Linear(d, hd))
                    layers.append(nn.ReLU())
                    d = hd
                self.net = nn.Sequential(*layers)
            
            def forward(self, batch):
                obs = batch.get("obs", batch.get("OBS"))
                features = self.net(obs)
                # Ray RLModule requires a nested dictionary output:
                return {ENCODER_OUT: {ACTOR: features, CRITIC: features}}
                
        self.encoder = CustomEncoder(obs_dim, hidden_dims)
        
        # 打印信息用于验证自定义模型已被正确加载
        print(f"[CustomStorageRLModule] 初始化完成，特征提取器输入维度: {obs_dim}")