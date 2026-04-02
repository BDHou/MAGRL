import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MaskedStorageModel(TorchModelV2, nn.Module):
    """
    接收字典输入并处理动作掩码 (Action Masking) 的自定义 PyTorch 模型
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # 初始化 RLlib 模型基类和 PyTorch 模块基类
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. 解析字典观测空间，获取真实的物理特征维度
        # 我们在环境里定义了 Dict({"obs": Box(10,), "action_mask": Box(3,)})
        true_obs_space = obs_space.original_space["obs"]
        input_dim = true_obs_space.shape[0]

        # 2. 构建 Actor 网络 (策略网络：负责输出动作)
        self.actor_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_outputs) # num_outputs 等于动作空间的维度 (如 3)
        )

        # 3. 构建 Critic 网络 (价值网络：负责评估当前状态有多好，PPO 必须)
        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Value Function 永远只输出一个标量评价
        )
        
        # 用于临时存储当前 step 的 Value，供 PPO 算法调用
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """
        前向传播逻辑：环境吐出的状态会在这里变成神经网络的输入
        """
        # 1. 分离特征与掩码
        # RLlib 会自动把环境里的 Dict 包装成 input_dict
        actual_obs = input_dict["obs"]["obs"]         # 形状: [Batch, 10]
        action_mask = input_dict["obs"]["action_mask"] # 形状: [Batch, 3]

        # 2. 物理特征经过 Actor 网络，得到原始的动作 Logits
        logits = self.actor_net(actual_obs)

        # ==========================================
        # 3. 核心操作：应用掩码 (Action Masking)
        # ==========================================
        # 将 action_mask 中值为 0 (非法) 对应的 Logit 替换为 -1e9
        # 值为 1 (合法) 对应的 Logit 保持不变
        inf_tensor = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(action_mask == 1.0, logits, inf_tensor)
        # 4. 计算 Critic 的 Value (注意这里只用到了 actual_obs，与动作无关)
        self._value_out = self.critic_net(actual_obs).squeeze(1)

        # 5. 返回被 Mask 处理过的新 Logits 
        # (RLlib 内部会对这个结果做 Softmax 抽样)
        return masked_logits, state

    def value_function(self):
        """
        返回上一次 forward 算出来的 Value，供 PPO 损失函数使用
        """
        return self._value_out