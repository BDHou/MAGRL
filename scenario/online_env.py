# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np

from .online_backend import OnlineBackend, StepResult


# -----------------------------
# Env Config
# Config 就是 reward 和 episode 规则的开关，不影响电网物理计算。
# -----------------------------
@dataclass
class OnlineEnvConfig:
    """
    多智能体 Online Env 的配置（可按你项目慢慢调）
    """
    # episode 长度（如果 None，就用 backend.T_total - t0）
    #episode_horizon=48：每条 episode 走 48 步就停（比如 48 个时间点）。
    episode_horizon: Optional[int] = 48

    # PF 失败是否直接结束 episode（更“硬”，训练更稳定）
    # False：不立刻结束，继续跑（但会给惩罚），更“柔和”
	# True：失败就直接终止，训练更稳定（常用）
    terminate_on_pf_fail: bool = False

    # PF 失败惩罚（越大越不允许失败）
    # 潮流失败时给一个很大的负奖励 （-50.0）（告诉 agent：别把系统搞崩）
    pf_fail_penalty: float = -50.0

    # reward 权重（默认：电压违规 > 反向潮流 > 外送）
    # w_v_viol / w_rpf / w_export：奖励里三项惩罚的权重
	# 电压越界 num_v_viol：最重要（权重默认最大 2.0）
	# 反向潮流线路数 num_rpf_lines
	# 外送 export
    w_v_viol: float = 2.0
    w_rpf: float = 0.5
    w_export: float = 0.2

    # 可选：惩罚动作幅度（避免 agent 乱打无意义大动作）
    # 惩罚动作太大（防止 agent 一上来就乱打 0.9、-0.9 这种极端动作）
    w_action_l2: float = 0.05

    # 观测是否每个 agent 都复制一份（True 更安全；False 更省内存但要小心外部 inplace 改）
    # 每个 agent 拿到的 obs（图）会复制一份，避免你后面代码不小心 inplace 修改导致互相影响
    copy_obs_per_agent: bool = True

    # done 字段风格：是否按 Gymnasium 新版区分 terminated/truncated
    # 是否输出 gymnasium 新版那种 terminated/truncated（RLlib 有时候喜欢这个）
    use_terminated_truncated: bool = False


# -----------------------------
# Multi-Agent Online Env
# -----------------------------
class MultiAgentOnlineEnv:
    """
    把 OnlineBackend 包成“多智能体标准接口”。

    核心思路：
      - backend 只管物理仿真：step(action_vec)->obs_graph/targets/metrics
      backend 的输入输出规则固定：
	•	输入：一个长向量 action_vec
	•	输出：
	•	obs_graph：图观测（给 GNN 的）
	•	targets：监督学习目标（比如线潮流、bus 电压真值）
	•	metrics：统计指标（越界数量、反潮流线路数等）
      - env 只管多智能体封装：
          dict(action) -> 拼 vector -> backend.step -> dict(obs/reward/done/info)
          env 的工作就是“翻译”：
	•	输入：字典动作
	•	转换：拼成向量
	•	调用：backend.step
	•	输出：字典形式 obs/reward/done/info（每个 agent 都有一份）

    默认每个 agent 对应一个 PV（也就是一个 controllable sgen）
      agent_id: "pv_0", "pv_1", ...
      意思：你有几个 PV，就有几个 agent。
        例如 n_pv=3 → agent_ids 默认就是：
        "pv_0", "pv_1","pv_2",

    action 形式：
      - 若 backend.enable_curtail_action == False：
          每个 agent action = float 或 shape(1,) -> q_action
          如果你没启用 curtailment，那么每个 agent 只控制一个数：无功 Q。
            action=0.7 或 [0.7] 都行
            解释为：给该 PV 一个 q 设置（是 q_frac 还是 q_mvar 看 backend.mode）
      - 若 True：
          每个 agent action = (q, curt) 或 shape(2,)
          q 是 q_frac/q_mvar（由 backend.mode 决定），curt ∈ [0,1]
          如果你启用了 curtailment，那么每个 agent 控制两个数：
            第一个：q（无功）
            第二个：curt（削减比例，0~1，比如 0.2=削减20%）
    """

    # 未来训练时，会给 env 一个字典：{"pv_0": 0.2, "pv_1": -0.1, "pv_2": 0.0}
    # env 内部会把它拼成 backend 要的向量：[0.2, -0.1, 0.0]
    # 然后调用 backend.step(action_vec) 得到潮流结果，再把输出变成多智能体格式。
    
    #  __init__：构建这个环境需要什么
    def __init__(
        self,
        backend: OnlineBackend, #backend：物理仿真引擎（必需）
        env_cfg: Optional[OnlineEnvConfig] = None,
        #env_cfg：环境配置（reward、episode长度等），不传就用默认
        *,
        agent_ids: Optional[List[str]] = None,
        # agent_ids：你可以自己定义 agent 的名字，比如 ["A","B","C"]
        # 不传就默认 "pv_0","pv_1"...
        # *,：这是 Python 语法，表示 agent_ids 必须用关键字传参，比如：
        # MultiAgentOnlineEnv(backend, agent_ids=[...])
    ):
    # 初始化：把传进来的东西存起来
        self.backend = backend
        self.cfg = env_cfg or OnlineEnvConfig()

        # --- agent ids / mapping ---
        # 最关键：确定“有几个 agent”以及它们的名字
        self.n_agents = int(self.backend.n_pv)
        # backend 在建网的时候确定了 PV 数量：backend.n_pv
	    # env 这里直接说：agent 数 = PV 数
        # 比如 backend 有 3 个 PV → self.n_agents = 3

        # 如果你没传 agent_ids：默认生成 pv_0…pv_{n-1}
        # 例子：n_agents=3 → ["pv_0","pv_1","pv_2"]
        if agent_ids is None:
            self.agent_ids = [f"pv_{i}" for i in range(self.n_agents)]
        # 如果你传了 agent_ids：要检查长度对不对
        # 为什么要检查长度？因为必须一一对应：
        # 有 3 个 PV → 必须有 3 个 agent 名字,否则 env 不知道哪个 agent 控制哪个 PV
        else:
            if len(agent_ids) != self.n_agents:
                raise ValueError(f"agent_ids length mismatch: got {len(agent_ids)}, expect {self.n_agents}")
            self.agent_ids = list(agent_ids)

        # --- mapping: agent_id -> index ---
        # 重要映射：agent_id → PV索引
        self.agent_to_i = {aid: i for i, aid in enumerate(self.agent_ids)}
        # 这是一个字典，用来查：这是一个字典，用来查：
	    # "pv_0" 对应第 0 个 PV
        # "pv_1" 对应第 1 个 PV
	    # …
        # 举例：
        # self.agent_ids = ["pv_0","pv_1","pv_2"]
        # self.agent_to_i = {"pv_0":0, "pv_1":1, "pv_2":2}
        # 它的作用：后面拼 action_vec 时必须按固定顺序放进去。
        # 否则很容易出现 bug：
        # pv_0 的动作写到 pv_2 上了 → 训练完全乱套


        # --- episode control ---
        # episode 控制变量（env自己管理的）
        # 这三个变量就是 env 自己用来管理 episode 的：
        # t0：reset 的起点时间（比如从第 0 步开始）
        # step_count：这个 episode 走了几步了
        # horizon：这个 episode 最多允许走多少步（比如 48 步）
        # 你可以理解为：env 用它们决定什么时候 done。
        self.t0: int = 0
        self.step_count: int = 0
        self.horizon: int = 0

        # --- last good obs (用于 pf_fail 时返回，不至于 None 崩掉训练) ---
        # last good obs：为了 PF fail 时不崩
        # 这是一个非常实用的“稳定训练”技巧：
        # 	•	如果某一步潮流失败（pf_fail），backend 会返回 obs_graph=None
        # 	•	但 RL 训练一般不喜欢 None，容易崩（尤其是 GNN 网络输入必须是 tensor）
        # 	•	所以 env 保存“上一次成功的 obs_graph”
        # 	•	以后 pf_fail 时，就返回这个 last_good_obs，训练不会中断
        # _last_info_meta 是为了留 debug 信息（上一次成功时的 meta）。
        self._last_obs_graph: Optional[Dict[str, np.ndarray]] = None
        self._last_info_meta: Optional[Dict[str, Any]] = None

    # -----------------------------
    # Utilities
    # 当成：env 的“工具箱”，专门做 5 件事：
	# 1.	每个 agent 的动作到底是 1 维还是 2 维
	# 2.	不管你给什么形式的动作，都统一成标准形状
	# 3.	把多智能体 dict 动作拼成 backend 要的一个大向量
	# 4.	把 backend 的一个 obs_graph “广播”给每个 agent
	# 5.	算 reward、done 的格式
    # -----------------------------


    # _agent_action_dim()：它只回答一个问题：每个 agent 的 action 有几个数？
	# •	如果 backend 没开 curtail（enable_curtail_action=False）
    # → 每个 agent 只控制 Q（1 个数）
    # → 返回 1
	# •	如果 backend 开了 curtail（enable_curtail_action=True）
    # → 每个 agent 控制 (Q, Curtail)（2 个数）
    # → 返回 2
    # 这一步非常关键，因为后面所有处理动作的函数都要知道“应该是 1 维还是 2 维”。
    def _agent_action_dim(self) -> int:
        return 2 if self.backend.enable_curtail_action else 1


    # _normalize_agent_action(...)它要解决的问题是什么？
    # 用户（或者训练算法）可能给 action 的格式很乱，比如：
	# 0.7（float），1（int）[0.7]，np.array([0.7])
	# •	(0.7, 0.2)  （如果开了curtail）
    # 但 backend 最终希望每个 agent 的动作都能被统一处理。所以这个函数要做的是：
    # 无论你给什么形式，都把它变成一个 numpy 数组。并且形状固定为：
	# •	ad=1 → array([q])
	# •	ad=2 → array([q, curt])
    def _normalize_agent_action(self, a: Union[float, int, np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """
        把 agent 的 action 统一成 shape(agent_action_dim,) 的 np.ndarray
        """
        ad = self._agent_action_dim() #先问：每个 agent 动作维度是 1 还是 2。
        # 这里是“兼容各种输入”：
        if isinstance(a, (float, int)): #如果给的是一个数字 0.7
            arr = np.array([float(a)], dtype=float) #→ 变成 [0.7]
        else:  #否则（比如 list/tuple/np.array）
            arr = np.asarray(a, dtype=float).reshape(-1)
            #→ np.asarray(...) 转成 numpy
            # → .reshape(-1) 强制变成一维（比如 [[0.7]] 也拉平）


        # 这行很重要：容错。
        if arr.size == 1 and ad == 2:  #如果 backend 需要 2 维动作 (q, curt)，但你只给了一个数：
            # 用户只给了 q，curt 默认 0
            arr = np.array([float(arr[0]), 0.0], dtype=float)
            # 那就自动补一个 curt=0.0

        # 如果最后维度还是不对，就报错。
        if arr.size != ad:
            raise ValueError(f"agent action dim mismatch: got {arr.size}, expect {ad}")
        return arr.astype(float)
        #最终保证是 float 类型的 numpy 数组。

    #  _assemble_joint_action(actions: Dict[str, Any])它要解决的问题是什么？
    # 你的多智能体输入是：
    #     {
    # "pv_0": 0.2,
    # "pv_1": -0.1,
    # "pv_2": 0.0
    # }
    # 但 backend 只接受一个向量：无 curtail：[0.2, -0.1, 0.0]。有 curtail：[q0,q1,q2, curt0,curt1,curt2]
    # 所以这个函数负责 拼接
    def _assemble_joint_action(self, actions: Dict[str, Any]) -> np.ndarray:
        """
        dict(agent_id -> action) -> backend 需要的全局向量
        backend action_vec:
          - no curtail: [q0, q1, ...]
          - with curtail: [q0..qN-1, curt0..curtN-1]
        """
        # 缺的 agent 默认 0（更鲁棒）
        ad = self._agent_action_dim() #每个 agent 动作维度 1 还是 2。


        # 先创建空的数组：
        # q：长度 = agent 数量（PV数量）
        # 如果开了 curtail，再创建 curt 数组，否则 curt=None
        q = np.zeros(self.n_agents, dtype=float)
        curt = np.zeros(self.n_agents, dtype=float) if self.backend.enable_curtail_action else None

        for aid in self.agent_ids: # 对每个 agent_id（比如 pv_0/pv_1/pv_2）：
            i = self.agent_to_i[aid] #查它应该写到哪个位置 i（0/1/2）
            # 这里做了一个很实用的“鲁棒性设计”：
            if aid not in actions: #如果某个 agent 没给 action（字典里缺了 pv_1）
                aa = np.zeros(ad, dtype=float) #→ 默认它的动作全是 0
            else:
                aa = self._normalize_agent_action(actions[aid])

            # 把 normalized 的动作写入：
            # aa[0] 永远是 q
            # aa[1]（如果存在）是 curt
            q[i] = float(aa[0])
            if curt is not None:
                curt[i] = float(aa[1])

        # 最终返回 backend action_vec：
        # 无 curtail：直接返回 q，有 curtail：把 [q, curt] 拼起来
        if curt is None:
            return q
        return np.concatenate([q, curt], axis=0)

    # _broadcast_obs(obs_graph)它干什么？
    # backend 每一步输出一个图观测 obs_graph（全局电网图）。
    # 但多智能体 env 需要返回：
    #     {
    # "pv_0": obs_graph,
    # "pv_1": obs_graph,
    # "pv_2": obs_graph,
    # }
    # 也就是 每个 agent 都拿到一份观测（默认一样）。
    def _broadcast_obs(self, obs_graph: Optional[Dict[str, np.ndarray]]) -> Dict[str, Optional[Dict[str, np.ndarray]]]:
        """
        默认所有 agent 拿同一个全局图 obs（GNN 可共享）
        """
        # 准备一个输出 dict，遍历每个 agent。
        out: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        for aid in self.agent_ids:
            #如果潮流失败、或者没有观测，则每个 agent 都是 None（但 step 里其实会用 last_good_obs，避免这里真的 None）。
            if obs_graph is None:
                out[aid] = None
            else:
                out[aid] = {k: v.copy() for k, v in obs_graph.items()} if self.cfg.copy_obs_per_agent else obs_graph
            '''
            else:这行是一个“安全/性能”开关：
            •	copy_obs_per_agent=True（安全）
            •	每个 agent 得到一份自己的拷贝
            •	防止外部代码对某个 agent 的 obs 做 inplace 修改，影响别的 agent
            •	copy_obs_per_agent=False（省内存）
            •	所有 agent 指向同一个 obs_graph 对象
            •	更省，但如果你外部不小心改了，会连带影响所有 agent
            '''   
        return out
    
    # _compute_global_reward(sr, joint_action)它干什么？
    # 它给所有 agent 一个 共享的全局 reward（最简单、最稳定）。
    def _compute_global_reward(self, sr: StepResult, joint_action: np.ndarray) -> float:
        """
        先用一个“全局共享 reward”（最稳，不容易乱）：
          reward = - w_v* num_v_viol - w_rpf * num_rpf_lines - w_export * export - w_a * ||action||^2
          	•	电压越界越多 → 更差
            •	反向潮流线路越多 → 更差
            •	外送(export) → 更差（按你的设定）
            •	动作幅度太大 → 更差（防止乱打大动作）
        """
        
        # 如果潮流失败：
        # 	reward 直接给一个大负数（比如 -50）
	    #     让训练算法学会“别把系统搞崩”

        if not sr.ok or sr.metrics is None:
            return float(self.cfg.pf_fail_penalty)

        # 从 metrics 里取关键指标：
        # num_v_viol：电压越界的 bus 数
        # num_rpf_lines：反向潮流线路数
        # export：是否外送（0/1）
        m = sr.metrics
        num_v_viol = float(m["num_v_viol"])
        num_rpf = float(m["num_rpf_lines"])
        export = float(m["export"])

        # 动作惩罚项：把 joint_action 每个分量平方，再取平均。
        #动作越大 → 惩罚越大,让 agent 学会“能小就小”
        act_pen = float(np.mean(joint_action.astype(float) ** 2))  # 平均 L2
        # 按权重组合成 reward。
        # 注意：这里 reward 是负的（惩罚型），训练目标是让惩罚变小（reward 变大/不那么负）。
        r = (
            - self.cfg.w_v_viol * num_v_viol
            - self.cfg.w_rpf * num_rpf
            - self.cfg.w_export * export
            - self.cfg.w_action_l2 * act_pen
        )
        return float(r)


    # _make_done_flags(terminated, truncated)这个函数只负责输出 done 的格式。
    # 旧风格（RLlib 常见）：{"done": {"pv_0": True, "pv_1": True, "__all__": True}}
    # Gymnasium 新风格：分成两类：terminated：自然终止（比如失败）truncated：时间到了被截断（horizon到了）
    def _make_done_flags(self, terminated: bool, truncated: bool) -> Dict[str, Any]:
        """
        输出 done 格式：
          - 旧风格：done_dict[agent]=done, done_dict["__all__"]=done
          - Gymnasium 风格：terminated_dict / truncated_dict
        """

        # 如果你选用新风格：
        if self.cfg.use_terminated_truncated:
            term = {aid: terminated for aid in self.agent_ids}
            trun = {aid: truncated for aid in self.agent_ids}
            term["__all__"] = terminated
            trun["__all__"] = truncated
            return {"terminated": term, "truncated": trun}
            #每个 agent 都给同样的 terminated/truncated，另外还必须提供 __all__。
        # 否则走旧风格：
        else:
            done = bool(terminated or truncated)
            dd = {aid: done for aid in self.agent_ids}
            dd["__all__"] = done
            return {"done": dd}
            # 只要 terminated 或 truncated 有一个为 True，就 done=True。

    # -----------------------------
    # Public API
    # Public API（reset + step）
    # reset()：开始一局/一个 episode，拿到初始观测
	# step()：每一步接收多智能体动作 → 调 backend 跑潮流 → 返回多智能体需要的四个东西（obs, reward, done, info）
    # -----------------------------

    # reset 的输入输出是什么？
    # 输入：t0（从第几个时间步开始，比如 0 表示从一天开始）
    # 输出：两个 dict：obs_dict: {agent_id: obs_graph}，info_dict: {agent_id: info}
    # 在多智能体框架里，reset 一般要返回每个 agent 的初始观测。
    def reset(self, *, t0: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        reset -> (obs_dict, info_dict)

        obs_dict: {agent_id: obs_graph}
        info_dict: {agent_id: info}
        """
        # 保存起点和清零计数器
        self.t0 = int(t0) #这局从哪个时间步开始
        self.step_count = 0 #已经走了多少步（用于判断 horizon）

        if self.cfg.episode_horizon is None: #如果没指定 episode 长度（episode_horizon=None）
            self.horizon = int(self.backend.T_total - self.t0) 
            #就从 t0 一直跑到 backend 预生成的时间序列末尾
        else: #如果指定了（比如 48）
            self.horizon = int(self.cfg.episode_horizon)
            #那就每局固定 48 步（常用于训练）

        # 调 backend.reset 得到初始 StepResult
        sr0 = self.backend.reset(t0=self.t0)
        # sr0 是一个 StepResult，里面有：
        #     •	sr0.ok：潮流是否成功
        #     •	sr0.obs_graph：图观测（x, edge_index, edge_attr）
        #     •	sr0.metrics：电压越界数、反向潮流线路数等
        #     •	sr0.meta：t/day/hour/seed 等
        #     •	sr0.info：runpp_robust 的信息
        #注意： backend 的 reset 其实内部做了一个 “action=0 的 step”，
        #所以 reset 直接就有 obs/metrics。

        # 记录 last good obs，存“最后一次成功的观测 last_good_obs”
        # 为什么要做这个？
        # 训练时可能出现 PF fail
        # •	如果 PF fail 返回 obs=None，很多 RL 库会崩
        # •	所以 env 保存一个“上一次成功的 obs”
        # •	PF fail 时就用这个“最后成功观测”顶一下（保持训练稳定）
        if sr0.ok and sr0.obs_graph is not None:
            self._last_obs_graph = sr0.obs_graph
            self._last_info_meta = sr0.meta
        else:
            self._last_obs_graph = None
            self._last_info_meta = sr0.meta

        # 构造 obs_dict：给每个 agent 发一份 obs_graph
        	# •	如果 reset 的潮流成功：用 sr0.obs_graph
	        # •	如果失败：用 _last_obs_graph（可能是 None，但通常 reset 不太会 fail）  
        #_broadcast_obs 的结果是：
        #         {
        # "pv_0": obs_graph,
        # "pv_1": obs_graph,
        # "pv_2": obs_graph,
        # }
        # 这就是未来 GNN 的输入：每个 agent 都拿到同一张电网图（最常见做法）。
        obs_dict = self._broadcast_obs(sr0.obs_graph if sr0.ok else self._last_obs_graph)

        #构造 info_dict：每个 agent 拿到同一份 info
        # info：每个 agent 都拿到同一份 meta/metrics（你后面可拆成局部）
        # info_common：这一步的“附加信息包”
        # 每个 agent 都收到一份一样的（复制 dict）  
        # 这里 targets=None 的原因：
        # •	reset 通常只需要 obs
        # •	你后面如果想 reset 也返回 targets（用于监督学习预热），可以改成 sr0.targets
        info_common = {
            "ok": bool(sr0.ok),
            "meta": sr0.meta,
            "info": sr0.info,
            "metrics": sr0.metrics,
            "targets": None,  # reset 不强制塞 targets（你想要可改成 sr0.targets）
        }
        info_dict = {aid: dict(info_common) for aid in self.agent_ids}
        return obs_dict, info_dict


    # step(self, actions)step 的输入输出是什么？
    # 输入：多智能体动作
    #     actions = {
    # "pv_0": 0.2,
    # "pv_1": -0.1,
    # "pv_2": 0.0
    # }
    # 输出（旧风格）四件套：
	# 1.	obs_dict：每个 agent 的新观测（图）
	# 2.	rew_dict：每个 agent 的 reward（目前共享同一个 reward）
	# 3.	done_dict：是否结束（含 “all”）
	# 4.	info_dict：各种附加信息（metrics/targets/meta 等）
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        """
        多智能体 step：

        输入：
          actions: {agent_id -> action}

        输出（旧风格）：
          obs_dict, rew_dict, done_dict, info_dict

        输出（Gymnasium 风格）：
          obs_dict, rew_dict, {"terminated":..., "truncated":...}, info_dict
        """
        self.step_count += 1 # 步数 +1，表示这一局走到了第几步。

        # 多智能体动作拼成 backend 的 joint_action 向量
        #     例如：
        # •	无 curtail：[q0, q1, q2]
        # •	有 curtail：[q0,q1,q2, curt0,curt1,curt2]
        joint_action = self._assemble_joint_action(actions)
        sr = self.backend.step(joint_action) #调 backend.step 跑一遍潮流
        #     backend 做的事是：
        # •	写入负荷/PV P（根据真实 profile）
        # •	写入 Q（根据 action）
        # •	runpp_robust
        # •	构造 obs_graph / targets / metrics

        # horizon 判断，判断是否到达 episode horizon（是否该截断）
        reached_horizon = (self.step_count >= self.horizon)
        # 如果到了 horizon，就该结束这一局
	    # 这种结束一般属于 truncated（时间到了），不是系统崩溃

        # 分支 A：PF fail（sr.ok == False），PF fail 处理
        if not sr.ok:
            # 1) reward：大惩罚 。(A1) reward：给大惩罚
            # 比如 -50，告诉 agent：你做了一个导致潮流失败的动作，非常差。
            global_r = float(self.cfg.pf_fail_penalty)

            # 2) obs：返回 last good obs（训练更稳）。(A2) obs：返回 last good obs（避免 None）
            # 这是训练稳定性的关键：
            # 否则 obs=None 很多算法直接报错。
            obs_graph = self._last_obs_graph
            obs_dict = self._broadcast_obs(obs_graph)

            # 3) done：可选直接终止。(A3) done：是否立刻结束这局？
            # •	terminate_on_pf_fail=True：一旦 PF fail，就直接结束 episode（terminated）
	        # •	terminate_on_pf_fail=False：不一定结束，除非到了 horizon（truncated）
            # 这就是你之前问的 terminated vs truncated 的含义：
            # •	terminated：系统真正“坏了/结束了”
            # •	truncated：时间到了，被截断结束
            terminated = bool(self.cfg.terminate_on_pf_fail)
            truncated = bool(reached_horizon and not terminated)


            done_pack = self._make_done_flags(terminated=terminated, truncated=truncated)

            # 4) info。(A4) info：把失败原因、joint_action 写进去
            # 这里能看到：
            #     •	为什么失败（sr.info 里可能有 error）
            #     •	当时动作是什么（joint_action）

            # 对 debug 非常有用。
            info_common = {
                "ok": False,
                "meta": sr.meta,
                "info": sr.info,
                "metrics": None,
                "targets": None,
                "joint_action": joint_action.copy(),
            }
            info_dict = {aid: dict(info_common) for aid in self.agent_ids}


            # (A5) rew_dict：每个 agent 同样的惩罚
            # 每个 agent 目前共享 reward（后面可做 credit assignment）
            rew_dict = {aid: global_r for aid in self.agent_ids}
            rew_dict["__all__"] = global_r
            # (A6) return（兼容 done 输出）
            return obs_dict, rew_dict, done_pack.get("done", done_pack), info_dict  # 兼容输出
            #如果你用旧风格，就返回 done_dict
            # 如果你用新风格，就返回 {terminated,truncated}


        # sr.ok True。分支 B：成功（sr.ok == True）
        # 记录 last good obs
        #(B1) 更新 last good obs。下一次如果失败，就用这个顶住。
        if sr.obs_graph is not None:
            self._last_obs_graph = sr.obs_graph
            self._last_info_meta = sr.meta

        # (B2) 算 reward（全局共享）。这会用 metrics 计算惩罚项（电压越界、反向潮流等）。
        global_r = self._compute_global_reward(sr, joint_action)

        # (B3) obs_dict：把新的 obs_graph 广播给每个 agent
        obs_dict = self._broadcast_obs(sr.obs_graph)

        # 全局共享 reward：每个 agent 一样（先稳，后面再分配 credit）
        # (B4) rew_dict：每个 agent 都拿同一个 global_r
        rew_dict = {aid: float(global_r) for aid in self.agent_ids}
        rew_dict["__all__"] = float(global_r)

        # (B5) done：正常情况下不会 terminated，只会 truncated（时间到）
        terminated = False
        truncated = bool(reached_horizon)

        done_pack = self._make_done_flags(terminated=terminated, truncated=truncated)

        # (B6) info：把 targets 放进去（非常重要！）
        # 重点：targets=sr.targets
        这意味着你每一步都有：
        #     •	obs_graph（forecast + last_vm 等构造的“输入图”）
        #     •	targets（潮流真实结果，线功率、节点电压、rpf标签等）
        # 这就是 GNN 监督学习最需要的 “(X, Y)” 配对数据来源。
        info_common = {
            "ok": True,
            "meta": sr.meta,
            "info": sr.info,
            "metrics": sr.metrics,
            # targets 对监督训练（GNN/预测任务）非常重要，所以这里直接带上
            "targets": sr.targets,
            "joint_action": joint_action.copy(),
        }
        info_dict = {aid: dict(info_common) for aid in self.agent_ids}

        return obs_dict, rew_dict, done_pack.get("done", done_pack), info_dict

    # -----------------------------
    # Helpful accessors
    # -----------------------------
    # get_agent_ids()
    	# 返回所有 agent 的名字列表，比如：
        # ["pv_0", "pv_1", "pv_2"]  
    #     为什么需要？
	# •	训练代码（无论 RLlib 还是自己写）都需要知道：
	# •	这局里有哪些 agent
	# •	actions / obs / reward / info 的 key 应该有哪些
	# •	这样你在写训练 loop 的时候不会写死 agent 数量。  
    def get_agent_ids(self) -> List[str]:
        return list(self.agent_ids)
    
    # get_action_space_hint()它不是“真的 action space”，
    #只是一个提示说明书：告诉你每个 agent 的动作大概长什么样、范围是什么。
    #比如返回：
	# 没有 curtail 的情况下：
    #         {
    #     "pv_0": {"dim": 1, "q": (-1, 1)},
    #     "pv_1": {"dim": 1, "q": (-1, 1)},
    #     ...
    #     }
    #有 curtail 的情况下：
    #     {
    # "pv_0": {"dim": 2, "q": (-1,1), "curt": (0,1)},
    # ...
    # }
    def get_action_space_hint(self) -> Dict[str, Any]:
        """
        不强绑定 gym.Space，但给一个“动作范围提示”，方便你接 RLlib / SB3 / 自写训练 loop。
        """
        #根据 backend.mode 判断 q 的范围
        #   q_frac：你设计的是 -1 到 +1 的比例（后面在 backend 里乘以 qlim）
        # q_mvar：动作是 真实 Mvar，但最大可用 Q 取决于当前 P（qlim = sqrt(S^2 - P^2)），所以范围是动态的。
        if self.backend.mode == "q_frac":
            q_range = (-1.0, 1.0)
        else:
            # q_mvar 模式下，每个 PV 的 qlim 随 p_now 变化，严格范围要运行时算
            q_range = ("dynamic", "dynamic")

        if not self.backend.enable_curtail_action:
            return {aid: {"dim": 1, "q": q_range} for aid in self.agent_ids}
        else:
            return {aid: {"dim": 2, "q": q_range, "curt": (0.0, 1.0)} for aid in self.agent_ids}

    def get_obs_space_hint(self) -> Dict[str, Any]:
        """
        同样给 hint：obs 是图张量字典（x, edge_index, edge_attr 等）。
        真实 shape 由 feeder 决定。
        """
        return {
            "type": "graph_dict",
            "keys": ["x", "edge_index", "edge_attr"],
            "note": "each agent receives the same global graph by default; you can slice subgraphs later",
        }