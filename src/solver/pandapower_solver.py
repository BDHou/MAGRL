import logging
from re import T
import warnings
import pandapower as pp
import pandapower.topology as top
from typing import Dict
import pandas as pd
import networkx as nx

# 禁用pandapower相关的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandapower")
warnings.filterwarnings("ignore", category=UserWarning, module="pandapower")
warnings.filterwarnings("ignore", message=".*Matrix is exactly singular.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
class GridPandapowerSolver:
    def __init__(self, logger=None):
        self.logger = logger
    
    # def reset_network(self):
    #     net = pp.create_empty_network()

    def run_powerflow(self, 
        net: pp.pandapowerNet = None, 
        enable_multi_source: bool = False,
        allow_distribution_equiv: bool = False
    ):
        '''
        运行潮流计算，需要把数据字典转换为pandapower网络再作为数据。因为直接改net性能会好很多
        Args:
            net: pandapower网络
        Returns:
            Dict[str, pd.DataFrame]: 潮流计算后的结果，只返回以'res_'开头的DataFrame，去掉前缀作为key
        '''

        try:
            if enable_multi_source:
                self.active_island_gens_as_slack(net, allow_distribution_equiv)
                
            pp.runpp(
                net,
                algorithm="nr", 
                tolerance_mva=1e-8, 
                max_iteration=1000, 
                init="flat", 
                enforce_q_lims=True, 
                check_connectivity=True,
                numba=True
            )
        except Exception as e:
            raise e
        res = self.get_powerflow_result(net)

        # TBD
        # self.analyze_islands(net, respect_switches=True, limit_print=2000)
        # exit()
        # TBD
        return res

    def voltage_sensitivity(
        self,
        net: pp.pandapowerNet,
        dp_mw: float = 0.01,
        dq_mvar: float = 0.01,
        target_buses=None,
        observe_buses=None,
    ) -> Dict[str, pd.DataFrame]:
        """
        基于有限差分计算电压灵敏度矩阵：
        dV/dP 与 dV/dQ（V 为母线电压幅值 pu）。

        Args:
            net: pandapower 网络
            dp_mw: 有功扰动 (MW)
            dq_mvar: 无功扰动 (MVar)
            target_buses: 扰动施加的母线列表，默认全部在运母线
            observe_buses: 观测电压的母线列表，默认全部在运母线

        Returns:
            Dict[str, pd.DataFrame]: {"dV_dP": df, "dV_dQ": df}
            行为观测母线，列为扰动母线
        """
        if target_buses is None:
            target_buses = list(net.bus[net.bus.in_service].index)
        if observe_buses is None:
            observe_buses = list(net.bus[net.bus.in_service].index)

        # 先跑一次基准潮流
        pp.runpp(
            net,
            algorithm="nr",
            tolerance_mva=1e-8,
            max_iteration=1000,
            init="flat",
            enforce_q_lims=True,
            check_connectivity=True,
            numba=True,
        )
        base_vm = net.res_bus.vm_pu.copy()

        # 创建一个临时负荷，后续不断移动到不同母线
        temp_idx = pp.create_load(net, bus=target_buses[0], p_mw=0.0, q_mvar=0.0, name="temp_sens")

        dV_dP = pd.DataFrame(index=observe_buses, columns=target_buses, dtype=float)
        dV_dQ = pd.DataFrame(index=observe_buses, columns=target_buses, dtype=float)

        for bus in target_buses:
            net.load.at[temp_idx, "bus"] = bus

            # dV/dP
            net.load.at[temp_idx, "p_mw"] = dp_mw
            net.load.at[temp_idx, "q_mvar"] = 0.0
            pp.runpp(
                net,
                algorithm="nr",
                tolerance_mva=1e-8,
                max_iteration=1000,
                init="flat",
                enforce_q_lims=True,
                check_connectivity=True,
                numba=True,
            )
            dv_p = (net.res_bus.vm_pu - base_vm).loc[observe_buses] / dp_mw
            dV_dP[bus] = dv_p.values

            # dV/dQ
            net.load.at[temp_idx, "p_mw"] = 0.0
            net.load.at[temp_idx, "q_mvar"] = dq_mvar
            pp.runpp(
                net,
                algorithm="nr",
                tolerance_mva=1e-8,
                max_iteration=1000,
                init="flat",
                enforce_q_lims=True,
                check_connectivity=True,
                numba=True,
            )
            dv_q = (net.res_bus.vm_pu - base_vm).loc[observe_buses] / dq_mvar
            dV_dQ[bus] = dv_q.values

        # 清理临时负荷
        net.load.drop(index=temp_idx, inplace=True)

        return {"dV_dP": dV_dP, "dV_dQ": dV_dQ}

    def active_island_gens_as_slack(self, net, allow_distribution_equiv: bool = False):
        """
        为每个孤岛选举并创建平衡节点(Slack/Ext_Grid)，保证潮流收敛。
        
        业务逻辑（按优先级）：
        1. 遍历所有孤岛（连通分量）
        2. 如果孤岛已有在运的 ext_grid → 跳过
        3. 如果孤岛有在运的 gen → 选 p_mw 最大的设为 slack=True
        4. 如果都没有 → 在 sgen/load 中选 |p_mw| 最大的，在其 bus 上新建 ext_grid，并关闭原元件
        
        Args:
            net: pandapower 网络对象
            allow_distribution_equiv bool: 是否允许配网元件(net_type='distribution')参与等效为ext_grid，
                                           默认为False，即不允许配网的sgen和load进行等效
        
        Returns:
            activated_count int: 处理的孤岛数量（设置 slack gen 或 新建 ext_grid）
        """
        # 1. 先把所有 Gen 的 slack 属性重置为 False
        # (这一步极重要！防止上个 Step 是孤岛，这个 Step 并网了，结果导致双 Slack 冲突)
        net.gen['slack'] = False 
        
        # 2. 生成拓扑图并获取所有连通分量
        mg = top.create_nxgraph(net, respect_switches=True)
        islands = list(nx.connected_components(mg))
        
        # 3. 找出已有 ext_grid 的母线集合
        active_ext_buses = set(net.ext_grid[net.ext_grid.in_service].bus)
        
        # 4. 计算新 ext_grid 的起始索引，避免冲突
        if net.ext_grid.empty:
            next_ext_grid_idx = 0
        else:
            next_ext_grid_idx = net.ext_grid.index.max() + 1
        
        activated_count = 0
        new_ext_grid_count = 0
        
        for island in islands:
            island_buses = set(island)
            
            # 4.1 如果该岛已有在运的 ext_grid，跳过
            if not active_ext_buses.isdisjoint(island_buses):
                continue
            
            # 4.2 检查岛内是否有在运的 gen
            gens_in_island = net.gen[net.gen.in_service & net.gen.bus.isin(island_buses)]
            
            if not gens_in_island.empty:
                # 有 gen：选 p_mw 最大的设为 slack
                boss_gen_idx = gens_in_island['p_mw'].idxmax()
                net.gen.at[boss_gen_idx, 'slack'] = True
                activated_count += 1
                continue

            # --- 4.3 & 4.4 收集候选者 (sgen 和 load) ---
            # 构造筛选掩码
            sg_mask = net.sgen.in_service & net.sgen.bus.isin(island_buses)
            ld_mask = net.load.in_service & net.load.bus.isin(island_buses)

            # 处理配网过滤逻辑
            if not allow_distribution_equiv:
                if 'net_type' in net.sgen.columns:
                    sg_mask &= (net.sgen.net_type != 'distribution')
                if 'net_type' in net.load.columns:
                    ld_mask &= (net.load.net_type != 'distribution')

            # 提取关键列并标记来源表
            sg_cand = net.sgen.loc[sg_mask, ['bus', 'p_mw']].assign(table='sgen')
            ld_cand = net.load.loc[ld_mask, ['bus', 'p_mw']].assign(table='load')

            candidates = pd.concat([sg_cand, ld_cand])

            if candidates.empty:
                continue

            # --- 4.5 选出 |p_mw| 最大的优胜者 ---
            # 使用 idxmax 找到绝对值最大的行索引
            winner_idx = candidates['p_mw'].abs().idxmax()
            winner = candidates.loc[winner_idx]

            # --- 4.6 创建新的 ext_grid ---
            # 自动分配 index 避免冲突，同时保持名称可追溯
            new_ext_idx = pp.create_ext_grid(
                net, 
                bus=winner.bus, 
                vm_pu=1.0, 
                name=f"island_slack_from_{winner.table}_{winner_idx}",
                in_service=True
            )

            # --- 4.7 关停原元件 ---
            # 根据 winner 的 table 属性动态定位并修改
            if winner.table == 'sgen':
                net.sgen.at[winner_idx, 'in_service'] = False
            else:
                net.load.at[winner_idx, 'in_service'] = False
            
            # # 4.3 没有 gen，收集岛内所有在运的 sgen 和 load 作为候选者
            # candidates = []
            
            # # 收集 sgen 候选者
            # sgens_in_island = net.sgen[net.sgen.in_service & net.sgen.bus.isin(island_buses)]
            # for idx, row in sgens_in_island.iterrows():
            #     # 检查是否为配网元件，如果不允许配网等效则跳过
            #     if not allow_distribution_equiv:
            #         net_type = row.get('net_type', '') if 'net_type' in net.sgen.columns else ''
            #         if net_type == 'distribution':
            #             continue
            #     candidates.append({
            #         'type': 'sgen',
            #         'idx': idx,
            #         'bus': row.bus,
            #         'p_mw': row.p_mw,
            #         'score': abs(row.p_mw)
            #     })
            
            # # 收集 load 候选者
            # loads_in_island = net.load[net.load.in_service & net.load.bus.isin(island_buses)]
            # for idx, row in loads_in_island.iterrows():
            #     # 检查是否为配网元件，如果不允许配网等效则跳过
            #     if not allow_distribution_equiv:
            #         net_type = row.get('net_type', '') if 'net_type' in net.load.columns else ''
            #         if net_type == 'distribution':
            #             continue
            #     candidates.append({
            #         'type': 'load',
            #         'idx': idx,
            #         'bus': row.bus,
            #         'p_mw': row.p_mw,
            #         'score': abs(row.p_mw)
            #     })
            
            # # 4.4 如果没有候选者，跳过这个孤岛
            # if not candidates:
            #     continue
            
            # # 4.5 选出得分最高的候选者（|p_mw| 最大）
            # winner = max(candidates, key=lambda x: x['score'])
            
            # # 4.6 在获胜元件的 Bus 上新建 ext_grid
            # winner_bus = winner['bus']
            # new_ext_idx = next_ext_grid_idx + new_ext_grid_count
            
            # pp.create_ext_grid(
            #     net, 
            #     bus=winner_bus, 
            #     vm_pu=1.0, 
            #     name=f"island_slack_{new_ext_idx}",
            #     index=new_ext_idx,
            #     in_service=True
            # )
            
            # # 4.7 将获胜的原元件 in_service 设为 False，防止功率重复计算
            # if winner['type'] == 'sgen':
            #     net.sgen.at[winner['idx'], 'in_service'] = False
            # else:  # load
            #     net.load.at[winner['idx'], 'in_service'] = False
            
            activated_count += 1
            # new_ext_grid_count += 1
                
        return activated_count

    def analyze_islands(self, net, respect_switches: bool = True, limit_print: int = 10):
        """
        分析 pandapower 网络的孤岛情况
        :param net: pandapower 网络
        :param respect_switches: 是否考虑开关状态
        :param limit_print: 只打印前 N 个孤岛的详细信息
        """
        print(f"\n{'='*20} 分析孤岛 (考虑开关: {respect_switches}) {'='*20}")
        
        # 1. 生成图
        mg = top.create_nxgraph(net, respect_switches=respect_switches)

        # 2. 获取连通分量 (Islands)
        # 结果是一个生成器，转为列表并按节点数量从小到大排序
        # islands = sorted(list(nx.connected_components(mg)), key=len)
        islands = sorted(list(nx.connected_components(mg)), key=len)[-20:]
        # islands = sorted(list(nx.connected_components(mg)), key=len)[:-1]
        
        print(f"总计发现 {len(islands)} 个独立连通区域 (Islands)")
        
        # 3. 统计“有源孤岛”和“无源孤岛”
        # 获取外网 (External Grid) 所在的母线 ID
        ext_grid_buses = set(net.ext_grid.bus.values)
        gen_buses = set(net.gen.bus.values) if not net.gen.empty else set()
        sgen_buses = set(net.sgen.bus.values) if not net.sgen.empty else set()
        load_buses = set(net.load.bus.values) if not net.load.empty else set()
        source_buses = gen_buses
        # source_buses = ext_grid_buses
        # source_buses = ext_grid_buses | gen_buses
        # source_buses = ext_grid_buses | gen_buses | sgen_buses
        
        energized_count = 0
        dead_count = 0
        
        # 4. 详细打印前 N 个小岛
        print(f"\n--- 打印前 {limit_print} 个最小的孤岛 (通常是问题所在) ---")
        cnt = 0
        for i, island_nodes in enumerate(islands):
            cnt += (len(island_nodes))
            island_list = list(island_nodes)
            
            # 检查该岛屿是否有电源
            has_source = not source_buses.isdisjoint(island_nodes)
            if has_source:
                energized_count += 1
                status = "🟢 有源 (Energized)"
            else:
                dead_count += 1
                status = "🔴 无源 (Dead/Blackout)"
            
            has_load = not load_buses.isdisjoint(island_nodes)
            if has_load:
                energized_count += 1
                status += " | 🟢 有荷 (Load)"
            else:
                dead_count += 1
                status += " | 🔴 无荷 (No Load)"
                
            # 仅打印前 limit_print 个，或者特定的“大死岛”
            if i < limit_print:
                # 获取这些母线上的负载
                load_mask = net.load.bus.isin(island_list)
                island_loads = net.load[load_mask]
                total_load = island_loads.p_mw.sum()
                
                # 分离正负荷和负负荷（负的p_mw通常表示分布式电源）
                positive_loads = island_loads[island_loads.p_mw >= 0]
                negative_loads = island_loads[island_loads.p_mw < 0]
                positive_load_sum = positive_loads.p_mw.sum()
                negative_load_sum = negative_loads.p_mw.sum()
                
                # 按net_type区分主网负荷和配网负荷（默认transmission，空字符串和NaN都视为transmission）
                if 'net_type' in island_loads.columns:
                    net_types = island_loads['net_type'].replace('', 'transmission').fillna('transmission')
                else:
                    net_types = 'transmission'
                trans_loads = island_loads[net_types == 'transmission']
                distr_loads = island_loads[net_types == 'distribution']
                trans_load_sum = trans_loads.p_mw.sum()
                distr_load_sum = distr_loads.p_mw.sum()
                
                # 获取电压等级
                vn_kv = net.bus.loc[island_list, 'vn_kv'].unique()
                
                print(f"\n[Island] 孤岛 #{i+1} | 节点数: {len(island_list)} | {status}")
                # print(f"   包含母线 IDs: {island_list}")
                print(f"   电压等级: {vn_kv} kV")
                print(f"   岛内总负荷: {total_load:.4f} MW (正负荷: {positive_load_sum:.4f} MW, 负负荷: {negative_load_sum:.4f} MW)")
                print(f"   主网负荷: {trans_load_sum:.4f} MW ({len(trans_loads)}个), 配网负荷: {distr_load_sum:.4f} MW ({len(distr_loads)}个)")
                if len(negative_loads) > 0:
                    print(f"   负负荷详情 (共{len(negative_loads)}个):")
                    for idx, row in negative_loads.iterrows():
                        nt = row.get('net_type', 'transmission') if 'net_type' in island_loads.columns else 'transmission'
                        nt = nt if nt else 'transmission'  # 空字符串也视为transmission
                        # print(f"      - load[{idx}] @ bus {row.bus}: {row.p_mw:.4f} MW ({nt})")
                
                # 检查是否有开关连接到这个岛（仅作为调试参考）
                # 这可以帮你看是不是开关断开导致它隔离
                if respect_switches:
                    # 这是一个简单的启发式检查，看该岛边缘是否有断开的开关
                    print("   (提示: 检查连接这些母线的 line/trafo 上的 switch 是否为 open)")

        print(f"\n{'-'*40}")
        print(f"统计总结:")
        print(f"  总失电节点数: {cnt}")
        print(f"  包含电源的区域: {energized_count} 个 (正常运行区域)")
        print(f"  完全失电的区域: {dead_count} 个 (这是你需要处理的垃圾数据或停电区)")

        dead_buses = top.unsupplied_buses(net, respect_switches=True)
        print(f"失电母线索引集合: {len(dead_buses)}")
        
        dead_buses_idx = net.res_bus[net.res_bus['vm_pu'].isna()].index
        print(f"结果异常的母线: {len(dead_buses_idx)}")


        print('分析潮流与拓扑计算不一致的原因')
        # 1. 获取拓扑上失电的集合 (物理断开)
        topo_dead_set = top.unsupplied_buses(net, respect_switches=True)

        # 2. 获取潮流计算失败的集合 (数学无解)
        # 假设 runpp 已经跑过了，vm_pu 为 NaN 或者 0 的都算失败
        # pf_dead_set = set(net.res_bus[net.res_bus.vm_pu.isna() | (net.res_bus.vm_pu == 0)].index)
        in_service_buses = net.bus[net.bus.in_service].index
        pf_dead_set = set(net.res_bus[
            (net.res_bus.vm_pu.isna() | (net.res_bus.vm_pu == 0)) & 
            (net.res_bus.index.isin(in_service_buses))
        ].index)


        # 3. 计算“差集”：明明连着，却算不出数的节点
        ghost_buses = pf_dead_set - topo_dead_set

        print(f"👻 幽灵节点数量: {len(ghost_buses)}")
        print(f"这些节点物理相连，但潮流计算失败。")

        # 4. 诊断这些幽灵节点所在的孤岛情况
        if len(ghost_buses) > 0:
            # 取出一个幽灵节点看看
            sample_bus = list(ghost_buses)[0]
            
            # 找到这个节点所在的连通分量(孤岛)
            mg = top.create_nxgraph(net, respect_switches=True)
            
            # --- 修复开始 ---
            # 1. 先检查该节点是否在图里 (避免 StopIteration)
            if sample_bus not in mg.nodes:
                print(f"⚠️ 节点 {sample_bus} 不在图 mg 中！(原因: 可能是 in_service=False)")
                # 检查一下它的状态
                in_service = net.bus.at[sample_bus, 'in_service']
                print(f"   检查 net.bus.in_service: {in_service}")
            else:
                # 2. 如果在图里，再找它属于哪个岛
                island = next(c for c in nx.connected_components(mg) if sample_bus in c)
                
                print(f"\n--- 典型幽灵岛分析 (包含节点 {sample_bus}) ---")
                print(f"岛内节点总数: {len(island)}")
                
                # 检查岛内有没有 Ext_Grid
                has_ext_grid = not set(net.ext_grid.bus).isdisjoint(island)
                # 检查岛内有没有 Sgen
                has_sgen = not set(net.sgen.bus).isdisjoint(island)
                
                print(f"是否有外网 (Slack): {'✅' if has_ext_grid else '❌'}")
                print(f"是否有分布式电源 (Sgen): {'✅' if has_sgen else '❌'}")
                
                if has_sgen and not has_ext_grid:
                    print("\n👉 破案了：这是一个只有光伏/Sgen，没有主网/Slack的孤岛。")

        return islands

    def run_powerflow_and_analyze(self, grid_data_dict: Dict) -> Dict:
        """
        输入一个数据字典，完成setup，求解，并分析潮流结果。
        Args:
            grid_data_dict: 电网数据字典
        Returns:
            Dict: 包含 'powerflow_result' 和 'analysis_result' 的字典
        """
        # 1. Setup Network
        net = self.setup_network_from_data_dict(grid_data_dict)
        return self.run_powerflow_on_net_and_analyze(net, grid_data_dict)

    def run_powerflow_on_net_and_analyze(self, net, grid_data_dict: Dict) -> Dict:
        """
        在给定的pandapower网络上运行潮流，并使用grid_data_dict进行分析。
        跳过网络构建过程，用于加速。
        
        Args:
            net: 已构建的pandapower网络对象
            grid_data_dict: 对应的电网数据字典（用于拓扑分析等）
        Returns:
            Dict: 包含 'powerflow_result' 和 'analysis_result' 的字典
        """
        from src.parsers.result_analyzer import ResultAnalyzer
        from src.analysis.topology import GridTopology

        # 2. Run Powerflow
        powerflow_result = self.run_powerflow(net)
        
        # 3. Analyze Result
        # 需要构建graph (如果拓扑分析很慢，这里可能也需要优化，但先按照需求只优化setup_network)
        topology = GridTopology.from_data_dict(grid_data_dict, logger=self.logger)
        
        analyzer = ResultAnalyzer(logger=self.logger)
        analysis_result = analyzer.analyze_powerflow_result(
            powerflow_result=powerflow_result,
            graph=topology.graph,
            net_data=grid_data_dict
        )
        
        return {
            "powerflow_result": powerflow_result,
            "analysis_result": analysis_result
        }
    
    def get_network_status(self, net):
        out = {}
        for k, df in net.items():
            if isinstance(df, pd.DataFrame) and df.empty:
                continue
            if (k.startswith("res_") and isinstance(df, pd.DataFrame)):
                continue
            out[k] = df
        return out
    
    def get_powerflow_result(self, net):
        '''
        获取潮流计算后的结果，只返回以'res_'开头的DataFrame，去掉前缀作为key
        Args:
            net: pandapower网络
        Returns:
            Dict[str, pd.DataFrame]: 潮流计算后的结果，只返回以'res_'开头的DataFrame，去掉前缀作为key
        '''
        out = {}
        static_cols_to_merge = ['psr_id', 'name', 'feeder', 'station', 'in_service']

        # for k, df in net.items():
        #     if isinstance(df, pd.DataFrame) and df.empty:
        #         continue
        #     if not (k.startswith("res_") and isinstance(df, pd.DataFrame)):
        #         continue
        #     elem = k[4:]  # 去掉前缀res_
        #     out[elem] = df

        for k, res_df in net.items():
            # 基础过滤：必须是 'res_' 开头，且是 DataFrame，且不为空
            if not (isinstance(k, str) and k.startswith("res_")):
                continue
            if not isinstance(res_df, pd.DataFrame) or res_df.empty:
                continue
                
            # 提取元件类型，例如 'res_line' -> 'line'
            element_type = k[4:] 
            
            # 获取对应的原始输入表 (静态表)
            input_df = net.get(element_type)

            # 执行合并 (Left Join via Index)
            # pandapower 保证了 res_df 的 index 和 input_df 的 index 是对齐的
            # .join() 默认按索引合并，效率很高
            # print(input_df.head)
            # print(element_type)
            try:
                out[element_type] = res_df.join(input_df[static_cols_to_merge])
            except Exception as e:
                # logging.debug(f"Error merging {element_type}: {e}")
                pass

        return out
    
    def setup_network_from_data_dict(self, grid_data_dict: Dict):
        """
        把grid_data_dict转换为pandapower网络
        Args:
            grid_data_dict: 电网数据字典
        Returns:
            net: pandapower网络
        """
        net = pp.create_empty_network()

        indexes, names, vn_kvs, in_services = [], [], [], []
        feeders, stations, psr_ids, net_types = [], [], [], []
        for bus in grid_data_dict['bus']:
            feeders.append(bus.get('feeder', ''))
            stations.append(bus.get('station', ''))
            psr_ids.append(bus.get('psr_id', ''))
            net_types.append(bus.get('net_type', ''))
            indexes.append(bus['id'])
            names.append(bus.get('name', ''))
            vn_kvs.append(bus.get('vn_kv', 0))
            in_services.append(bus.get('in_service', True))
        pp.create_buses(net, nr_buses=len(indexes), name=names, index=indexes, vn_kv=vn_kvs, in_service=in_services)
        net.bus['feeder'] = feeders
        net.bus['station'] = stations
        net.bus['psr_id'] = psr_ids
        net.bus['net_type'] = net_types

        if 'load' in grid_data_dict:
            indexes, names, buses, p_mws, q_mvars, in_services = [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for load in grid_data_dict['load']:
                try:
                    indexes.append(load['id'])
                    names.append(load.get('name', ''))
                    buses.append(load['bus'])
                    p_mws.append(load.get('p_mw', 0))
                    q_mvars.append(load.get('q_mvar', 0))
                    in_services.append(load.get('in_service', True))
                    feeders.append(load.get('feeder', ''))
                    stations.append(load.get('station', ''))
                    psr_ids.append(load.get('psr_id', ''))
                    net_types.append(load.get('net_type', ''))
                except Exception as e:
                    print(load)
                    raise e
            pp.create_loads(net, name=names, index=indexes, buses=buses, p_mw=p_mws, q_mvar=q_mvars, in_service=in_services)
            # print(feeders)
            # print(stations)
            # print(psr_ids)
            # exit()
            net.load['feeder'] = feeders
            net.load['station'] = stations
            net.load['psr_id'] = psr_ids
            net.load['net_type'] = net_types
        if 'shunt' in grid_data_dict:
            indexes, names, buses, p_mws, q_mvars, in_services = [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for shunt in grid_data_dict['shunt']:
                indexes.append(shunt['id'])
                names.append(shunt.get('name', ''))
                buses.append(shunt['bus'])
                p_mws.append(shunt.get('p_mw', 0))
                q_mvars.append(shunt.get('q_mvar', 0))
                in_services.append(shunt.get('in_service', True))
                feeders.append(shunt.get('feeder', ''))
                stations.append(shunt.get('station', ''))
                psr_ids.append(shunt.get('psr_id', ''))
                net_types.append(shunt.get('net_type', ''))
            pp.create_shunts(net, name=names, index=indexes, buses=buses, p_mw=p_mws, q_mvar=q_mvars, in_service=in_services)
            net.shunt['feeder'] = feeders
            net.shunt['station'] = stations
            net.shunt['psr_id'] = psr_ids
            net.shunt['net_type'] = net_types

        if 'ext_grid' in grid_data_dict:
            psr_ids = []
            for ext_grid in grid_data_dict['ext_grid']:
                pp.create_ext_grid(
                    net,
                    name=ext_grid.get('name', ''),
                    index=ext_grid.get('id', 0),
                    bus=ext_grid['bus'],
                    vm_pu=ext_grid.get('vm_pu', 0),
                    va_degree=ext_grid.get('va_degree', 0),
                    in_service=ext_grid.get('in_service', True),
                )
                psr_ids.append(ext_grid.get('psr_id', ''))
            net.ext_grid['psr_id'] = psr_ids

        if 'gen' in grid_data_dict:
            indexes, names, buses, p_mws, vm_pus, va_degrees, in_services, slacks = [], [], [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for gen in grid_data_dict['gen']:
                indexes.append(gen['id'])
                names.append(gen.get('name', ''))
                buses.append(gen['bus'])
                p_mws.append(gen.get('p_mw', 0))
                vm_pus.append(gen.get('vm_pu', 0))
                va_degrees.append(gen.get('va_degree', 0))
                in_services.append(gen.get('in_service', True))
                slacks.append(gen.get('slack', False))
                feeders.append(gen.get('feeder', ''))
                stations.append(gen.get('station', ''))
                psr_ids.append(gen.get('psr_id', ''))
                net_types.append(gen.get('net_type', ''))
            pp.create_gens(net, name=names, index=indexes, buses=buses, p_mw=p_mws, vm_pu=vm_pus, va_degree=va_degrees, in_service=in_services, slack=slacks)
            net.gen['feeder'] = feeders
            net.gen['station'] = stations
            net.gen['psr_id'] = psr_ids
            net.gen['net_type'] = net_types

        if 'sgen' in grid_data_dict:
            indexes, names, buses, p_mws, vm_pus, va_degrees, in_services = [], [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for gen in grid_data_dict['sgen']:
                indexes.append(gen['id'])
                names.append(gen.get('name', ''))
                buses.append(gen['bus'])
                p_mws.append(gen.get('p_mw', 0))
                vm_pus.append(gen.get('vm_pu', 0))
                va_degrees.append(gen.get('va_degree', 0))
                in_services.append(gen.get('in_service', True))
                feeders.append(gen.get('feeder', ''))
                stations.append(gen.get('station', ''))
                psr_ids.append(gen.get('psr_id', ''))
                net_types.append(gen.get('net_type', ''))
            pp.create_sgens(net, name=names, index=indexes, buses=buses, p_mw=p_mws, vm_pu=vm_pus, va_degree=va_degrees, in_service=in_services)
            net.sgen['feeder'] = feeders
            net.sgen['station'] = stations
            net.sgen['psr_id'] = psr_ids
            net.sgen['net_type'] = net_types

        if 'line' in grid_data_dict:
            indexes, names, from_buses, to_buses, length_kms, r_ohm_per_kms, x_ohm_per_kms, c_nf_per_kms, max_i_kas, in_services = [], [], [], [], [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for line in grid_data_dict['line']:
                indexes.append(line['id'])
                names.append(line.get('name', ''))
                from_buses.append(line['from_bus'])
                to_buses.append(line['to_bus'])
                length_kms.append(line['length_km'])
                r_ohm_per_kms.append(line['r_ohm_per_km'])
                x_ohm_per_kms.append(line['x_ohm_per_km'])
                c_nf_per_kms.append(line['c_nf_per_km'])
                max_i_kas.append(line.get('max_i_ka', 0))
                in_services.append(line.get('in_service', True))
                feeders.append(line.get('feeder', ''))
                stations.append(line.get('station', ''))
                psr_ids.append(line.get('psr_id', ''))
                net_types.append(line.get('net_type', ''))
            pp.create_lines_from_parameters(net, from_buses=from_buses, to_buses=to_buses, length_km=length_kms, r_ohm_per_km=r_ohm_per_kms, x_ohm_per_km=x_ohm_per_kms, c_nf_per_km=c_nf_per_kms, max_i_ka=max_i_kas, name=names, index=indexes, in_service=in_services)
            net.line['feeder'] = feeders
            net.line['station'] = stations
            net.line['psr_id'] = psr_ids
            net.line['net_type'] = net_types

        if 'trafo' in grid_data_dict:
            indexes, names, hv_buses, lv_buses, sn_mvas, vk_percents, vkr_percents, tap_poss, tap_neutrals, tap_maxs, tap_mins, tap_step_percents, pfe_kws, i0_percents, tap_sides, vn_lv_kvs, vn_hv_kvs, in_services = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for trafo in grid_data_dict['trafo']:
                indexes.append(trafo['id'])
                names.append(trafo.get('name', ''))
                hv_buses.append(trafo['hv_bus'])
                lv_buses.append(trafo['lv_bus'])
                sn_mvas.append(trafo['sn_mva'])
                vk_percents.append(trafo['vk_percent'])
                vkr_percents.append(trafo['vkr_percent'])
                tap_poss.append(trafo['tap_pos'])
                tap_neutrals.append(trafo['tap_neutral'])
                tap_maxs.append(trafo['tap_max'])
                tap_mins.append(trafo['tap_min'])
                tap_step_percents.append(trafo['tap_step_percent'])
                pfe_kws.append(trafo['pfe_kw'])
                i0_percents.append(trafo['i0_percent'])
                tap_sides.append(trafo['tap_side'])
                vn_lv_kvs.append(trafo['vn_lv_kv'])
                vn_hv_kvs.append(trafo['vn_hv_kv'])
                in_services.append(trafo.get('in_service', True))
                feeders.append(trafo.get('feeder', ''))
                stations.append(trafo.get('station', ''))
                psr_ids.append(trafo.get('psr_id', ''))
                net_types.append(trafo.get('net_type', ''))
            pp.create_transformers_from_parameters(net, hv_buses=hv_buses, lv_buses=lv_buses, sn_mva=sn_mvas, vn_lv_kv=vn_lv_kvs, vn_hv_kv=vn_hv_kvs, vkr_percent=vkr_percents, vk_percent=vk_percents, pfe_kw=pfe_kws, i0_percent=i0_percents, tap_side=tap_sides, tap_neutral=tap_neutrals, tap_max=tap_maxs, tap_min=tap_mins, tap_step_percent=tap_step_percents, name=names, index=indexes, in_service=in_services)
            net.trafo['feeder'] = feeders
            net.trafo['station'] = stations
            net.trafo['psr_id'] = psr_ids
            net.trafo['net_type'] = net_types
        if 'trafo3w' in grid_data_dict:
            indexes, names, hv_buses, mv_buses, lv_buses, vn_lv_kvs, vn_hv_kvs, vn_mv_kvs, sn_hv_mvas, sn_mv_mvas, sn_lv_mvas, vk_hv_percents, vk_mv_percents, vk_lv_percents, vkr_hv_percents, vkr_mv_percents, vkr_lv_percents, pfe_kws, i0_percents, tap_maxs, tap_mins, tap_neutrals, tap_step_percents, tap_sides, tap_poss, in_services = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            feeders, stations, psr_ids, net_types = [], [], [], []
            for trafo3w in grid_data_dict['trafo3w']:
                try:
                    indexes.append(trafo3w['id'])
                    names.append(trafo3w.get('name', ''))
                    hv_buses.append(trafo3w['hv_bus'])
                    mv_buses.append(trafo3w['mv_bus'])
                    lv_buses.append(trafo3w['lv_bus'])
                    vn_lv_kvs.append(trafo3w['vn_lv_kv'])
                    vn_hv_kvs.append(trafo3w['vn_hv_kv'])
                    vn_mv_kvs.append(trafo3w['vn_mv_kv'])
                    sn_hv_mvas.append(trafo3w['sn_hv_mva'])
                    sn_mv_mvas.append(trafo3w['sn_mv_mva'])
                    sn_lv_mvas.append(trafo3w['sn_lv_mva'])
                    vk_hv_percents.append(trafo3w['vk_hv_percent'])
                    vk_mv_percents.append(trafo3w['vk_mv_percent'])
                    vk_lv_percents.append(trafo3w['vk_lv_percent'])
                    vkr_hv_percents.append(trafo3w['vkr_hv_percent'])
                    vkr_mv_percents.append(trafo3w['vkr_mv_percent'])
                    vkr_lv_percents.append(trafo3w['vkr_lv_percent'])
                    pfe_kws.append(trafo3w['pfe_kw'])
                    i0_percents.append(trafo3w['i0_percent'])
                    tap_maxs.append(trafo3w['tap_max'])
                    tap_mins.append(trafo3w['tap_min'])
                    tap_neutrals.append(trafo3w['tap_neutral'])
                    tap_step_percents.append(trafo3w['tap_step_percent'])
                    tap_sides.append('hv')
                    tap_poss.append(trafo3w['tap_pos'])
                    in_services.append(trafo3w.get('in_service', True))
                    feeders.append(trafo3w.get('feeder', ''))
                    stations.append(trafo3w.get('station', ''))
                    psr_ids.append(trafo3w.get('psr_id', ''))
                    net_types.append(trafo3w.get('net_type', ''))
                except Exception as e:
                    print(trafo3w)
                    raise e
            pp.create_transformers3w_from_parameters(net, hv_buses=hv_buses, mv_buses=mv_buses, lv_buses=lv_buses, vn_lv_kvs=vn_lv_kvs, vn_hv_kvs=vn_hv_kvs, vn_mv_kvs=vn_mv_kvs, sn_hv_mva=sn_hv_mvas, sn_mv_mva=sn_mv_mvas, sn_lv_mva=sn_lv_mvas, vk_hv_percent=vk_hv_percents, vk_mv_percent=vk_mv_percents, vk_lv_percent=vk_lv_percents, vkr_hv_percent=vkr_hv_percents, vkr_mv_percent=vkr_mv_percents, vkr_lv_percent=vkr_lv_percents, pfe_kw=pfe_kws, i0_percent=i0_percents, tap_max=tap_maxs, tap_min=tap_mins, tap_neutral=tap_neutrals, tap_step_percent=tap_step_percents, tap_side=tap_sides, tap_pos=tap_poss, name=names, index=indexes, in_service=in_services, vn_hv_kv=vn_hv_kvs, vn_mv_kv=vn_mv_kvs, vn_lv_kv=vn_lv_kvs)
            net.trafo3w['feeder'] = feeders
            net.trafo3w['station'] = stations
            net.trafo3w['psr_id'] = psr_ids 
            net.trafo3w['net_type'] = net_types
        if 'switch' in grid_data_dict:
            # 1. 预先检查并初始化自定义列
            # 如果列不存在，先创建并填充默认值（如 None 或 False）
            custom_cols = ['is_zdhkg', 'psr_id', 'in_service']
            for col in custom_cols:
                if col not in net.switch.columns:
                    net.switch[col] = None  # 或者 pd.NA, False 等默认值

            indexes, names, buses, elements, ets, closeds= [], [], [], [], [], []
            in_services, is_zdhkgs, psr_ids, net_types = [], [], [], []
            feeders, stations = [], []
            for switch in grid_data_dict['switch']:
                try:
                    indexes.append(switch['id'])
                    names.append(switch.get('name', ''))
                    buses.append(switch['bus'])
                    elements.append(switch['element'])
                    ets.append(switch['et'])
                    closeds.append(switch['closed'])
                    in_services.append(switch.get('in_service', True))
                    is_zdhkgs.append(switch.get('is_zdhkg', None))
                    psr_ids.append(switch.get('psr_id', None))
                    feeders.append(switch.get('feeder', ''))
                    stations.append(switch.get('station', ''))
                    net_types.append(switch.get('net_type', ''))
                except Exception as e:
                    print(switch)
                    raise e
            pp.create_switches(net, name=names, index=indexes, buses=buses, elements=elements, et=ets, closed=closeds)
            # TBC: 理论上没问题，但是如果在create的时候顺序被打乱了就会出问题，应该是不会有问题的
            net.switch['is_zdhkg'] = is_zdhkgs
            net.switch['psr_id'] = psr_ids
            net.switch['in_service'] = in_services
            net.switch['feeder'] = feeders
            net.switch['station'] = stations
            net.switch['net_type'] = net_types
        return net 
