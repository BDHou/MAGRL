# grid/builder.py

import json
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

class GridBuilder:
    """
    一个专门负责从不同数据源构建电网图的工具类。
    """
    def __init__(self, logger=None):
        """
        构造函数为未来的扩展保留，例如传入配置或日志记录器。
        """
        self.logger = logger
        self.available_types = [
            'bus', 'gen', 'sgen', 'load', 'ext_grid', 'shunt',  # node elements
            'line', 'trafo', 'trafo3w', 'switch'  # connection elements
        ]  # 后续可以考虑外部输出图里需要的设备种类


    def _build_edges_with_index(self, G: nx.Graph, relationship_list: List[Tuple[str, str]]):
        """
        私有辅助方法：从关系列表创建边，并为每条边添加一个'index'属性。
        """
        # 使用 enumerate 和列表推导式，更高效、更 Pythonic
        edges_with_data = [
            (rel[0], rel[1], {'index': i}) 
            for i, rel in enumerate(relationship_list)
        ]
        G.add_edges_from(edges_with_data)
        if self.logger:
            self.logger.debug(f"The network has {len(edges_with_data)} edges.")

    def _set_node_attributes(self, G: nx.Graph, nodes_info: Dict[str, Dict], attributes_to_set:  Optional[List[str]] = None):
        """
        私有辅助方法：为图中的节点批量设置多个属性。
        
        Args:
            G: nx.Graph: 要设置属性的图对象
            nodes_info: Dict[str, Dict]: 节点信息字典，格式为 {node_id: {attr1: value1, attr2: value2, ...}}
            attributes_to_set: List[str] = None: 要设置的属性列表，如果为None则设置所有节点拥有的属性
        """
        if attributes_to_set is None:
            # 收集所有节点拥有的所有属性
            nx.set_node_attributes(G, nodes_info)
        else: 
            for keyword in attributes_to_set:
                # 准备一个 {node_id: attribute_value} 格式的字典
                # 使用 .get(keyword) 来避免当某个节点缺少该属性时出错
                attributes_dict = {
                    node_id: info.get(keyword)
                    for node_id, info in nodes_info.items()
                }
                # 一次性为所有节点设置一个属性
                nx.set_node_attributes(G, attributes_dict, keyword)

    def build_from_components(self, 
                              relationship_list: List[Tuple[str, str]], 
                              nodes_info: Dict[str, Dict], 
                              attributes_to_set: Optional[List[str]] = None,
                              element_types: Optional[List[str]] = None) -> nx.Graph:
        """
        从关系列表和节点信息字典中，构建一个带属性的完整网络图。
        这是这个类的主要入口方法。

        Args:
            relationship_list: 包含元组的列表，每个元组代表一条边 (u, v)。
            nodes_info: 字典，键是节点ID，值是包含该节点所有属性的字典。
            attributes_to_set: 一个字符串列表，指定需要从 nodes_info 中提取并设置到图上的属性。

        Returns:
            一个配置完成的 networkx.Graph 对象。
        """
        if element_types is not None:
            filtered_nodes_info = {}
            for node_id, info in nodes_info.items():
                node_type = info.get('type')
                if node_type is None:
                    node_type = node_id.split(' ')[0]
                if node_type in element_types:
                    filtered_nodes_info[node_id] = info
            nodes_info = filtered_nodes_info
            
            valid_nodes = set(nodes_info.keys())
            relationship_list = [rel for rel in relationship_list if rel[0] in valid_nodes and rel[1] in valid_nodes]

        G = nx.Graph()

        # Step 1: 添加所有节点（从 nodes_info 的键中获取）
        G.add_nodes_from(nodes_info.keys())
        
        # Step 2: 使用辅助方法建立边和边的'index'属性
        self._build_edges_with_index(G, relationship_list)

        # Step 3: 使用辅助方法设置节点的属性
        self._set_node_attributes(G, nodes_info, attributes_to_set)
        
        return G
    
    def build_from_qs_file(self, qs_file_path: str, element_types: Optional[List[str]] = None) -> nx.Graph:
        """
        从 QS 文件创建网络图。
        """
        from src.parsers.qs_to_pandapower_converter import QsToPandapowerConverter
    
        with open(qs_file_path, "r") as f:
            data_dict = json.load(f)
        pp_dict, rel, nodes_info = QsToPandapowerConverter.create_pp_dict_with_network_from_qs_dict(data_dict)
        return self.build_from_components(rel, nodes_info, ['name', 'type', 'id'], element_types=element_types)

    def build_from_data_dict(self, data_dict: Dict[str, Any], element_types: Optional[List[str]] = None) -> nx.Graph:
        '''
        从一个完整的符合pandapower格式的数据字典中构建网络图。
        '''
        nodes_info = dict()
        relationship = list()
        
        target_types = self.available_types if element_types is None else element_types
        
        for key, values in data_dict.items():
            if key not in target_types:
                continue
            for value in values:
                label = f'{key} {value["id"]}'
                # 确保 value 是一个字典副本，以免修改原始数据
                nodes_info[label] = value.copy()
                nodes_info[label]['type'] = key
                rels = list()
                if key == 'line':
                    from_label = f'bus {value["from_bus"]}'
                    to_label = f'bus {value["to_bus"]}'
                    rels = [(from_label, label), (label, to_label)]
                elif key == 'switch':
                    from_label = f'bus {value["bus"]}'
                    if value["et"] == 'l':
                        to_label = f'line {value["element"]}'
                    elif value["et"] == 't':
                        to_label = f'trafo {value["element"]}'
                    elif value["et"] == '3w':
                        to_label = f'trafo3w {value["element"]}'
                    elif value["et"] == 'b':
                        to_label = f'bus {value["element"]}'
                    rels = [(from_label, label), (label, to_label)]
                elif key == 'trafo':
                    hv_label = f'bus {value["hv_bus"]}'
                    lv_label = f'bus {value["lv_bus"]}'
                    rels = [(hv_label, label), (label, lv_label)]
                elif key == 'trafo3w':
                    hv_label = f'bus {value["hv_bus"]}'
                    mv_label = f'bus {value["mv_bus"]}'
                    lv_label = f'bus {value["lv_bus"]}'
                    rels = [(hv_label, label), (label, mv_label), (label, lv_label)]
                # Single line elements
                elif key == 'shunt':
                    rels = [(f'bus {value["bus"]}', label)]
                elif key == 'gen':
                    rels = [(f'bus {value["bus"]}', label)]
                elif key == 'load':
                    rels = [(f'bus {value["bus"]}', label)]
                elif key == 'ext_grid':
                    rels = [(f'bus {value["bus"]}', label)]
                elif key == 'sgen':
                    rels = [(f'bus {value["bus"]}', label)]
                relationship.extend(rels)

        # 要删掉形成的环路,这个单纯是用来构建拓扑图的
        if 'switch' in target_types and 'switch' in data_dict:
            for switch in data_dict['switch']:
                from_label = f'bus {switch["bus"]}'
                if switch["et"] == 'l':
                    to_label = f'line {switch["element"]}'
                elif switch["et"] == 't':
                    to_label = f'trafo {switch["element"]}'
                elif switch["et"] == '3w':
                    to_label = f'trafo3w {switch["element"]}'
                elif switch["et"] == 'b':
                    to_label = f'bus {switch["element"]}'
                else:
                    continue
                rels = [(from_label, to_label)]
                if (from_label, to_label) in relationship:
                    relationship.remove((from_label, to_label))
                if (to_label, from_label) in relationship:
                    relationship.remove((to_label, from_label))

        # 删掉重复的边
        relationship = list(set(relationship))
        return self.build_from_components(relationship, nodes_info, None)

    def build_from_data_frame(self, 
                             network_data: Dict[str, pd.DataFrame], 
                             powerflow_result: Dict[str, pd.DataFrame],
                             element_types: Optional[List[str]] = None) -> nx.Graph:
        """
        从 Pandapower 格式的网络拓扑数据和潮流求解结果创建网络图。
        
        Args:
            network_data: Dict[str, pd.DataFrame]: 原始网络拓扑数据，包含bus, line, trafo等组件的连接信息
            powerflow_result: Dict[str, pd.DataFrame]: 潮流求解结果，包含vm_pu, va_degree, p_mw, q_mvar等计算结果
        
        Returns:
            nx.Graph: 包含拓扑结构和潮流计算结果的网络图
        """
        nodes_info = dict()
        relationship = list()
        
        target_types = self.available_types if element_types is None else element_types
        
        # 首先从原始网络拓扑数据构建基本拓扑结构
        for key, df in network_data.items():
            # 检查 df 是否为 DataFrame
            if not isinstance(df, pd.DataFrame):
                print(f"警告：{key} 不是 DataFrame，跳过处理")
                continue
                
            if df.empty:
                continue
            
            # 只处理 target_types 中定义的类型
            if key not in target_types:
                continue
                
            for idx, row in df.iterrows():
                label = f'{key} {idx}'
                
                # 创建节点信息，包含原始拓扑信息
                node_info = row.to_dict()
                node_info['id'] = idx
                node_info['type'] = key
                
                # 添加潮流计算结果（如果存在）
                if key in powerflow_result and not powerflow_result[key].empty:
                    if idx in powerflow_result[key].index:
                        result_row = powerflow_result[key].loc[idx]
                        for col in result_row.index:
                            node_info[col] = result_row[col]
                
                nodes_info[label] = node_info
                
                # 建立连接关系
                rels = []
                if key == 'line':
                    from_label = f'bus {row["from_bus"]}'
                    to_label = f'bus {row["to_bus"]}'
                    rels = [(from_label, label), (label, to_label)]
                elif key == 'switch':
                    from_label = f'bus {row["bus"]}'
                    if row["et"] == 'l':
                        to_label = f'line {row["element"]}'
                    elif row["et"] == 't':
                        to_label = f'trafo {row["element"]}'
                    elif row["et"] == '3w':
                        to_label = f'trafo3w {row["element"]}'
                    elif row["et"] == 'b':
                        to_label = f'bus {row["element"]}'
                    rels = [(from_label, label), (label, to_label)]
                elif key == 'trafo':
                    hv_label = f'bus {row["hv_bus"]}'
                    lv_label = f'bus {row["lv_bus"]}'
                    rels = [(hv_label, label), (label, lv_label)]
                elif key == 'trafo3w':
                    hv_label = f'bus {row["hv_bus"]}'
                    mv_label = f'bus {row["mv_bus"]}'
                    lv_label = f'bus {row["lv_bus"]}'
                    rels = [(hv_label, label), (label, mv_label), (label, lv_label)]
                # Single line elements
                elif key in ['shunt', 'gen', 'load', 'ext_grid', 'sgen']:
                    rels = [(f'bus {row["bus"]}', label)]
                
                relationship.extend(rels)
        
        # 处理开关，删除形成的环路（与build_from_data_dict中的逻辑相同）
        if 'switch' in target_types and 'switch' in network_data:
            for idx, switch in network_data['switch'].iterrows():
                from_label = f'bus {switch["bus"]}'
                if switch["et"] == 'l':
                    to_label = f'line {switch["element"]}'
                elif switch["et"] == 't':
                    to_label = f'trafo {switch["element"]}'
                elif switch["et"] == '3w':
                    to_label = f'trafo3w {switch["element"]}'
                elif switch["et"] == 'b':
                    to_label = f'bus {switch["element"]}'
                
                if (from_label, to_label) in relationship:
                    relationship.remove((from_label, to_label))
                if (to_label, from_label) in relationship:
                    relationship.remove((to_label, from_label))
        
        # 删除重复的边
        relationship = list(set(relationship))
        return self.build_from_components(relationship, nodes_info, None)
        