# grid/topology.py

import copy
import json
import logging
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict, deque
import pandas as pd

from .builder import GridBuilder
TIE_SWITCH_ID_START = 100000
class GridTopology:
    """
    电网拓扑结构类，包含网络拓扑图，节点电气信息，以及一些电网拓扑的操作与分析。
    """
    def __init__(self, graph=None, logger=None):
        """
        初始化分析器。
        如果提供了 graph 对象，则使用它。
        如果没有提供，则创建一个新的空图。
        """
        self.logger = logger
        if graph is None:
            self.graph = nx.Graph()
            if self.logger:
                self.logger.debug("未提供图，已创建一个新的空图。")
        elif isinstance(graph, nx.Graph):
            self.graph = graph
            if self.logger:
                self.logger.debug(f'The network has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.')
                self.logger.debug("已使用提供的图进行初始化。")
        else:
            raise TypeError("Input must be a networkx.Graph object or None.")

    def __deepcopy__(self, memo):
        """
        自定义深拷贝方法，跳过不可序列化的logger对象。
        
        Args:
            memo: deepcopy的备忘录字典
            
        Returns:
            GridTopology: 深拷贝的新实例
        """
        # 创建新实例，不包含logger（logger不能被深拷贝）
        cls = self.__class__
        new_topology = cls.__new__(cls)
        
        # 添加到memo避免循环引用
        memo[id(self)] = new_topology
        
        # 深拷贝graph
        new_topology.graph = copy.deepcopy(self.graph, memo)
        
        # logger保持引用，不深拷贝（因为logger应该在不同的topology实例间共享）
        new_topology.logger = self.logger
        
        return new_topology

    @classmethod
    def from_components(cls, 
                        relationship_list: List[Tuple[str, str]], 
                        nodes_info: Dict[str, Dict], 
                        attributes_to_set: Optional[List[str]] = None,
                        element_types: Optional[List[str]] = None) -> 'GridTopology':
        """
        工厂类方法，从原始数据便捷地创建实例。
        方法名改为 from_data 以更好地反映输入。
        """
        builder = GridBuilder()
        grid_graph = builder.build_from_components(relationship_list, nodes_info, attributes_to_set, element_types=element_types)
        return cls(grid_graph)

    @classmethod
    def from_qs_file(cls, qs_file_path: str, element_types: Optional[List[str]] = None) -> 'GridTopology':
        """
        从 QS 文件创建拓扑图。
        """
        buidler = GridBuilder()
        grid_graph = buidler.build_from_qs_file(qs_file_path, element_types=element_types)
        return cls(grid_graph)
    
    
    @classmethod
    def from_data_dict(cls, data_dict: Dict[str, Any], logger=None, element_types: Optional[List[str]] = None) -> 'GridTopology':
        """
        从 Pandapower 格式的数据字典创建拓扑图。
        """
        buidler = GridBuilder(logger=logger)
        grid_graph = buidler.build_from_data_dict(data_dict, element_types=element_types)
        return cls(grid_graph, logger=logger)

    @classmethod
    def from_data_frame(cls, 
                       network_data: Dict[str, pd.DataFrame], 
                       powerflow_result: Dict[str, pd.DataFrame],
                       element_types: Optional[List[str]] = None) -> 'GridTopology':
        """
        从 Pandapower 格式的网络拓扑数据和潮流求解结果创建拓扑图。
        
        Args:
            network_data: Dict[str, pd.DataFrame]: 原始网络拓扑数据，包含bus, line, trafo等组件的连接信息
            powerflow_result: Dict[str, pd.DataFrame]: 潮流求解结果，包含vm_pu, va_degree, p_mw, q_mvar等计算结果
        
        Returns:
            GridTopology: 包含拓扑结构和潮流计算结果的拓扑图对象
        """
        buidler = GridBuilder()
        grid_graph = buidler.build_from_data_frame(network_data, powerflow_result, element_types=element_types)
        return cls(grid_graph)
    
    @classmethod
    def from_pandapower_net(cls, net, powerflow_result: Dict[str, pd.DataFrame], element_types: Optional[List[str]] = None) -> 'GridTopology':
        """
        从 pandapower 网络对象和潮流求解结果创建拓扑图。
        
        Args:
            net: pandapower 网络对象
            powerflow_result: Dict[str, pd.DataFrame]: 潮流求解结果
        
        Returns:
            GridTopology: 包含拓扑结构和潮流计算结果的拓扑图对象
        """
        from ..solver.pandapower_solver import GridPandapowerSolver
        
        solver = GridPandapowerSolver()
        network_data = solver.get_network_status(net)
        
        buidler = GridBuilder()
        grid_graph = buidler.build_from_data_frame(network_data, powerflow_result, element_types=element_types)
        return cls(grid_graph)
    
    # ------------------------------------------------------------
    # 电网属性更新
    # ------------------------------------------------------------
    def update_from_powerflow_result(self, powerflow_result: Dict[str, pd.DataFrame]) -> None:
        """
        根据潮流求解结果更新图中节点的属性。
        
        Args:
            powerflow_result: Dict[str, pd.DataFrame]: 潮流求解结果，包含vm_pu, va_degree, p_mw, q_mvar等计算结果
        """
        for element_type, df in powerflow_result.items():
            if df.empty:
                continue
                
            for idx, row in df.iterrows():
                node_label = f'{element_type} {idx}'
                
                # 检查节点是否存在
                if node_label in self.graph.nodes:
                    # 更新节点属性
                    for col, value in row.items():
                        self.graph.nodes[node_label][col] = value
                    print(f"已更新节点 {node_label} 的潮流计算结果")
                else:
                    print(f"警告：节点 {node_label} 不存在于图中，跳过更新")
    
    def update_from_network_data(self, network_data: Dict[str, pd.DataFrame]) -> None:
        """
        根据网络拓扑数据更新图中节点的属性。
        
        Args:
            network_data: Dict[str, pd.DataFrame]: 原始网络拓扑数据
        """
        for element_type, df in network_data.items():
            if df.empty:
                continue
                
            for idx, row in df.iterrows():
                node_label = f'{element_type} {idx}'
                
                # 检查节点是否存在
                if node_label in self.graph.nodes:
                    # 更新节点属性
                    for col, value in row.items():
                        self.graph.nodes[node_label][col] = value
                    print(f"已更新节点 {node_label} 的网络拓扑数据")
                else:
                    print(f"警告：节点 {node_label} 不存在于图中，跳过更新")
    
    def update_from_pandapower_net(self, net) -> None:
        """
        从 pandapower 网络对象更新图的属性。
        这是一个便捷方法，会自动获取网络拓扑数据和潮流计算结果。
        
        Args:
            net: pandapower 网络对象
        """
        from src.solver.pandapower_solver import GridPandapowerSolver
        
        solver = GridPandapowerSolver()
        
        # 获取网络拓扑数据
        network_data = solver.get_network_status(net)
        
        # 获取潮流计算结果
        powerflow_result = solver.get_powerflow_result(net)
        
        # 更新网络拓扑数据
        self.update_from_network_data(network_data)
        
        # 更新潮流计算结果
        self.update_from_powerflow_result(powerflow_result)
        
        print("已从 pandapower 网络对象更新图属性")

    # ------------------------------------------------------------
    # 电网信息获取
    # ------------------------------------------------------------
    def get_nodes_count(self) -> int:
        """返回图中节点的数量（这里假设所有节点都是母线）。"""
        return self.graph.number_of_nodes()

    def get_edges_count(self) -> int:
        """返回图中边的数量。"""
        return self.graph.number_of_edges()

    def get_connected_components(self) -> List[set]:
        """返回图中的连通分量。"""
        return list(nx.connected_components(self.graph))

    def get_type_count(self) -> Dict[str, int]:
        """
        获取图中各类节点各自的数量
        """
        type_dict = defaultdict(int)
        for node, infos in self.graph.nodes(data=True):
            type_dict[infos['type']] += 1
        return type_dict
    
    # ------------------------------------------------------------
    # 电网拓扑分析与操作
    # ------------------------------------------------------------
    def _is_electrical_edge(self, node) -> bool:
        """
        判断两个节点之间的边是否为电气边（line、trafo、trafo3w）。
        
        Args:
            node: 节点
            
        Returns:
            bool: 如果是电气边返回True，否则返回False
        """
        # 获取边的连接类型
        node_data = self.graph.nodes[node]
        node_type = node_data['type']
        electrical_types = {'line', 'trafo', 'trafo3w', 'switch'}  # 25-9-20 新增switch
        # electrical_types = {'line', 'trafo', 'trafo3w'}
        return node_type in electrical_types

    def get_electrical_distance(self, source: str, target: str) -> int:
        """
        计算两个节点之间的电气距离。
        电气距离：只有在经过type为line、trafo、trafo3w的节点时才加1距离。
        
        Args:
            source: 源节点
            target: 目标节点
            
        Returns:
            int: 电气距离，如果不可达返回-1
        """
        if source == target:
            return 0
            
        # 使用BFS计算电气距离，找到目标节点就立即返回
        from collections import deque
        
        queue = deque([(source, 0)])  # (节点, 电气距离)
        visited = {source}
        
        while queue:
            current, dist = queue.popleft()
            
            if current == target:
                return dist
                
            # 遍历邻居节点
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 判断是否为电气边
                    if self._is_electrical_edge(neighbor):
                        queue.append((neighbor, dist + 1))
                    else:
                        queue.append((neighbor, dist))  # 非电气边不增加距离
        
        return -1  # 不可达

    def check_in_service(self, nodes: Union[str, List[str]]) -> dict:
        """
        检查输入节点的in_service属性是否为True。
        输入可以是单个节点（str）或节点列表（List[str]），返回每个节点的in_service状态字典。

        Args:
            nodes str | List[str]: 节点名或节点名列表

        Returns:
            flag bool: 是否存在非在运节点
            result dict: {节点名: in_service状态（bool或None）}
        """
        if isinstance(nodes, str):
            nodes = [nodes]
        result = {}
        flag = False
        for node in nodes:
            node_data = self.graph.nodes[node]
            in_service_status = node_data.get('in_service', None)
            if in_service_status in (False, 0):
                result[node] = False
                flag = True
            else:
                result[node] = True
            # print(f"Node {node} in_service: {result[node]}")
        return flag, result

    def search_nodes_by_keyword(self, keyword: str, attribute: str) -> List[str]:
        """
        搜索包含关键词的节点。
        
        Args:
            keyword: 搜索关键词
            attribute: 搜索的属性名称
            
        Returns:
            List[str]: 包含关键词的节点列表
        """
        matching_nodes = []
        
        for node, data in self.graph.nodes(data=True):
            if attribute in data:
                attribute_value = str(data[attribute])
                if keyword.lower() in attribute_value.lower():
                    # print(keyword, attribute_value)
                    matching_nodes.append(node)

        return matching_nodes

    def _multi_source_electrical_bfs(self, source_nodes: List[str], max_depth: int = None, allow_non_in_service: bool = False) -> Tuple[set, Dict[str, int]]:
        """
        多源电气距离BFS搜索，返回可达节点集合和节点深度字典。
        
        Args:
            source_nodes: 源节点列表
            max_depth: 最大电气距离，为None时表示不限制
            
        Returns:
            Tuple[set, Dict[str, int]]: (可达节点集合, 节点深度字典)
        """
        from collections import deque
        
        visited = set(source_nodes)
        queue = deque([(node, 0) for node in source_nodes])  # (节点, 电气距离)
        nodes_reachable = set(source_nodes)
        node_depth = {node: 0 for node in source_nodes}  # 记录每个节点的最小电气距离
        
        while queue:
            current, distance = queue.popleft()

            # 如果设置了最大深度且已到达最大深度，则不再向外扩展
            # if max_depth is not None and distance > max_depth:
            #     continue
            # 达到了最大深度，但是不可用作边界节点的（无pq或无vmva），继续bfs
            if max_depth is not None and distance > max_depth:
                # 检查当前节点是否具有边界节点属性
                node_data = self.graph.nodes[current]
                has_vmva = 'vm_pu' in node_data and 'va_degree' in node_data
                has_pq = 'p_mw' in node_data and 'q_mvar' in node_data
                
                # 如果节点有边界节点属性，则停止从该节点继续搜索
                if has_vmva or has_pq:
                    continue

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    # 如果设置不允许非在运节点，则跳过非在运节点
                    if not allow_non_in_service:
                        if (self.check_in_service(neighbor)[0] 
                            and self.graph.nodes[neighbor]['type'] not in ['load', 'shunt', 'ext_grid', 'sgen', 'gen']):
                            continue
                    # 计算电气距离
                    if self._is_electrical_edge(neighbor):
                        new_distance = distance + 1
                    else:
                        new_distance = distance
                    
                    # 如果超出最大深度，检查是否具有边界节点属性
                    if max_depth is not None and new_distance > max_depth:
                        neighbor_data = self.graph.nodes[neighbor]
                        has_vmva = 'vm_pu' in neighbor_data and 'va_degree' in neighbor_data
                        has_pq = 'p_mw' in neighbor_data and 'q_mvar' in neighbor_data
                        
                        # 如果邻居节点有边界节点属性，则加入但不再继续搜索
                        if has_vmva or has_pq:
                            visited.add(neighbor)
                            nodes_reachable.add(neighbor)
                            node_depth[neighbor] = new_distance
                            print(neighbor, 'has_vmva or has_pq')
                            continue
                        # 如果邻居节点没有边界节点属性，则跳过
                        # else:
                        #     continue
                    
                    visited.add(neighbor)
                    nodes_reachable.add(neighbor)
                    node_depth[neighbor] = new_distance
                    queue.append((neighbor, new_distance))
        
        return nodes_reachable, node_depth

    def extract_boundary_nodes(self, nodes_reachable: set) -> List[str]:
        """
        从可达节点集合中提取边界节点。
        边界节点：在最大深度上，且有邻居在子图外。
        
        Args:
            nodes_reachable: 可达节点集合
            # (应该不需要node_depth: 节点深度字典)
            # (max_depth: 最大深度)
            
        Returns:
            List[str]: 边界节点列表
        """
        boundary_nodes = []
        for node in nodes_reachable:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in nodes_reachable:
                    boundary_nodes.append(node)
                    break
        return boundary_nodes

    def multi_source_electrical_bfs(
        self, 
        source_nodes: List[str], 
        max_depth: int = None
    ) -> Tuple['GridTopology', List[str]]:
        """
        多源电气距离BFS搜索，可指定最大电气距离，并返回边界节点。

        Args:
            source_nodes List[str]: 源节点名称列表
            max_depth int, optional: 最大电气距离（深度），为None时表示不限制，遍历所有可达节点

        Returns:
            subgraph GridTopology: 包含所有可达节点的子图
            boundary_nodes List[str]: 边界节点列表（在最大距离上，且连接子图外节点）
        """
        # 执行多源BFS
        nodes_reachable, node_depth = self._multi_source_electrical_bfs(source_nodes, max_depth)
        
        # 创建子图
        subgraph = self.graph.subgraph(nodes_reachable).copy()
        
        # 提取边界节点
        boundary_nodes = []
        if max_depth is not None:
            boundary_nodes = self.extract_boundary_nodes(nodes_reachable)
        
        return GridTopology(subgraph), boundary_nodes

    def search_and_bfs_by_keyword(self, keyword: str, max_depth: int = None) -> Tuple['GridTopology', List[str]]:
        """
        通过关键词搜索节点，并对这些节点进行多源电气距离BFS搜索。
        
        Args:
            keyword str: 搜索关键词，会在节点的name属性中搜索
            max_depth int, optional: 最大电气距离（深度），为None时表示不限制，遍历所有可达节点
            
        Returns:
            subgraph GridTopology: 包含所有可达节点的子图
            boundary_nodes List[str]: 边界节点列表（在最大距离上，且连接子图外节点）
        """
        # 搜索包含关键词的节点
        source_nodes = self.search_nodes_by_keyword(keyword, 'name')
        
        if not source_nodes:
            print(f"未找到包含关键词 '{keyword}' 的节点")
            return GridTopology(nx.Graph()), []
        
        print(f"找到 {len(source_nodes)} 个包含关键词 '{keyword}' 的节点")
        
        # 执行多源BFS
        return self.multi_source_electrical_bfs(source_nodes, max_depth)
    
    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """
        获取两个节点间的最短路径（基于图的最短路径，不区分电气/非电气边）。

        Args:
            source str: 起始节点名称
            target str: 目标节点名称

        Returns:
            path List[str]: 最短路径上的节点名称列表（包含起点和终点）。如果不可达，返回空列表。
        """
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            print(f"节点 {source} 和 {target} 之间不存在路径。")
            return []
        except nx.NodeNotFound as e:
            print(f"节点未找到: {e}")
            return []

    def get_adjacent_nodes(self, node: str) -> List[str]:
        """
        获取节点的相邻节点。
        """
        return list(self.graph.neighbors(node))

    def _multi_source_topology_bfs(self, source_nodes: List[str], max_depth: int = None) -> Tuple[set, Dict[str, int]]:
        """
        多源拓扑距离BFS搜索，返回可达节点集合和节点深度字典。
        
        Args:
            source_nodes: 源节点列表
            max_depth: 最大拓扑距离，为None时表示不限制
            
        Returns:
            Tuple[set, Dict[str, int]]: (可达节点集合, 节点深度字典)
        """
        from collections import deque
        
        visited = set(source_nodes)
        queue = deque([(node, 0) for node in source_nodes])  # (节点, 拓扑距离)
        nodes_reachable = set(source_nodes)
        node_depth = {node: 0 for node in source_nodes}  # 记录每个节点的最小拓扑距离
        
        while queue:
            current, distance = queue.popleft()

            # 如果设置了最大深度且已到达最大深度，则不再向外扩展
            if max_depth is not None and distance >= max_depth:
                continue
                
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    new_distance = distance + 1
                    
                    # 如果超出最大深度则不加入
                    if max_depth is not None and new_distance > max_depth:
                        continue
                    
                    visited.add(neighbor)
                    nodes_reachable.add(neighbor)
                    node_depth[neighbor] = new_distance
                    queue.append((neighbor, new_distance))
        
        return nodes_reachable, node_depth

    def multi_source_topology_bfs(
        self, 
        source_nodes: List[str], 
        max_depth: int = None
    ) -> Tuple['GridTopology', List[str]]:
        """
        多源拓扑距离BFS搜索，可指定最大拓扑距离，并返回边界节点。

        Args:
            source_nodes List[str]: 源节点名称列表
            max_depth int, optional: 最大拓扑距离（深度），为None时表示不限制，遍历所有可达节点

        Returns:
            subgraph GridTopology: 包含所有可达节点的子图
            boundary_nodes List[str]: 边界节点列表（在最大距离上，且连接子图外节点）
        """
        # 执行多源BFS
        nodes_reachable, node_depth = self._multi_source_topology_bfs(source_nodes, max_depth)
        
        # 创建子图
        subgraph = self.graph.subgraph(nodes_reachable).copy()
        
        # 提取边界节点
        boundary_nodes = []
        if max_depth is not None:
            boundary_nodes = self.extract_boundary_nodes(nodes_reachable)
        
        return GridTopology(subgraph), boundary_nodes

    def search_and_topology_bfs_by_keyword(self, keyword: str, max_depth: int = None) -> Tuple['GridTopology', List[str]]:
        """
        通过关键词搜索节点，并对这些节点进行多源拓扑距离BFS搜索。
        
        Args:
            keyword str: 搜索关键词，会在节点的name属性中搜索
            max_depth int, optional: 最大拓扑距离（深度），为None时表示不限制，遍历所有可达节点
            
        Returns:
            subgraph GridTopology: 包含所有可达节点的子图
            boundary_nodes List[str]: 边界节点列表（在最大距离上，且连接子图外节点）
        """
        # 搜索包含关键词的节点
        source_nodes = self.search_nodes_by_keyword(keyword, 'name')
        
        if not source_nodes:
            print(f"未找到包含关键词 '{keyword}' 的节点")
            return GridTopology(nx.Graph()), []
        
        print(f"找到 {len(source_nodes)} 个包含关键词 '{keyword}' 的节点")
        
        # 执行多源拓扑BFS
        return self.multi_source_topology_bfs(source_nodes, max_depth)

    def get_nearest_element_of_type(self, source_node: str, target_type: str, max_depth: int = None) -> Tuple[Optional[str], int]:
        """
        获取离源节点最近的指定类型的节点及其电气距离。
        
        Args:
            source_node: 源节点名称
            target_type: 目标节点类型 (e.g., 'trafo3w', 'bus')
            max_depth: 最大搜索深度（电气距离）
            
        Returns:
            Tuple[Optional[str], int]: (最近的节点名称, 电气距离)。如果未找到，返回 (None, -1)。
        """
        from collections import deque
        
        if source_node not in self.graph:
             return None, -1

        # 如果源节点本身就是目标类型
        if self.graph.nodes[source_node].get('type') == target_type:
            return source_node, 0
            
        queue = deque([(source_node, 0)])  # (节点, 电气距离)
        visited = {source_node}
        
        while queue:
            current, dist = queue.popleft()
            
            if max_depth is not None and dist > max_depth:
                continue

            # 检查当前节点是否为目标类型 (除起始节点外)
            if current != source_node and self.graph.nodes[current].get('type') == target_type:
                return current, dist
                
            # 遍历邻居节点
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 判断是否为电气边，决定距离是否增加
                    if self._is_electrical_edge(neighbor):
                        new_dist = dist + 1
                    else:
                        new_dist = dist
                    
                    # 优化：如果在加入队列前检查，可以更快返回（但要注意距离计算的准确性，这里BFS保证最短路径）
                    if self.graph.nodes[neighbor].get('type') == target_type:
                        return neighbor, new_dist
                    
                    queue.append((neighbor, new_dist))
        
        return None, -1

    def get_all_nodes_shortest_distance_to_type(self, target_type: str, nodes_subset: List[str] = None) -> Dict[str, Tuple[str, int]]:
        """
        计算（子）图中所有节点到最近的指定类型节点的最短电气距离。
        
        Args:
            target_type: 目标节点类型 (e.g., 'trafo3w')
            nodes_subset: 可选，限制计算范围的节点列表。如果为None，则计算全图。
            
        Returns:
            Dict[str, Tuple[str, int]]: {节点名: (最近的目标节点名, 电气距离)}
        """
        # 1. 找到所有的目标类型节点作为 BFS 的起点
        if nodes_subset:
             # 只在子集中找目标节点
             sources = [n for n in nodes_subset if self.graph.nodes[n].get('type') == target_type]
             valid_nodes = set(nodes_subset)
        else:
             sources = [n for n, node in self.graph.nodes(data=True) if node.get('type') == target_type]
             valid_nodes = None # 全图

        if not sources:
            return {}

        # 多源 BFS
        from collections import deque
        # queue 存储 (current_node, nearest_source, distance)
        # 注意：源节点到自己的距离是 0
        queue = deque([(s, s, 0) for s in sources])
        
        # 记录结果: node -> (nearest_source, distance)
        results = {s: (s, 0) for s in sources}
        
        while queue:
            current, source, dist = queue.popleft()
            
            for neighbor in self.graph.neighbors(current):
                if valid_nodes and neighbor not in valid_nodes:
                    continue
                
                # 计算新距离
                # 如果 neighbor 是电气设备，距离+1。否则+0。
                if self._is_electrical_edge(neighbor):
                    new_dist = dist + 1
                else:
                    new_dist = dist
                
                # 如果未访问过，或者找到了更近的路径
                if neighbor not in results or new_dist < results[neighbor][1]:
                     results[neighbor] = (source, new_dist)
                     queue.append((neighbor, source, new_dist))
                     
        return results

    def extract_minimal_powerflow_region(self, transformer_name: str, max_depth=1) -> Tuple[List[str], List[str]]:
        """
        提取主变周围可以复现其潮流的最小区域。
        理论上提取到的节点除了核心节点，应该只有switch和bus节点，边界节点应该只有bus节点。
        只有遇到vmva节点或者pq节点才停止，保证可以被等效
        
        通过搜索name属性找到主变节点，然后以该节点为源进行BFS搜索，对于每个节点：
        - 如果该节点有p_mw和q_mvar属性，或者有vm_pu和va_degree属性，则停止向外扩展
        - 否则继续向外扩展
        通过tag来标记节点是哪一侧的
        - s: source node
        - h: high voltage side
        - m: middle voltage side
        - l: low voltage side
        
        Args:
            transformer_name str: 主变节点的name属性值
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 节点信息字典
            Set[str]: 区域内节点集合
            Set[str]: 边界节点集合
        """
        from collections import deque
        
        # 通过name属性搜索主变节点
        source_nodes = self.search_nodes_by_keyword(transformer_name, 'name')
        
        if not source_nodes:
            logging.warning(f"警告：未找到name属性为 '{transformer_name}' 的节点")
            return {}, set(), set()
        
        # 如果找到多个匹配的节点，取第一个
        source_node = source_nodes[0]
        if len(source_nodes) > 1:
            logging.debug(f"找到 {len(source_nodes)} 个匹配的节点，使用第一个：{source_node}")
        
        # 初始化BFS
        visited = set()
        queue = deque([(source_node, 's', 0)])
        nodes_in_region = set()
        boundary_nodes = set()
        vmva_nodes = set()

        while queue:
            current, tag, depth = queue.popleft()
            
            # 如果已经访问过，跳过
            if current in visited:
                continue
                
            node_data = self.graph.nodes[current]
            
            # # 如果这个节点not in service，直接不要
            # if node_data.get('in_service', True) is False:
            #     continue
            
            visited.add(current)
            nodes_in_region.add(current)

            if tag == 's':
                for neighbor in self.graph.neighbors(current):
                    nb = self.graph.nodes[neighbor]['id']
                    if nb == node_data.get('hv_bus', None):
                        queue.append((neighbor, 'h', depth))
                    elif nb == node_data.get('mv_bus', None):
                        queue.append((neighbor, 'm', depth))
                    elif nb == node_data.get('lv_bus', None):
                        queue.append((neighbor, 'l', depth))
                    else:
                        raise ValueError(f'Unknown neighbor: {self.graph.nodes[neighbor]}')
                    # visited.add(neighbor)

            elif tag == 'h':  # 高压侧，找vmva节点  # 25-9-27 高压侧暂时只找一层, 也即发现就停止，后续可能要修改
                has_vmva = 'vm_pu' in node_data and 'va_degree' in node_data and node_data['type'] == 'bus'
                if has_vmva:
                    vmva_nodes.add(current)
                    boundary_nodes.add(current)
                    # print(f"节点 {current} 具有vmva边界属性，停止向外扩展")
                    continue
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, 'h', depth))
                        # visited.add(neighbor)

            elif tag == 'm':  # 中压侧，找pq节点
                has_pq = 'p_mw' in node_data and 'q_mvar' in node_data and node_data['type'] == 'bus'
                if has_pq:  # 找到pq就深度加一
                    if depth + 1 == max_depth:
                        boundary_nodes.add(current)
                        # print(f"节点 {current} 具有pq边界属性，停止向外扩展")
                        continue
                    for neighbor in self.graph.neighbors(current):  # 未达到深度就+1继续搜索
                        if neighbor not in visited:
                            queue.append((neighbor, 'm', depth + 1))
                else:
                    for neighbor in self.graph.neighbors(current):  # 未达到深度就+1继续搜索
                        if neighbor not in visited:
                            queue.append((neighbor, 'm', depth))

            elif tag == 'l':  # 低压侧，找pq节点
                has_pq = 'p_mw' in node_data and 'q_mvar' in node_data and node_data['type'] == 'bus'
                if has_pq:
                    if depth + 1 == max_depth:
                        boundary_nodes.add(current)
                        # print(f"节点 {current} 具有pq边界属性，停止向外扩展")
                        continue
                    for neighbor in self.graph.neighbors(current):
                        if neighbor not in visited:
                            queue.append((neighbor, 'l', depth + 1))
                else:
                    for neighbor in self.graph.neighbors(current):
                        if neighbor not in visited:
                            queue.append((neighbor, 'l', depth))
            else:
                raise ValueError(f'Unknown tag: {tag}')

        
        # 返回节点的完整信息dict
        result_nodes = set(nodes_in_region)
        nodes_info = defaultdict(list)
        for rn in result_nodes:
            type = self.graph.nodes[rn]['type']
            nodes_info[type].append(dict(self.graph.nodes[rn]))
        logging.debug(f"以 {transformer_name} 为源节点，提取到 {len(result_nodes)} 个节点的最小潮流区域，其中 {len(boundary_nodes)} 个边界节点")
        return nodes_info, result_nodes, boundary_nodes, vmva_nodes

    def save_to_json(self, file_path: str) -> None:
        """
        将图保存为JSON文件。这个后续还要改一下
        
        Args:
            file_path: 保存的文件路径
        """
        from .serializer import GridSerializer
        GridSerializer.to_json_file(self, file_path)

    # ------------------------------------------------------------
    # 配网合并操作
    # ------------------------------------------------------------
    def merge_with_other_distribution_topology(self, other_topology: 'GridTopology', keyword: str) -> 'GridTopology':
        """
        将两个拓扑图根据给定属性合并，并返回合并后的拓扑图。
        2025-8-9: 先暂时只考虑有个联络开关重合的情况，有多个重合节点的情况暂时没有考虑
        Args:
            other_topology: 另一个拓扑图
            keyword: 合并的属性

        Returns:
            GridTopology: 合并后的拓扑图
            int: 合并的节点数
        """
        # 遍历other_topology的节点，先给每个节点的index都加上当前已有的数量，防止重复
        current_type_count = self.get_type_count()
        new_switch_count = 0
        for n, node in other_topology.graph.nodes(data=True):
            type = node['type']
            node['id'] += current_type_count[type]
            if type == 'bus':
                continue
            elif type == 'load' or type == 'shunt' or type == 'ext_grid':
                node['bus'] += current_type_count['bus']
            elif type == 'gen':
                node['bus'] += current_type_count['bus']
                node['slack'] = False  # 因为只能有一个slack，所以这里需要设置为False
            elif type == 'line':
                node['from_bus'] += current_type_count['bus']
                node['to_bus'] += current_type_count['bus']
            elif type == 'trafo':
                node['hv_bus'] += current_type_count['bus']
                node['lv_bus'] += current_type_count['bus']
            elif type == 'trafo3w':
                node['hv_bus'] += current_type_count['bus']
                node['mv_bus'] += current_type_count['bus']
                node['lv_bus'] += current_type_count['bus']
            elif type == 'switch':
                node['bus'] += current_type_count['bus']
                if node['et'] == 'b':
                    node['element'] += current_type_count['bus']
                elif node['et'] == 'l':
                    node['element'] += current_type_count['line']
                elif node['et'] == 't':
                    node['element'] += current_type_count['trafo']
                elif node['et'] == 't3':
                    node['element'] += current_type_count['trafo3w']
                else:
                    raise ValueError(f'Unknown node type: {type}')
                new_switch_count += 1
            else:
                raise ValueError(f'Unknown node type: {type}')

        g1_relabel_mapping = {node: f'1-{node}' for node in self.graph.nodes}
        g1_key_dict = {d[keyword]: n for n, d in self.graph.nodes(data=True)}
        g2_relabel_mapping = {node: f'2-{node}' for node in other_topology.graph.nodes}
        g2_key_dict = {d[keyword]: n for n, d in other_topology.graph.nodes(data=True)}

        collapse_count = 0
        # merged_node_list = []
        tie_switch1, tie_switch2 = None, None
        merged_node_name = None
        # 相交的应该就是那个联络开关的位置，但是我们不把这两个合并了，另外再创建一个联络开关节点
        for k, v in g1_key_dict.items():
            if k in g2_key_dict:
                merged_node_name = f'{1}-{g1_key_dict[k]}~{2}-{g2_key_dict[k]}'
                # g1_relabel_mapping[v] = merged_node_name
                # g2_relabel_mapping[g2_key_dict[k]] = merged_node_name
                collapse_count += 1
                # merged_node_list.append(merged_node_name)
                tie_switch1 = self.graph.nodes[v]
                tie_switch2 = other_topology.graph.nodes[g2_key_dict[k]]
                break
        g1_final = nx.relabel_nodes(self.graph, g1_relabel_mapping, copy=True)
        g2_final = nx.relabel_nodes(other_topology.graph, g2_relabel_mapping, copy=True)
        g_final = nx.compose(g1_final, g2_final)
        
        # 最后再更新一下联络开关的属性，一侧是bus，一侧是element
        g_final.add_node(
            merged_node_name, 
            id = 200000,  # 联络开关从200000开始，目前只有一个，后续要改写一下
            type='switch', 
            bus=tie_switch1['bus'], 
            et='b',
            element=tie_switch2['bus'],
            closed=False,
            is_zdhkg=True,
        )
        g_final.add_edge(f'1-bus {tie_switch1["bus"]}', merged_node_name)
        g_final.add_edge(merged_node_name, f'2-bus {0}')
        # g_final.nodes[merged_node_name]['type'] = 'switch'
        # g_final.nodes[merged_node_name]['bus'] = tie_switch1['bus']
        # g_final.nodes[merged_node_name]['element'] = tie_switch2['element']

        return GridTopology(g_final), collapse_count, merged_node_name
    
    # ------------------------------------------------------------
    # 主网区域提取操作
    # ------------------------------------------------------------
    def get_sub_topology(self, nodes: List[str]) -> 'GridTopology':
        """
        获取子拓扑图。
        """
        subgraph = self.graph.subgraph(nodes).copy()
        return GridTopology(subgraph)
    
    def extract_sub_transmission_topology_on_path_between(self, feed1_keyword: str, feed2_keyword: str, attribute: str = 'name', search_depth: int = 0) -> 'GridTopology':
        """
        提取连接两个配网馈线的主网区域子拓扑图。
        Args:
            feed1_keyword str: 第一个馈线上的负荷节点关键词
            feed2_keyword str: 第二个馈线上的负荷节点关键词
            attribute str: 节点属性关键词
            search_depth int: 搜索深度

        Returns:
            Dict[str, Any]: 处理后的子拓扑图的Pandapower格式字典    
            GridTopology: 处理后的子拓扑图
            List[str]: 连接馈线的节点列表
        """
        # 先找到两个馈线上的负荷节点
        f1_nodes = self.search_nodes_by_keyword(feed1_keyword, attribute)
        f2_nodes = self.search_nodes_by_keyword(feed2_keyword, attribute)
        for node in f1_nodes:
            if 'load' in node:
                f1_node = node
                break
        for node in f2_nodes:
            if 'load' in node:
                f2_node = node
                break
        connection_nodes_trans = [f1_node, f2_node]
        path = self.get_shortest_path(f1_node, f2_node)  # 然后找到两个馈线上的负荷节点之间的最短路径 

        # 检查路径上是否存在非在运节点
        flag, in_service_status = self.check_in_service(path)
        for node in path:
            if not in_service_status[node]:
                print(f"Node {node} is not in service")
        if flag:
            print(in_service_status)
            raise Exception('非在运节点')

        sub_trans, boundary_nodes = self.multi_source_electrical_bfs(path, search_depth)  # 然后提取最短路径上的节点


        sub_trans_dict = sub_trans.to_pp_dict()
        sub_trans_dict, gen_id, sub_trans = self._handle_boundary_nodes(boundary_nodes, sub_trans_dict, sub_trans)  # 处理边界节点
        sub_trans_dict, sub_trans = self._handle_slack(sub_trans_dict, gen_id, sub_trans)  # 处理slack
        
        return sub_trans_dict, sub_trans, connection_nodes_trans

    def _handle_boundary_nodes(self, boundary_nodes: List[str], sub_trans_dict: Dict[str, Any], sub_trans: 'GridTopology') -> Dict[str, Any]:
        """
        处理边界节点，视情况将边界节点转换为ext_grid、gen、sgen、load。
        Args:
            boundary_nodes List[str]: 边界节点列表
            sub_trans_dict Dict[str, Any]: 子拓扑图的Pandapower格式字典

        Returns:
            Dict[str, Any]: 处理后的子拓扑图的Pandapower格式字典    
            int: gen数量
        """
        type_count = self.get_type_count()
        ext_grid_id = type_count.get('ext_grid', 0)
        gen_id = type_count.get('gen', 0)
        sgen_id = type_count.get('sgen', 0)
        load_id = type_count.get('load', 0)
        for bn in boundary_nodes:
            # 如果有Vm和Va，建模为ext_grid  
            if 'vm_pu' in self.graph.nodes[bn] and 'va_degree' in self.graph.nodes[bn]:
                sub_trans_dict['ext_grid'].append(dict(
                    name=f'{bn} boundary-ext_grid {ext_grid_id}',
                    bus=int(bn.split(' ')[1]),
                    vm_pu=self.graph.nodes[bn]['vm_pu'],
                    va_degree=self.graph.nodes[bn]['va_degree'],
                    in_service=self.graph.nodes[bn]['in_service'],
                    id=ext_grid_id,
                ))
                sub_trans.graph.add_node(
                    f'{bn} boundary-ext_grid {ext_grid_id}',
                    bus=int(bn.split(' ')[1]),
                    type='ext_grid',
                    vm_pu=self.graph.nodes[bn]['vm_pu'],
                    va_degree=self.graph.nodes[bn]['va_degree'],
                    in_service=self.graph.nodes[bn]['in_service'],
                    id=ext_grid_id,
                )
                sub_trans.graph.add_edge(bn, f'{bn} boundary-ext_grid {ext_grid_id}')
                ext_grid_id += 1
            # 如果有P和Vm，建模为gen
            # elif 'p_mw' in self.graph.nodes[bn] and 'vm_pu' in self.graph.nodes[bn]:
            #     print(self.graph.nodes[bn])  # 25-9-25: 打印一下这个节点
            #     sub_trans_dict['gen'].append(dict(
            #         name=f'{bn} boundary-gen {gen_id}',
            #         bus=int(bn.split(' ')[1]),
            #         p_mw=self.graph.nodes[bn]['p_mw'],
            #         vm_pu=self.graph.nodes[bn]['vm_pu'],
            #         # q_mvar=self.graph.nodes[bn]['q_mvar'],
            #         in_service=self.graph.nodes[bn]['in_service'],
            #         is_boundary=True,
            #         id=gen_id,
            #     ))
            #     sub_trans.graph.add_node(
            #         f'{bn} boundary-gen {gen_id}',
            #         bus=int(bn.split(' ')[1]),
            #         type='gen',
            #         p_mw=self.graph.nodes[bn]['p_mw'],
            #         vm_pu=self.graph.nodes[bn]['vm_pu'],
            #         in_service=self.graph.nodes[bn]['in_service'],
            #         id=gen_id,
            #     )
            #     sub_trans.graph.add_edge(bn, f'{bn} boundary-gen {gen_id}')
            #     gen_id += 1
            else:  # 25-9-26 就考虑建模为load或者ext_grid
                # 如果该node为switch，说明这个switch是失效的，直接删掉得了  # 25-9-25: 直接删掉肯定就不准了呀！
                if self.graph.nodes[bn]['type'] in ['switch', 'line', 'trafo', 'trafo3w']:
                    # 删除bn这个节点
                    print(f'not available node {bn}')
                    sub_trans.graph.remove_node(bn)
                    continue
                # print(self.graph.nodes[bn])
                # 如果没有p_mw，说明这个至少是个丁字形结构  # 25-8-26：暂时先砍掉，之后用更高级的算法来处理  
                if 'p_mw' not in self.graph.nodes[bn]:
                    print(f'T-shape node {bn}')
                    continue

                # 如果P为正，说明在发出功率，建模为sgen  # 25-9-24: 暂时都建模为load
                # if self.graph.nodes[bn]['p_mw'] > 0:
                #     sub_trans_dict['sgen'].append(dict(
                #         name=f'{bn} boundary-sgen {sgen_id}',
                #         bus=int(bn.split(' ')[1]),
                #         p_mw=self.graph.nodes[bn]['p_mw'],
                #         q_mvar=self.graph.nodes[bn]['q_mvar'],
                #         slack=False,
                #         in_service=self.graph.nodes[bn]['in_service'],
                #         is_boundary=True,
                #         id=sgen_id,
                #     ))  
                #     sub_trans.graph.add_node(
                #         f'{bn} boundary-sgen {sgen_id}',
                #         bus=int(bn.split(' ')[1]),
                #         type='sgen',
                #         p_mw=self.graph.nodes[bn]['p_mw'],
                #         q_mvar=self.graph.nodes[bn]['q_mvar'],
                #         in_service=self.graph.nodes[bn]['in_service'],
                #         id=sgen_id,
                #     )
                #     sub_trans.graph.add_edge(bn, f'{bn} boundary-sgen {sgen_id}')
                #     sgen_id += 1
                #     print(bn, 'sgen')
                # 如果P为负，说明在吸收功率，建模为load  # 25-9-24: 暂时都建模为load
                # else:
                sub_trans_dict['load'].append(dict(
                    name=f'{bn} boundary-load {load_id}',
                    bus=int(bn.split(' ')[1]),
                    p_mw=-self.graph.nodes[bn]['p_mw'],
                    q_mvar=-self.graph.nodes[bn]['q_mvar'],
                    in_service=self.graph.nodes[bn]['in_service'],
                    is_boundary=True,
                    id=load_id,
                ))
                sub_trans.graph.add_node(
                    f'{bn} boundary-load {load_id}',
                    bus=int(bn.split(' ')[1]),
                    type='load',
                    p_mw=-self.graph.nodes[bn]['p_mw'],
                    q_mvar=-self.graph.nodes[bn]['q_mvar'],
                    in_service=self.graph.nodes[bn]['in_service'],
                    id=load_id,
                )
                sub_trans.graph.add_edge(bn, f'{bn} boundary-load {load_id}')
                load_id += 1
                print(bn, 'load')
        return sub_trans_dict, gen_id, sub_trans
    
    def _handle_slack(self, data_dict, gen_id, sub_trans: 'GridTopology'):
        """
        处理slack，选择pS最大的gen作为slack。
        """
    # 先把所有的gen的slack都重置为False
        for gen in data_dict['gen']:
            gen['slack'] = False
        nodes = [n for n, a in sub_trans.graph.nodes(data=True) if 'slack' in a]
        nx.set_node_attributes(sub_trans.graph, {n: False for n in nodes}, 'slack')

        # 如果有ext_grid,ext_grid就能提供slack,后续看情况添加ext的均分
        if 'ext_grid' in data_dict and len(data_dict['ext_grid']) > 0:
            return data_dict, sub_trans
        # 如果没有ext_grid,但是有gen,选择p_mw最大的gen作为slack
        if 'gen' in data_dict and len(data_dict['gen']) > 0:
            max_p_mw = max(gen['p_mw'] for gen in data_dict['gen'])
            for gen in data_dict['gen']:
                if gen['p_mw'] == max_p_mw:
                    gen['slack'] = True
                    break
            for n, a in sub_trans.graph.nodes(data=True):
                if a['type'] == 'gen' and a['p_mw'] == max_p_mw:
                    a['slack'] = True
                    break
            return data_dict, sub_trans
        # 如果没有ext_grid,也没有gen,但是有sgen,选择s最大的转化成slack
        if 'sgen' in data_dict and len(data_dict['sgen']) > 0:
            max_s_va = -99999
            chosen_sgen = None
            for sgen in data_dict['sgen']:
                s_va_square = sgen['p_mw']**2 + sgen['q_mvar']**2
                if s_va_square > max_s_va:
                    max_s_va = s_va_square
                    chosen_sgen = sgen
            data_dict['gen'].append(dict(
                name=f'{chosen_sgen["name"]} boundary-gen {gen_id}',
                bus=chosen_sgen['bus'],
                p_mw=chosen_sgen['p_mw'],
                vm_pu=1.0,  # 这里先设置为1.0，后续再根据实际情况调整
                # q_mvar=chosen_sgen['q_mvar'],
                slack=True,
                in_service=chosen_sgen['in_service'],
                id=gen_id,
            ))
            sub_trans.graph.add_node(
                f'bus {chosen_sgen["bus"]} boundary-gen {gen_id}',
                bus=chosen_sgen['bus'],
                type='gen',
                p_mw=chosen_sgen['p_mw'],
                vm_pu=1.0,
                in_service=chosen_sgen['in_service'],
                slack=True,
                id=gen_id,
            )
            sub_trans.graph.add_edge(f'bus {chosen_sgen["bus"]}', f'bus {chosen_sgen["bus"]} boundary-gen {gen_id}')
            sub_trans.graph.remove_node(chosen_sgen["name"])
            data_dict['sgen'].remove(chosen_sgen)
            gen_id += 1
            return data_dict, sub_trans
        # 如果没有ext_grid,也没有gen,也没有sgen,但是有load,有load也不顶用啊，直接报错！
        raise Exception('没有找到slack')

    # ------------------------------------------------------------
    # 主配网合并操作
    # ------------------------------------------------------------
    def merge_distribution_with_transmission_topology(self, trans_topology: 'GridTopology', connection_nodes_trans: List[str]) -> 'GridTopology':
        """
        将两个拓扑图根据给定属性合并，并返回合并后的拓扑图。
        要求当前是配网拓扑图，trans_topology是主网拓扑图
        """
        # 最开始，先把配网中用来代替主网模型的gen节点全部删掉
        gen_nodes_to_remove = [n for n, node in self.graph.nodes(data=True) if node.get('type') == 'gen']
        for n in gen_nodes_to_remove:
            self.graph.remove_node(n)

        # 先遍历主网拓扑图，给每个节点都加上配网的节点数量，防止重复
        current_type_count = self.get_type_count()
        trans_relabel_mapping = dict()
        for n, node in trans_topology.graph.nodes(data=True):
            type = node['type']
            node['id'] += current_type_count[type]
            trans_relabel_mapping[n] = f'Trans-{type} {node["id"]}'
            if type == 'bus':
                continue
            elif type == 'load' or type == 'shunt' or type == 'ext_grid' or type == 'sgen':
                node['bus'] += current_type_count['bus']
            elif type == 'gen':
                node['bus'] += current_type_count['bus']
                # node['slack'] = False  # 因为只能有一个slack，所以这里需要设置为False  # 25-8-18: 先暂且就用主网的slack
            elif type == 'line':
                node['from_bus'] += current_type_count['bus']
                node['to_bus'] += current_type_count['bus']
            elif type == 'trafo':
                node['hv_bus'] += current_type_count['bus']
                node['lv_bus'] += current_type_count['bus']
            elif type == 'trafo3w':
                node['hv_bus'] += current_type_count['bus']
                node['mv_bus'] += current_type_count['bus']
                node['lv_bus'] += current_type_count['bus']
            elif type == 'switch':
                node['bus'] += current_type_count['bus']
                if node['et'] == 'b':
                    node['element'] += current_type_count['bus']
                elif node['et'] == 'l':
                    node['element'] += current_type_count['line']
                elif node['et'] == 't':
                    node['element'] += current_type_count['trafo']
                elif node['et'] == 't3':
                    node['element'] += current_type_count['trafo3w']
                else:
                    raise ValueError(f'Unknown node type: {type}')
            else:
                raise ValueError(f'Unknown node type: {type}')
        
        # 更新一下connected node的id
        connection_nodes_trans_new = []
        for connected_node in connection_nodes_trans:
            type, id = connected_node.split()
            connection_nodes_trans_new.append(f'Trans-{type} {int(id) + current_type_count[type]}')

        # 将主网配网的拓扑图先放到一个图里
        trans_final = nx.relabel_nodes(trans_topology.graph, trans_relabel_mapping, copy=True)
        merged_topology = nx.compose(self.graph, trans_final)

        # 得到配网和主网连接的节点映射表
        connection_nodes_distr = [n for n in self.graph.nodes if isinstance(n, str) and 'bus 0' in n]
        trans_distr_map = dict()
        for dist_node in connection_nodes_distr:
            dist_name = self.graph.nodes[dist_node]['name'][-3:]
            for trans_node in connection_nodes_trans_new:
                if dist_name in trans_final.nodes[trans_node]['name']:
                    trans_distr_map[trans_node] = dist_node
                    break

        
        # 在两个连接的bus之间添加一个联络开关
        tie_switch_id = TIE_SWITCH_ID_START
        for trans_load in connection_nodes_trans_new:
            trans_label = f'Trans-bus {trans_final.nodes[trans_load]["bus"]}'
            tie_switch_name = f'{trans_label}~{trans_distr_map[trans_load]}'
            merged_topology.add_node(
                tie_switch_name,
                id=tie_switch_id,
                type='switch',
                bus=trans_final.nodes[trans_load]['bus'],  # 从主网向配网连接
                et='b',
                element=self.graph.nodes[trans_distr_map[trans_load]]['id'],
                closed=True,
                is_zdhkg=False,  # 联络开关不参与潮流计算，这个开关是不能开的
            )
            merged_topology.add_edge(trans_label, tie_switch_name)  # 要从那个主网的bus连接过来
            merged_topology.add_edge(tie_switch_name, trans_distr_map[trans_load])
            merged_topology.remove_node(trans_load)
            tie_switch_id += 1

        return GridTopology(merged_topology)

        


    # ------------------------------------------------------------
    # 拓扑图序列化
    # ------------------------------------------------------------
    def to_pp_dict(self) -> Dict[str, Any]:
        """
        将拓扑图转换为Pandapower的格式。
        """
        graph_dict = dict(self.graph.nodes(data=True))
        pp_dict = defaultdict(list)
        for key, value in graph_dict.items():
            type = value['type']
            pp_dict[type].append(value)
        return pp_dict

    def to_pp_json(self, save_path: str):
        '''
        将拓扑图转换为Pandapower的JSON格式。
        2025-8-9: 暂时写在这，之后要系统性的写在serializer里
        '''
        pp_dict = self.to_pp_dict()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(pp_dict, f, indent=4, ensure_ascii=False)


    def simplify_graph(self) -> 'GridTopology':
        """
        简化图结构，删除两端都是'bus'或'switch'的中间节点。
        
        简化规则：
        1. 如果一个节点的两端邻居节点类型都是'bus'或'switch'之一，或该开关是断开的
        2. 则删除该节点，并将两端节点直接连接
        3. 重复此过程直到没有可简化的节点
        
        Returns:
            GridTopology: 简化后的拓扑图
        """
        # 创建图的副本以避免修改原图
        simplified_graph = self.graph.copy()
        
        # 记录简化过程
        removed_nodes = []
        added_edges = []
        
        # 持续简化直到没有可简化的节点
        while True:
            nodes_to_remove = []
            
            # 遍历所有节点，找到可以简化的节点
            for node in simplified_graph.nodes():
                if node in nodes_to_remove:
                    continue
                if simplified_graph.nodes[node].get('type') not in ['bus', 'switch']:  # 25-9-22 目前只移除这两种节点
                    continue
                # if simplified_graph.nodes[node].get('closed') == False:  # 把断开的开关也删掉  # 暂时不要这一条，不然可能会有多个孤岛
                #     nodes_to_remove.append(node)
                #     continue
                # 获取节点的邻居
                neighbors = list(simplified_graph.neighbors(node))
                
                # 如果节点只有两个邻居，且两个邻居的类型都是'bus'或'switch'
                if len(neighbors) == 2:
                    neighbor1, neighbor2 = neighbors
                    
                    # 检查邻居节点类型
                    neighbor1_type = simplified_graph.nodes[neighbor1].get('type', 'unknown')
                    neighbor2_type = simplified_graph.nodes[neighbor2].get('type', 'unknown')
                    
                    # 如果两个邻居都是'bus'或'switch'，则可以简化
                    if (neighbor1_type in ['bus', 'switch'] and 
                        neighbor2_type in ['bus', 'switch']):
                        nodes_to_remove.append(node)
            
            # 如果没有找到可简化的节点，退出循环
            if not nodes_to_remove:
                break
            
            # 删除找到的节点并连接其邻居
            for node in nodes_to_remove:
                # if simplified_graph.nodes[node].get('closed') == False:  # 暂时不要
                #     simplified_graph.remove_node(node)
                #     removed_nodes.append(node)
                #     continue
                
                neighbors = list(simplified_graph.neighbors(node))
                if len(neighbors) == 2:
                    neighbor1, neighbor2 = neighbors
                    
                    # 记录要添加的边
                    if not simplified_graph.has_edge(neighbor1, neighbor2):
                        simplified_graph.add_edge(neighbor1, neighbor2)
                        added_edges.append((neighbor1, neighbor2))
                    
                    # 删除节点
                    simplified_graph.remove_node(node)
                    removed_nodes.append(node)
        
        # 打印简化结果
        print(f"图简化完成：删除了 {len(removed_nodes)} 个节点，添加了 {len(added_edges)} 条边")
        if removed_nodes:
            print(f"删除的节点：{removed_nodes}")
        if added_edges:
            print(f"添加的边：{added_edges}")
        
        return GridTopology(simplified_graph)

    def get_open_switches(self) -> List[str]:
        """
        获取所有断开的开关。
        """
        return [n for n, node in self.graph.nodes(data=True) if node.get('type') == 'switch' and node.get('closed') == False]

    def get_graph_without_open_switches(self) -> nx.Graph:
        """
        获取一个删除了所有断开的开关的图。
        """
        graph = copy.deepcopy(self.graph)
        open_switches = self.get_open_switches()
        graph.remove_nodes_from(open_switches)
        return graph

    def __str__(self):
        """
        打印对象时输出电网节点数量信息。

        Returns:
            str: 电网节点数量描述字符串
        """
        return self.get_type_count()