# grid/visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Any
import matplotlib.patches as mpatches

from .topology import GridTopology

class GridVisualizer:
    """
    电网可视化器。支持按节点类型着色，并提供树状结构绘制能力。

    Args:
        topology GridTopology: 电网拓扑对象。
        type_color_map Dict[str, str] | None: 节点类型到颜色的映射，示例 {"bus": "#4C78A8"}。未提供时会使用默认颜色。
        default_color str: 当节点类型未在映射中时使用的默认颜色。
        layout_seed int | None: 随机布局的随机种子，保持可复现性；传入 None 则不固定。
    """
    def __init__(
        self,
        topology: GridTopology,
        type_color_map: Optional[Dict[str, str]] = None,
        default_color: str = "lightblue",
        layout_seed: Optional[int] = 42,
    ):
        if not isinstance(topology, GridTopology):
            raise TypeError("Input must be a GridTopology object.")
        self.topology = topology
        self.graph = self.topology.graph
        self.default_color = default_color
        # 默认颜色映射
        default_type_color_map = {
            "bus": "#1f77b4",     # 蓝
            "line": "#2ca02c",    # 绿
            "switch": "#ff7f0e",  # 橙
            "gen": "#9467bd",     # 紫
            "trafo": "#8c564b",   # 棕
            "shunt": "#17becf",   # 青
            "trafo3w": "#e377c2", # 粉
            "load": "#d62728",    # 红
        }
        # 若未提供则使用默认映射；若提供则作为初始映射
        self.type_color_map: Dict[str, str] = dict(type_color_map or default_type_color_map)
        self._layout_seed = layout_seed
        self._pos = None  # 延迟计算，避免大图初始化时卡在 spring_layout

        # 检测操作系统并设置matplotlib字体
        import platform
        system_name = platform.system().lower()
        if system_name == "windows":
            plt.rcParams['font.sans-serif'] = ['SimSun']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    @property
    def pos(self) -> Dict:
        """
        延迟计算的布局位置。仅在首次访问时执行 spring_layout。
        """
        if self._pos is None:
            self._pos = nx.spring_layout(self.graph, seed=self._layout_seed)
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def update_type_color_map(self, type_color_map: Dict[str, str], replace: bool = False) -> None:
        """
        更新节点类型到颜色的映射。

        Args:
            type_color_map Dict[str, str]: 新的类型-颜色配置。
            replace bool: 是否整体替换旧配置。False 表示增量更新，True 表示覆盖替换。

        Returns:
            None None: 无返回。
        """
        if replace:
            self.type_color_map = dict(type_color_map)
        else:
            self.type_color_map.update(type_color_map)

    def _auto_figsize(self, depth: int = 0, node_count: int = 0, figscale: float = 1.0) -> Tuple[int, int]:
        """
        根据搜索深度和子图节点数量自适应计算图片大小。
        以 depth=2 对应 (12, 9) 为基准，随深度和节点数增长而放大。

        Args:
            depth int: 搜索深度。
            node_count int: 子图的节点数量。
            figscale float: 用户缩放因子，乘在自适应 scale 上，默认 1.0。

        Returns:
            figsize Tuple[int, int]: (宽, 高)。
        """
        import math
        base_w, base_h = 12, 9

        # 深度因子：depth<=2 为 1.0，之后按 sqrt 增长
        depth_factor = max(1.0, math.sqrt(depth / 2.0))

        # 节点数因子：30 个节点以下为 1.0，之后按 sqrt 增长，上限 10
        node_factor = 1.0
        if node_count > 30:
            node_factor = min(math.sqrt(node_count / 30.0), 10)

        scale = max(depth_factor, node_factor) * figscale
        return (int(base_w * scale), int(base_h * scale))

    def _get_node_colors(self) -> list:
        """
        计算每个节点的颜色列表，按 `type` 属性映射颜色，缺省用默认色。

        Args:
            None None: 无参数。

        Returns:
            list list: 与 `self.graph.nodes()` 顺序一致的颜色列表。
        """
        colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "unknown")
            colors.append(self.type_color_map.get(node_type, self.default_color))
        return colors

    def _add_legend(self) -> None:
        """
        添加节点类型-颜色图例，仅显示当前图中出现过的类型；包含未知类型时一并显示。

        Args:
            None None: 无参数。

        Returns:
            None None: 无返回。
        """
        # 统计出现的类型
        used_types = set()
        for node in self.graph.nodes():
            used_types.add(self.graph.nodes[node].get("type", "unknown"))

        handles = []
        for node_type in sorted(used_types):
            color = self.type_color_map.get(node_type, self.default_color)
            handles.append(mpatches.Patch(color=color, label=node_type))

        if handles:
            plt.legend(handles=handles, title="节点类型", loc="best")

    def draw_basic(self) -> None:
        """
        绘制基础拓扑图（使用随机布局），并按节点类型着色。

        Args:
            None None: 无参数。

        Returns:
            None None: 无返回。
        """
        plt.figure(figsize=(10, 8))
        nx.draw(
            self.graph,
            self.pos,
            with_labels=True,
            node_color=self._get_node_colors(),
            node_size=700,
        )
        self._add_legend()
        plt.title("Basic Grid Topology")
        plt.show()

    def draw_with_attribute_labels(
        self,
        attribute: str,
        draw_tree: bool = False,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 16),
        node_size: int = 1500,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
    ) -> None:
        """
        绘制拓扑图，并使用指定的节点属性作为标签；节点按类型着色。

        Args:
            attribute str: 要显示为标签的节点属性名 (例如 'voltage')。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            node_size int: 节点尺寸。
            font_size int: 标签字号。
            font_color str: 标签字体颜色。
            edge_color str: 边颜色。

        Returns:
            None None: 无返回。
        """

        plt.figure(figsize=figsize)

        # 提取指定的属性作为标签
        labels = nx.get_node_attributes(self.graph, attribute)

        if draw_tree and nx.is_tree(self.graph):
            self.pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')  # 需要安装 pygraphviz
        else:
            self.pos = nx.spring_layout(self.graph, seed=42)

        nx.draw(
            self.graph,
            self.pos,
            labels=labels,
            with_labels=True,
            node_color=self._get_node_colors(),
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )
        self._add_legend()

        # 在节点ID旁边额外绘制一层节点ID，以作参考
        nx.draw_networkx_labels(
            self.graph,
            self.pos,
            labels={n: n for n in self.graph.nodes()},
            font_size=12,
            font_color='darkred',
            verticalalignment='bottom',
        )

        plt.title(f"Grid Topology with '{attribute}' Labels", size=15)

        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"Grid topology with labels saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def draw_with_highlighted_nodes(
        self,
        highlighted_nodes: List[Dict[str, Any]],
        attribute: str = 'name',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 9),
        default_node_size: int = 80,
        default_font_size: int = 8,
        default_font_color: str = "black",
        default_node_color: str = "lightgray",
        edge_color: str = "gray",
        highlight_node_size: int = 500,
        highlight_font_size: int = 12,
        highlight_font_color: str = "white",
        show_legend: bool = True,
    ) -> None:
        """
        绘制拓扑图，高亮显示指定的关键节点，其他节点只显示标签。

        Args:
            highlighted_nodes List[Dict[str, Any]]: 关键节点配置列表。每个字典必须包含 'name' 键，
                可选键包括 'node_size', 'color', 'font_size', 'font_color', 'label' 等。
                示例: [{'name': 'bus1', 'node_size': 3000, 'color': 'red', 'label': '关键母线'}]
            attribute str: 其他节点显示的属性名。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            default_node_size int: 其他节点的默认尺寸。
            default_font_size int: 其他节点的默认字号。
            default_font_color str: 其他节点的默认字体颜色。
            default_node_color str: 其他节点的默认颜色。
            edge_color str: 边颜色。
            highlight_node_size int: 高亮节点的默认尺寸。
            highlight_font_size int: 高亮节点的默认字号。
            highlight_font_color str: 高亮节点的默认字体颜色。
            # show_legend bool: 是否显示图例。

        Returns:
            None None: 无返回。
        
        Example:
            highlighted_nodes = [
                {'name': 'bus1', 'node_size': 3000, 'color': 'red', 'label': '关键母线'},
                {'name': 'bus2', 'node_size': 2000, 'color': 'blue', 'label': '关键母线2'}
            ]
            visualizer.draw_with_highlighted_nodes(highlighted_nodes, attribute='name', save_path='highlighted_nodes.png')
        """
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=figsize)

        # 提取高亮节点名称集合
        highlighted_names = {node_config['name'] for node_config in highlighted_nodes}
        
        # 创建节点颜色和尺寸映射
        node_colors = {}
        node_sizes = {}
        node_labels = {}
        node_font_sizes = {}
        node_font_colors = {}
        
        # 处理高亮节点
        for node_config in highlighted_nodes:
            node_name = node_config['name']
            found_nodes_generator = (node for node, data in self.graph.nodes(data=True) if data.get('name', '') == node_name)
            for node in found_nodes_generator:
                node_colors[node] = node_config.get('color', 'red')
                node_sizes[node] = node_config.get('node_size', highlight_node_size)
                node_labels[node] = node_config.get('label', node)
                node_font_sizes[node] = node_config.get('font_size', highlight_font_size)
                node_font_colors[node] = node_config.get('font_color', highlight_font_color)
        
        if nx.is_tree(self.graph):
            self.pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')  # 需要安装 pygraphviz
        else:
            self.pos = nx.spring_layout(self.graph, seed=42)

        nx.draw(
            self.graph,
            self.pos,
            labels={n: n for n in self.graph.nodes()},
            with_labels=True,
            node_color=default_node_color,
            node_size=default_node_size,
            font_size=default_font_size,
            font_color=default_font_color,
            edge_color=edge_color,
        )
        
        # 绘制高亮节点
        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            # labels={n: n for n in self.graph.nodes()},
            # with_labels=True,
            nodelist=list(node_colors.keys()),
            node_color=list(node_colors.values()),
            node_size=list(node_sizes.values()),
        )

        # 绘制标签（分两批：高亮节点和其他节点）
        # 高亮节点标签
        highlight_labels = {node: node_labels[node] for node in highlighted_names if node in self.graph}
        if highlight_labels:
            nx.draw_networkx_labels(
                self.graph,
                self.pos,
                labels=highlight_labels,
                font_size=highlight_font_size,
                font_color=highlight_font_color,
                font_weight='bold',
            )
        
        # 其他节点标签
        # other_labels = {node: node_labels[node] for node in self.graph.nodes() if node not in highlighted_names}
        # if other_labels:
        #     nx.draw_networkx_labels(
        #         self.graph,
        #         self.pos,
        #         labels=other_labels,
        #         font_size=default_font_size,
        #         font_color=default_font_color,
        #     )

        # 添加图例
        # if show_legend:
        #     self._add_legend()
        
        plt.title(f"Grid Topology with Highlighted Nodes", size=15)
        
        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"Grid topology with highlighted nodes saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def draw_tree(
        self,
        root: Optional[str] = None,
        attribute: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 9),
        node_size: int = 1200,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
        level_gap: float = 1.5,
        sibling_gap: float = 1.2,
    ) -> None:
        """
        检查当前图是否为树，若是则以分层（自上而下）方式绘制树状结构；否则不绘制并给出提示。

        Args:
            root str | None: 根节点名称。未提供时自动选择一个度为1的叶子作为根，若无叶子则任选一个节点。
            attribute str: 节点属性名。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            node_size int: 节点尺寸。
            font_size int: 标签字号。
            font_color str: 标签颜色。
            edge_color str: 边颜色。
            level_gap float: 层级之间的垂直间距。
            sibling_gap float: 同层兄弟节点之间的水平间距。

        Returns:
            None None: 无返回。
        """
        # 仅在无向树上绘制
        if not nx.is_tree(self.graph):
            print("当前图不是树结构，已跳过树状绘制。")
            return

        # 选取根
        if root is not None and root not in self.graph:
            raise ValueError(f"指定的根节点 {root} 不在图中。")
        if root is None:
            root = next((n for n, d in self.graph.degree() if d == 1), next(iter(self.graph.nodes())))
        # 优先使用 Graphviz dot 布局（需要 pygraphviz 或 pydot）
        pos = None
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')  # 需要安装 pygraphviz
        except Exception:
            try:
                pos = nx.nx_pydot.graphviz_layout(self.graph, prog='dot')  # 需要安装 pydot
            except Exception:
                pos = None

        if pos is None:
            # 回退到简单的分层坐标方案
            levels = nx.single_source_shortest_path_length(self.graph, root)
            max_level = max(levels.values()) if levels else 0
            level_to_nodes = {}
            for node, lvl in levels.items():
                level_to_nodes.setdefault(lvl, []).append(node)
            pos = {}
            for lvl in range(max_level + 1):
                nodes_in_level = level_to_nodes.get(lvl, [])
                count = len(nodes_in_level)
                if count == 0:
                    continue
                total_width = (count - 1) * sibling_gap
                xs = [(-total_width / 2.0) + i * sibling_gap for i in range(count)]
                y = -lvl * level_gap
                for x, node in zip(xs, nodes_in_level):
                    pos[node] = (x, y)

        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=figsize)

        labels = nx.get_node_attributes(self.graph, attribute)
        nx.draw(
            self.graph,
            pos,
            labels={n: n for n in self.graph.nodes()},
            with_labels=True,
            node_color=self._get_node_colors(),
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )
        self._add_legend()

        # 同样叠加节点ID参考层
        if attribute is not None:
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels=labels,
                font_size=12,
                font_color='darkred',
                verticalalignment='bottom',
            )

        plt.title("Tree Layout of Grid Topology", size=15)

        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"Tree topology saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def draw_centered_subgraph_by_electrical_distance(
        self,
        center_name: str,
        depth: int = 2,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        node_size: int = 1200,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
        center_node_size: int = 2000,
        center_node_color: str = "red",
        center_font_size: int = 12,
        center_font_color: str = "white",
        show_legend: bool = True,
        node_label_attributes: Optional[List[str]] = None,
        max_chars_per_line: int = 10,
        dim_closed_switches: bool = True,
        figscale: float = 1.0,
        simplify: bool = False,
    ) -> None:
        """
        以指定节点为中心，绘制指定深度内的临近节点子图（电气距离）。 (要有pq以及vmva这种边界节点，不然就会打印全图)

        Args:
            center_name str: 中心节点的name属性值。
            depth int: 搜索深度（电气距离）。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            node_size int: 普通节点尺寸。
            font_size int: 普通节点字号。
            font_color str: 普通节点字体颜色。
            edge_color str: 边颜色。
            center_node_size int: 中心节点尺寸。
            center_node_color str: 中心节点颜色。
            center_font_size int: 中心节点字号。
            center_font_color str: 中心节点字体颜色。
            show_legend bool: 是否显示图例。
            node_label_attributes List[str] | None: 要显示的节点属性列表，如['name', 'id']。为None时只显示节点ID。
            max_chars_per_line int: 每行最大字符数，超过则自动换行。
            dim_closed_switches bool: 是否将关闭状态的开关节点颜色调淡。

        Returns:
            None None: 无返回。
        
        Example:
            visualizer.draw_centered_subgraph_by_electrical_distance("bus1", depth=2, save_path="centered_subgraph.png")
            visualizer.draw_centered_subgraph_by_electrical_distance("bus1", depth=2, node_label_attributes=['name', 'id'])
        """
        # 查找中心节点
        center_nodes = self._find_nodes_by_name(center_name)
        if not center_nodes:
            print(f"未找到name属性为 '{center_name}' 的节点")
            return
        
        if len(center_nodes) > 1:
            print(f"找到多个name属性为 '{center_name}' 的节点: {center_nodes}")
            print("将使用第一个节点作为中心")
        
        center_node = center_nodes[0]
        
        # 使用多源BFS搜索获取子图
        if depth == -1:
            subgraph_topology = self.topology
            boundary_nodes = []
        else:   
            subgraph_topology, boundary_nodes = self.topology.multi_source_electrical_bfs(
                [center_node], max_depth=depth
            )
        
        if subgraph_topology.graph.number_of_nodes() == 0:
            print(f"在深度 {depth} 内未找到任何节点")
            return
        
        # 简化图结构
        if simplify:
            subgraph_topology = subgraph_topology.simplify_graph()

        # 创建子图的可视化器
        subgraph_visualizer = GridVisualizer(
            subgraph_topology,
            type_color_map=self.type_color_map,
            default_color=self.default_color,
            layout_seed=42
        )
        
        # 自适应图片大小
        if figsize is None:
            figsize = self._auto_figsize(depth, subgraph_topology.graph.number_of_nodes(), figscale)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=figsize)
        
        # 使用分层布局，中心节点在顶部
        pos = self._create_centered_layout(subgraph_topology.graph, center_node)
        
        # 生成节点标签
        node_labels = self._generate_node_labels(subgraph_topology.graph, node_label_attributes, max_chars_per_line)
        
        # 获取节点颜色，对关闭的开关进行特殊处理
        node_colors = subgraph_visualizer._get_node_colors()
        if dim_closed_switches:
            node_colors = self._adjust_colors_for_closed_switches(subgraph_topology.graph, node_colors)
        

        # 绘制所有节点（普通样式）
        nx.draw(
            subgraph_topology.graph,
            pos,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )
        
        # 高亮中心节点
        nx.draw_networkx_nodes(
            subgraph_topology.graph,
            pos,
            nodelist=[center_node],
            node_color=[center_node_color],
            node_size=[center_node_size],
        )
        
        # 绘制中心节点标签
        nx.draw_networkx_labels(
            subgraph_topology.graph,
            pos,
            labels={center_node: center_node},
            font_size=center_font_size,
            font_color=center_font_color,
            font_weight='bold',
        )
        
        # 高亮边界节点（如果有的话）
        if boundary_nodes:
            boundary_colors = ['blue'] * len(boundary_nodes)
            nx.draw_networkx_nodes(
                subgraph_topology.graph,
                pos,
                nodelist=boundary_nodes,
                node_color=boundary_colors,
                node_size=[node_size * 1.2] * len(boundary_nodes),
            )
        
        # 添加图例
        if show_legend:
            handles = []
            # 中心节点图例
            handles.append(mpatches.Patch(color=center_node_color, label=f'中心节点 ({center_name})'))
            # 边界节点图例
            if boundary_nodes:
                handles.append(mpatches.Patch(color='blue', label='边界节点'))
            # 节点类型图例
            type_count = {}
            for node in subgraph_topology.graph.nodes():
                node_type = subgraph_topology.graph.nodes[node].get("type", "unknown")
                type_count[node_type] = type_count.get(node_type, 0) + 1
            
            for node_type in sorted(type_count.keys()):
                color = self.type_color_map.get(node_type, self.default_color)
                count = type_count[node_type]
                handles.append(mpatches.Patch(color=color, label=f'{node_type} ({count})'))
            
            if handles:
                plt.legend(handles=handles, title="节点类型", loc="best")
        
        plt.title(f"以 '{center_name}' 为中心的子图 (深度={depth})", size=15)
        
        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"中心子图已保存到 {save_path}")
            plt.close()
        else:
            plt.show()

    def _find_nodes_by_name(self, name: str) -> List[str]:
        """
        根据节点的name属性查找节点ID。

        Args:
            name str: 要查找的name属性值。

        Returns:
            List[str]: 匹配的节点ID列表。
        """
        matching_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get('name', '') == name:
                matching_nodes.append(node)
        return matching_nodes

    def _generate_node_labels(self, graph: nx.Graph, node_label_attributes: Optional[List[str]] = None, max_chars_per_line: int = 10) -> Dict[str, str]:
        """
        生成节点标签字典，支持显示多个属性，并支持自动换行。

        Args:
            graph nx.Graph: 要生成标签的图。
            node_label_attributes List[str] | None: 要显示的节点属性列表。为None时只显示节点ID。
            max_chars_per_line int: 每行最大字符数，超过则自动换行。

        Returns:
            Dict[str, str]: 节点ID到标签字符串的映射。
        """
        def _wrap_text(text: str, max_chars: int) -> str:
            """
            将文本按指定字符数换行。
            
            Args:
                text str: 要换行的文本。
                max_chars int: 每行最大字符数。
                
            Returns:
                str: 换行后的文本。
            """
            if len(text) <= max_chars:
                return text
            
            lines = []
            current_line = ""
            
            for char in text:
                if len(current_line) >= max_chars:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            return "\n".join(lines)
        
        labels = {}
        for node in graph.nodes():
            if node_label_attributes is None:
                # 默认只显示节点ID
                labels[node] = _wrap_text(node, max_chars_per_line)
            else:
                # 显示指定的属性
                label_parts = []
                for attr in node_label_attributes:
                    if attr in graph.nodes[node]:
                        value = graph.nodes[node][attr]
                        if isinstance(value, (int, float)):
                            # 数字最多显示3位小数
                            if isinstance(value, int):
                                attr_text = f"{attr}={value}"
                            else:
                                attr_text = f"{attr}={value:.3f}"
                        else:
                            attr_text = f"{attr}={value}"
                    else:
                        # attr_text = f"{attr}=N/A"
                        continue
                    
                    # 对每个属性文本进行换行处理
                    wrapped_text = _wrap_text(attr_text, max_chars_per_line)
                    label_parts.append(wrapped_text)
                
                labels[node] = "\n".join(label_parts)
        return labels

    def _get_node_colors_by_attribute(self, graph: nx.Graph, attribute: str, 
                                     colormap: str = 'RdYlGn', 
                                     default_color: str = 'lightgray') -> List[str]:
        """
        根据节点属性值生成颜色列表，支持渐变色彩映射。

        Args:
            graph nx.Graph: 要生成颜色的图。
            attribute str: 用于着色的属性名称。
            colormap str: matplotlib颜色映射名称，默认为'RdYlGn'（红-黄-绿）。
            default_color str: 没有该属性的节点颜色，默认为浅灰色。

        Returns:
            List[str]: 与 `graph.nodes()` 顺序一致的颜色列表。
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        
        # 收集所有节点的属性值
        values = []
        valid_nodes = []
        
        for node in graph.nodes():
            if attribute in graph.nodes[node]:
                value = graph.nodes[node][attribute]
                if isinstance(value, (int, float)):
                    # 检查是否为NaN值
                    import math
                    if not math.isnan(value) and value != 0:
                        values.append(value)
                        valid_nodes.append(node)
        
        if not values:
            # 如果没有有效值，返回默认颜色
            return [default_color] * len(graph.nodes())
        
        # 创建颜色映射
        cmap = cm.get_cmap(colormap)
        
        # 归一化值到[0,1]范围
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            # 如果所有值都相同，使用中间颜色
            normalized_values = [0.5] * len(values)
        else:
            # 对于vm_pu属性，值越大越红，所以需要反向映射
            if attribute == 'vm_pu':
                normalized_values = [1 - (v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        
        # 生成颜色
        colors = []
        value_dict = dict(zip(valid_nodes, normalized_values))
        
        for node in graph.nodes():
            if node in value_dict:
                # 使用归一化值获取颜色
                color = cmap(value_dict[node])
                colors.append(color)
            else:
                # 没有该属性的节点使用默认颜色
                colors.append(default_color)
        
        return colors

    def _adjust_colors_for_closed_switches(self, graph: nx.Graph, node_colors: List[str]) -> List[str]:
        """
        调整关闭状态开关节点的颜色，使其变淡。

        Args:
            graph nx.Graph: 要处理的图。
            node_colors List[str]: 原始节点颜色列表。

        Returns:
            List[str]: 调整后的节点颜色列表。
        """
        import matplotlib.colors as mcolors
        
        adjusted_colors = []
        node_list = list(graph.nodes())
        
        for i, node in enumerate(node_list):
            node_data = graph.nodes[node]
            # 检查是否为开关且关闭状态
            if (node_data.get('type') == 'switch' and 
                'closed' in node_data and 
                node_data['closed'] is False):
                # 将颜色调淡（降低透明度或混合白色）
                original_color = node_colors[i]
                # 将颜色转换为RGBA格式
                original_rgba = mcolors.to_rgba(original_color)
                # 混合白色使颜色变淡（70%白色 + 30%原色）
                white_rgba = mcolors.to_rgba('white')
                final_color = tuple(
                    original_rgba[j] * 0.3 + white_rgba[j] * 0.7
                    for j in range(3)  # RGB 三个通道
                )
                # 转换回十六进制颜色
                final_hex = mcolors.to_hex(final_color)
                adjusted_colors.append(final_hex)
            else:
                adjusted_colors.append(node_colors[i])
        
        return adjusted_colors

    def _create_centered_layout(self, graph: nx.Graph, center_node: str) -> Dict[str, Tuple[float, float]]:
        """
        创建以指定节点为中心的分层布局。

        Args:
            graph nx.Graph: 要布局的图。
            center_node str: 中心节点ID。

        Returns:
            Dict[str, Tuple[float, float]]: 节点位置字典。
        """
        # 使用BFS计算每个节点到中心节点的距离
        distances = nx.single_source_shortest_path_length(graph, center_node)
        
        # 按距离分组
        level_to_nodes = {}
        for node, dist in distances.items():
            level_to_nodes.setdefault(dist, []).append(node)
        
        pos = {}
        max_level = max(distances.values()) if distances else 0
        
        # 为每一层分配位置
        for level in range(max_level + 1):
            nodes_in_level = level_to_nodes.get(level, [])
            count = len(nodes_in_level)
            if count == 0:
                continue
            
            # 计算水平位置
            if count == 1:
                xs = [0.0]
            else:
                total_width = (count - 1) * 2.0
                xs = [(-total_width / 2.0) + i * 2.0 for i in range(count)]
            
            # 垂直位置：中心节点在顶部，其他层向下排列
            y = -level * 2.0
            
            for x, node in zip(xs, nodes_in_level):
                pos[node] = (x, y)
        
        # 处理图中存在但不在BFS结果中的节点（比如孤立节点或断开的组件）
        all_nodes = set(graph.nodes())
        positioned_nodes = set(pos.keys())
        unpositioned_nodes = all_nodes - positioned_nodes
        
        if unpositioned_nodes:
            # 为未定位的节点分配位置，放在图的右侧
            unpositioned_list = list(unpositioned_nodes)
            for i, node in enumerate(unpositioned_list):
                # 在右侧垂直排列
                x = max_level * 2.0 + 2.0  # 在最后一层右侧
                y = -i * 2.0  # 垂直排列
                pos[node] = (x, y)
        
        return pos

    def draw_centered_subgraph_by_topology(
        self,
        center_name: str,
        depth: int = 2,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        node_size: int = 1200,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
        center_node_size: int = 2000,
        center_node_color: str = "red",
        center_font_size: int = 12,
        center_font_color: str = "white",
        show_legend: bool = True,
        node_label_attributes: Optional[List[str]] = None,
        max_chars_per_line: int = 10,
        dim_closed_switches: bool = True,
        figscale: float = 1.0,
        simplify: bool = False,
    ) -> None:
        """
        以指定节点为中心，绘制指定拓扑深度内的临近节点子图。

        Args:
            center_name str: 中心节点的name属性值。
            depth int: 搜索深度（拓扑距离）。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            node_size int: 普通节点尺寸。
            font_size int: 普通节点字号。
            font_color str: 普通节点字体颜色。
            edge_color str: 边颜色。
            center_node_size int: 中心节点尺寸。
            center_node_color str: 中心节点颜色。
            center_font_size int: 中心节点字号。
            center_font_color str: 中心节点字体颜色。
            show_legend bool: 是否显示图例。
            node_label_attributes List[str] | None: 要显示的节点属性列表，如['name', 'id']。为None时只显示节点ID。
            max_chars_per_line int: 每行最大字符数，超过则自动换行。
            dim_closed_switches bool: 是否将关闭状态的开关节点颜色调淡。

        Returns:
            None None: 无返回。
        
        Example:
            visualizer.draw_centered_subgraph_by_topology("bus1", depth=2, save_path="topology_centered_subgraph.png")
        """
        # 查找中心节点
        center_nodes = self._find_nodes_by_name(center_name)
        if not center_nodes:
            print(f"未找到name属性为 '{center_name}' 的节点")
            return
        
        if len(center_nodes) > 1:
            print(f"找到多个name属性为 '{center_name}' 的节点: {center_nodes}")
            print("将使用第一个节点作为中心")
        
        center_node = center_nodes[0]
        
        # 使用多源拓扑BFS搜索获取子图
        subgraph_topology, boundary_nodes = self.topology.multi_source_topology_bfs(
            [center_node], max_depth=depth
        )
        
        if subgraph_topology.graph.number_of_nodes() == 0:
            print(f"在深度 {depth} 内未找到任何节点")
            return
        
        # 简化图结构
        if simplify:
            subgraph_topology = subgraph_topology.simplify_graph()

        # 创建子图的可视化器
        subgraph_visualizer = GridVisualizer(
            subgraph_topology,
            type_color_map=self.type_color_map,
            default_color=self.default_color,
            layout_seed=42
        )
        
        # 自适应图片大小
        if figsize is None:
            figsize = self._auto_figsize(depth, subgraph_topology.graph.number_of_nodes(), figscale)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=figsize)
        
        # 使用分层布局，中心节点在顶部
        pos = self._create_centered_layout(subgraph_topology.graph, center_node)
        
        # 生成节点标签
        node_labels = self._generate_node_labels(subgraph_topology.graph, node_label_attributes, max_chars_per_line)
        
        # 获取节点颜色，对关闭的开关进行特殊处理
        node_colors = subgraph_visualizer._get_node_colors()
        if dim_closed_switches:
            node_colors = self._adjust_colors_for_closed_switches(subgraph_topology.graph, node_colors)
        
        # 绘制所有节点（普通样式）
        nx.draw(
            subgraph_topology.graph,
            pos,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )
        
        # 高亮中心节点
        nx.draw_networkx_nodes(
            subgraph_topology.graph,
            pos,
            nodelist=[center_node],
            node_color=[center_node_color],
            node_size=[center_node_size],
        )
        
        # 绘制中心节点标签
        nx.draw_networkx_labels(
            subgraph_topology.graph,
            pos,
            labels={center_node: center_node},
            font_size=center_font_size,
            font_color=center_font_color,
            font_weight='bold',
        )
        
        # 高亮边界节点（如果有的话）
        if boundary_nodes:
            boundary_colors = ['blue'] * len(boundary_nodes)
            nx.draw_networkx_nodes(
                subgraph_topology.graph,
                pos,
                nodelist=boundary_nodes,
                node_color=boundary_colors,
                node_size=[node_size * 1.2] * len(boundary_nodes),
            )
        
        # 添加图例
        if show_legend:
            handles = []
            # 中心节点图例
            handles.append(mpatches.Patch(color=center_node_color, label=f'中心节点 ({center_name})'))
            # 边界节点图例
            if boundary_nodes:
                handles.append(mpatches.Patch(color='blue', label='边界节点'))
            # 节点类型图例
            type_count = {}
            for node in subgraph_topology.graph.nodes():
                node_type = subgraph_topology.graph.nodes[node].get("type", "unknown")
                type_count[node_type] = type_count.get(node_type, 0) + 1
            
            for node_type in sorted(type_count.keys()):
                color = self.type_color_map.get(node_type, self.default_color)
                count = type_count[node_type]
                handles.append(mpatches.Patch(color=color, label=f'{node_type} ({count})'))
            
            if handles:
                plt.legend(handles=handles, title="节点类型", loc="best")
        
        plt.title(f"以 '{center_name}' 为中心的子图 (拓扑深度={depth})", size=15)
        
        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"拓扑中心子图已保存到 {save_path}")
            plt.close()
        else:
            plt.show()

    def draw_centered_subgraph_by_attribute(
        self,
        center_name: str,
        attribute: str,
        depth: int = 2,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        node_size: int = 1200,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
        center_node_size: int = 2000,
        center_node_color: str = "red",
        center_font_size: int = 12,
        center_font_color: str = "white",
        show_legend: bool = True,
        node_label_attributes: Optional[List[str]] = None,
        colormap: str = 'RdYlGn',
        default_color: str = 'lightgray',
        show_colorbar: bool = True,
        max_chars_per_line: int = 10,
        dim_closed_switches: bool = True,
        figscale: float = 1.0,
        simplify: bool = False,
    ) -> None:
        """
        以指定节点为中心，绘制指定深度内的临近节点子图，按指定属性值上色。

        Args:
            center_name str: 中心节点的name属性值。
            attribute str: 用于着色的属性名称。
            depth int: 搜索深度（电气距离）。
            save_path str | None: 保存路径。
            figsize tuple[int, int]: 图像尺寸。
            node_size int: 普通节点尺寸。
            font_size int: 普通节点字号。
            font_color str: 普通节点字体颜色。
            edge_color str: 边颜色。
            center_node_size int: 中心节点尺寸。
            center_node_color str: 中心节点颜色。
            center_font_size int: 中心节点字号。
            center_font_color str: 中心节点字体颜色。
            show_legend bool: 是否显示图例。
            node_label_attributes List[str] | None: 要显示的节点属性列表，如['name', 'id']。为None时只显示节点ID。
            colormap str: matplotlib颜色映射名称，默认为'RdYlGn'（红-黄-绿）。
            default_color str: 没有该属性的节点颜色，默认为浅灰色。
            show_colorbar bool: 是否显示颜色条。
            max_chars_per_line int: 每行最大字符数，超过则自动换行。
            dim_closed_switches bool: 是否将关闭状态的开关节点颜色调淡。

        Returns:
            None None: 无返回。
        
        Example:
            visualizer.draw_centered_subgraph_by_attribute("bus1", "vm_pu", depth=2, save_path="colored_subgraph.png")
        """
        # 查找中心节点
        center_nodes = self._find_nodes_by_name(center_name)
        if not center_nodes:
            print(f"未找到name属性为 '{center_name}' 的节点")
            return
        
        if len(center_nodes) > 1:
            print(f"找到多个name属性为 '{center_name}' 的节点: {center_nodes}")
            print("将使用第一个节点作为中心")
        
        center_node = center_nodes[0]
        
        # 使用多源BFS搜索获取子图
        subgraph_topology, boundary_nodes = self.topology.multi_source_electrical_bfs(
            [center_node], max_depth=depth
        )
        
        if subgraph_topology.graph.number_of_nodes() == 0:
            print(f"在深度 {depth} 内未找到任何节点")
            return
        
        # 简化图结构
        if simplify:
            subgraph_topology = subgraph_topology.simplify_graph()

        # 创建子图的可视化器
        subgraph_visualizer = GridVisualizer(
            subgraph_topology,
            type_color_map=self.type_color_map,
            default_color=self.default_color,
            layout_seed=42
        )
        
        # 自适应图片大小
        if figsize is None:
            figsize = self._auto_figsize(depth, subgraph_topology.graph.number_of_nodes(), figscale)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=figsize)
        
        # 使用分层布局，中心节点在顶部
        pos = self._create_centered_layout(subgraph_topology.graph, center_node)
        
        # 生成节点标签
        node_labels = self._generate_node_labels(subgraph_topology.graph, node_label_attributes, max_chars_per_line)
        
        # 获取节点颜色，对关闭的开关进行特殊处理
        node_colors = subgraph_visualizer._get_node_colors()
        if dim_closed_switches:
            node_colors = self._adjust_colors_for_closed_switches(subgraph_topology.graph, node_colors)
        
        # 根据属性值生成颜色
        node_colors = self._get_node_colors_by_attribute(
            subgraph_topology.graph, 
            attribute, 
            colormap=colormap, 
            default_color=default_color
        )
        
        # 绘制所有节点（按属性值着色）
        nx.draw(
            subgraph_topology.graph,
            pos,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )
        
        # 高亮中心节点
        nx.draw_networkx_nodes(
            subgraph_topology.graph,
            pos,
            nodelist=[center_node],
            node_color=[center_node_color],
            node_size=[center_node_size],
        )
        
        # 绘制中心节点标签
        nx.draw_networkx_labels(
            subgraph_topology.graph,
            pos,
            labels={center_node: center_node},
            font_size=center_font_size,
            font_color=center_font_color,
            font_weight='bold',
        )
        
        # 高亮边界节点（如果有的话）
        if boundary_nodes:
            boundary_colors = ['blue'] * len(boundary_nodes)
            nx.draw_networkx_nodes(
                subgraph_topology.graph,
                pos,
                nodelist=boundary_nodes,
                node_color=boundary_colors,
                node_size=[node_size * 1.2] * len(boundary_nodes),
            )
        
        # 添加颜色条
        if show_colorbar:
            import matplotlib.cm as cm
            import matplotlib.colorbar as cbar
            
            # 收集属性值用于颜色条
            values = []
            import math
            for node in subgraph_topology.graph.nodes():
                if attribute in subgraph_topology.graph.nodes[node]:
                    value = subgraph_topology.graph.nodes[node][attribute]
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        values.append(value)
            
            if values:
                cmap = cm.get_cmap(colormap)
                min_val, max_val = min(values), max(values)
                
                # 创建颜色条
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
                cbar.set_label(f'{attribute} 值', rotation=270, labelpad=20)
        
        # 添加图例
        if show_legend:
            handles = []
            # 中心节点图例
            handles.append(mpatches.Patch(color=center_node_color, label=f'中心节点 ({center_name})'))
            # 边界节点图例
            if boundary_nodes:
                handles.append(mpatches.Patch(color='blue', label='边界节点'))
            # 属性值着色说明
            handles.append(mpatches.Patch(color='lightgray', label=f'按 {attribute} 值着色'))
            # 默认颜色说明
            if default_color != 'lightgray':
                handles.append(mpatches.Patch(color=default_color, label=f'无 {attribute} 属性'))
            
            if handles:
                plt.legend(handles=handles, title="图例", loc="best")
        
        plt.title(f"以 '{center_name}' 为中心的子图 (按 {attribute} 着色, 深度={depth})", size=15)
        
        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"属性着色子图已保存到 {save_path}")
            plt.close()
        else:
            plt.show()

    def draw_subgraph_by_feeder(
        self,
        feeder_value: Any,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        node_size: int = 1200,
        font_size: int = 10,
        font_color: str = "black",
        edge_color: str = "gray",
        show_legend: bool = True,
        node_label_attributes: Optional[List[str]] = None,
        max_chars_per_line: int = 10,
        dim_closed_switches: bool = True,
        figscale: float = 1.0,
    ) -> None:
        """
        仅可视化 feeder 属性等于指定值的元件子图，绘图风格与中心子图函数保持一致。

        Args:
            feeder_value Any: feeder 属性目标值，仅保留 feeder == feeder_value 的节点。
            save_path str | None: 保存路径。
            figsize tuple[int, int] | None: 图像尺寸。为 None 时按节点数自适应。
            node_size int: 节点尺寸。
            font_size int: 标签字号。
            font_color str: 标签字体颜色。
            edge_color str: 边颜色。
            show_legend bool: 是否显示图例。
            node_label_attributes List[str] | None: 要显示的节点属性列表，如 ['name', 'id']。为 None 时只显示节点 ID。
            max_chars_per_line int: 每行最大字符数，超过则自动换行。
            dim_closed_switches bool: 是否将关闭状态的开关节点颜色调淡。
            figscale float: 图片缩放因子，乘在自适应尺寸上。

        Returns:
            None None: 无返回。

        Example:
            visualizer.draw_subgraph_by_feeder("feeder_1", save_path="feeder_subgraph.png")
        """
        feeder_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("feeder") == feeder_value
        ]

        if not feeder_nodes:
            print(f"未找到 feeder 属性为 '{feeder_value}' 的节点")
            return

        subgraph = self.graph.subgraph(feeder_nodes).copy()
        if subgraph.number_of_nodes() == 0:
            print(f"feeder='{feeder_value}' 的子图为空")
            return

        # 按子图中的最大度节点选一个拓扑中心，沿用分层布局风格。
        center_node = max(subgraph.degree, key=lambda x: x[1])[0]

        subgraph_topology = GridTopology(subgraph)
        subgraph_visualizer = GridVisualizer(
            subgraph_topology,
            type_color_map=self.type_color_map,
            default_color=self.default_color,
            layout_seed=42
        )

        if figsize is None:
            figsize = self._auto_figsize(0, subgraph.number_of_nodes(), figscale)

        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=figsize)

        pos = self._create_centered_layout(subgraph, center_node)
        node_labels = self._generate_node_labels(subgraph, node_label_attributes, max_chars_per_line)

        node_colors = subgraph_visualizer._get_node_colors()
        if dim_closed_switches:
            node_colors = self._adjust_colors_for_closed_switches(subgraph, node_colors)

        nx.draw(
            subgraph,
            pos,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
        )

        if show_legend:
            handles = []
            type_count: Dict[str, int] = {}
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get("type", "unknown")
                type_count[node_type] = type_count.get(node_type, 0) + 1

            for node_type in sorted(type_count.keys()):
                color = self.type_color_map.get(node_type, self.default_color)
                count = type_count[node_type]
                handles.append(mpatches.Patch(color=color, label=f'{node_type} ({count})'))

            if handles:
                plt.legend(handles=handles, title="节点类型", loc="best")

        plt.title(f"feeder = '{feeder_value}' 的元件子图", size=15)

        if save_path:
            plt.savefig(save_path, format="PNG", dpi=300)
            print(f"feeder 子图已保存到 {save_path}")
            plt.close()
        else:
            plt.show()

