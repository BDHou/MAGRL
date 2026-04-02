import json
import os
import networkx as nx
import numpy as np
import logging
from datetime import datetime, timedelta

class DistributionNetworkParser:
    def __init__(self):
        pass

    def parse(self, data_path, time_indexes=None, ref_timestamp=None, region_name='', save_network=False):
        self.region_name = region_name
        logging.debug(f"Parsing distribution network from {data_path}...")
        with open(data_path, "r", encoding='utf-8') as f:
            data_dict = json.load(f)
        
        if ref_timestamp is None:
            if time_indexes is None:
                self.time_series_idx = [i for i in range(96)]
            else:
                self.time_series_idx = time_indexes
        else:
            format_pattern = "%Y-%m-%d %H:%M:%S"
            closest_time, closest_idx = self._find_closest_time_point(
                start=datetime.strptime(data_dict['data']['timeSeries']['start'], format_pattern),
                end=datetime.strptime(data_dict['data']['timeSeries']['end'], format_pattern),
                count=int(data_dict['data']['timeSeries']['periods']),
                ref=ref_timestamp
            )
            self.time_series_idx = [closest_idx]
        # print(f'Closest time: {closest_time}, Closest index: {closest_idx}, Ref timestamp: {ref_timestamp}')
        # exit()

        self.G = nx.Graph()
        self.bus_id = {}
        relationship = []
        for rel in data_dict['data']['relationships']:
            self.G.add_edge(rel['from'], rel['to'])
            relationship.append((rel['from'], rel['to']))

        device_type_new = ['bus', 'trafo', 'gen', 'load', 'line', 'switch']
        self.id_counter = {dt: 0 for dt in device_type_new}
        self.bus_id = dict()  # 母线idx
        self.bus_vn = dict()  # 母线电压
        self.dev_id = dict()  # 每个设备在自己种类中的idx，仅在当前json中有效
        self.nodes_info = dict()  # 每个设备的信息，目前就包括

        # 第一遍遍历，获取所有设备信息
        for node in data_dict['data']['nodes']:
            neighbors = list(self.G.neighbors(node['id']))
            if node['label'] == 'bus':
                self._get_bus_info(node, neighbors)
            elif node['label'] == 'load':
                self._get_load_info(node, neighbors)
            elif node['label'] == 'gen':
                self._get_gen_info(node, neighbors)
            elif node['label'] == 'line':
                self._get_line_info(node, neighbors)
            elif node['label'] == 'trafo':
                self._get_trafo_info(node, neighbors)
            elif node['label'] == 'switch':
                self._get_switch_info(node, neighbors)
            else:
                raise NotImplementedError(f'NOT IMPLEMENTED: {node["id"]}')

        # switch的element需要通过图来分析，只能在第二遍的时候处理
        for node in self.nodes_info:
            if 'switch' in node:
                self.nodes_info[node]['element'] = self.dev_id[self.nodes_info[node]['element']]

        if save_network:
            raise(NotImplementedError('Out of date, need to revise'))
            # self.to_json_file(data_path, self.js_format_dict, relationship, self.nodes_info)

        # 加信息标记
        station, feeder, corresponding_load_name = self._get_feeder_info(data_dict)
        for node in self.nodes_info:
            self.nodes_info[node]['net_type'] = 'distribution'
            self.nodes_info[node]['feeder'] = feeder
            self.nodes_info[node]['station'] = station
        info = dict()
        info['corresponding_load_name'] = corresponding_load_name
        info['feeder'] = feeder
        info['station'] = station

        return relationship, self.nodes_info, info
    
    def _get_adjacent_bus(self, node):
        '''
        获取node的相邻母线，如果node已经是母线，则返回母线本身
        '''
        if 'bus' in node:
            return node
        neighbors = list(self.G.neighbors(node))
        for neighbor in neighbors:
            if 'bus' in neighbor:
                return neighbor
        return None

    def _get_adjacent_bus_id(self, node):
        '''
        获取node的相邻母线id，如果node已经是母线，则返回母线id
        '''
        return self.bus_id[self._get_adjacent_bus(node)]
    
    def _get_bus_info(self, node, neighbors):
        '''
        获取母线信息，目前只取pms里的一个，有名字的一个
        '''
        self.bus_id[node['id']] = self.id_counter['bus']
        self.bus_vn[node['id']] = node['param']['vnKv']

        sub_node = node['pmsInfo'][0]

        vn_kv_set = set()
        for sn in node['pmsInfo']:
            if sn.get('name', '') != '':
                sub_node = sn
                break
   
        self.nodes_info[node['id']] = {
            # 通用属性
            'id': self.id_counter['bus'],
            'name': sub_node.get('name', ''),
            'type': 'bus',
            'psr_id': sub_node['psrId'],
            'region_name': self.region_name,
            # 特有属性
            'vn_kv': node['param']['vnKv'],
        }
        self.dev_id[node['id']] = self.id_counter['bus']
        self.id_counter['bus'] += 1
    
    def _get_load_info(self, node, neighbors):
        '''
        获取负荷信息，目前只取pms里的一个，有名字的一个
        '''
        sub_node = node['pmsInfo'][0]
        for sn in node['pmsInfo']:
            if sn.get('name', '') != '':
                sub_node = sn
                break

        self.nodes_info[node['id']] = {
            # 通用属性
            'id': self.id_counter['load'],
            'name': sub_node.get('name', ''),
            'type': 'load',
            'region_name': self.region_name,
            'psr_id': sub_node['psrId'],
            # 特有属性
            'p_mw': np.array(node['yc']['pMw'])[self.time_series_idx],
            'q_mvar': np.array(node['yc']['qMvar'])[self.time_series_idx],
            'bus': self._get_adjacent_bus_id(node['id']),
        }
        self.dev_id[node['id']] = self.id_counter['load']
        self.id_counter['load'] += 1

    def _get_gen_info(self, node, neighbors):
        '''
        获取发电机信息，目前只取pms里的一个，有名字的一个
        '''
        sub_node = node['pmsInfo'][0]
        for sn in node['pmsInfo']:
            if sn.get('name', '') != '':
                sub_node = sn
                break

        self.nodes_info[node['id']] = {
            # 通用属性
            'id': self.id_counter['gen'],
            'name': sub_node.get('name', ''),
            'type': 'gen',
            'region_name': self.region_name,
            'psr_id': sub_node['psrId'],
            # 特有属性
            'p_mw': np.array(node['yc']['pMw'])[self.time_series_idx],  # [B[i] for i in A]
            'q_mvar': np.array(node['yc']['qMvar'])[self.time_series_idx],
            'bus': self._get_adjacent_bus_id(node['id']),
            'sn_mva': node['param']['snMva'],
            'slack': True,  # 配网应该就一个gen，就建模为slack好了。
            # 'slack': node['param']['control'] == 'Slack',
            'vm_pu': np.array(node['yc']['uKv'])[self.time_series_idx] / self.bus_vn[self._get_adjacent_bus(node['id'])],
        }
        self.dev_id[node['id']] = self.id_counter['gen']
        self.id_counter['gen'] += 1

    def _get_line_info(self, node, neighbors):
        '''
        获取线路信息，目前只取pms里的一个，有名字的一个
        '''
        line_length = node['param']['length'] / 1000  # 原始数据是m!!!! 这里要的是km，差了1000倍
        # line_length = 0
        single_line_length = 0
        for sub_node in node['pmsInfo']:
            # line_length += sub_node['length']
            # single_line_length = sub_node['length']
            # line_length = sub_node['length']
            if sub_node.get('name', '') != '':
                sub_node = sub_node
                
        r_ohm_per_km = node['param']['r']
        x_ohm_per_km = node['param']['x']
        c_nf_per_km = node['param']['c'] / 1000  # 原始数据是pf!!!! 这里要的是nf，差了1000倍
        # if c_per_km == 0:
        #     c_per_km = 1
        # if single_line_length < line_length:
        #     r_per_km, x_per_km, c_per_km = self.long_line_pi_equiv(r_per_km, x_per_km, c_per_km, 50, line_length)

        self.nodes_info[node['id']] = {
            # 通用属性
            'id': self.id_counter['line'],
            'name': sub_node.get('name', ''),
            'type': 'line',
            'psr_id': sub_node['psrId'],
            'region_name': self.region_name,
            # 特有属性
            'max_i_ka': sub_node['zll'],
            'r_ohm_per_km': r_ohm_per_km,
            'x_ohm_per_km': x_ohm_per_km,
            'c_nf_per_km': c_nf_per_km,
            'from_bus': self._get_adjacent_bus_id(neighbors[0]),
            'to_bus': self._get_adjacent_bus_id(neighbors[1]),
            'length_km': line_length,
        }
        self.dev_id[node['id']] = self.id_counter['line']
        self.id_counter['line'] += 1


    def _get_trafo_info(self, node, neighbors):
        sub_node = node['pmsInfo'][0]
        for sn in node['pmsInfo']:
            if sn.get('name', '') != '':
                sub_node = sn
                break

        if len(neighbors) == 2:
            v1 = self.bus_vn[self._get_adjacent_bus(neighbors[0])]
            v2 = self.bus_vn[self._get_adjacent_bus(neighbors[1])]
            if v1 > v2:
                hv_bus = self._get_adjacent_bus_id(neighbors[0])
                lv_bus = self._get_adjacent_bus_id(neighbors[1])
            else:
                hv_bus = self._get_adjacent_bus_id(neighbors[1])
                lv_bus = self._get_adjacent_bus_id(neighbors[0])

            self.nodes_info[node['id']] = {
                # 通用属性
                'id': self.id_counter['trafo'],
                'name': sub_node.get('name', ''),
                'type': 'trafo',
                'region_name': self.region_name,
                'psr_id': sub_node['psrId'],
                # 特有属性
                'hv_bus': hv_bus,
                'lv_bus': lv_bus,
                'sn_mva': node['param']['snMva'],
                'vk_percent': node['param']['impedanceVoltage'],
                'vkr_percent': node['param']['shortCircuitLoss'] * 100 / (node['param']['snMva'] * 1000),
                'tap_pos': node['param']['tapPos'],
                'tap_neutral': node['param']['tapNeutral'],
                'tap_max': node['param']['tapMax'],
                'tap_min': node['param']['tapMin'],
                'tap_step_percent': node['param']['tapStepPercent'],
                'pfe_kw': node['param']['noLoadLoss'],
                'i0_percent': node['param']['noLoadCurrent'],
                'tap_side': node['param']['tapSide'],
                'vn_lv_kv': node['param']['vnLvKv'],
                'vn_hv_kv': node['param']['vnHvKv'],
            }
            self.dev_id[node['id']] = self.id_counter['trafo']
            self.id_counter['trafo'] += 1
    
    def _get_switch_info(self, node, neighbors):
        sub_node = node['pmsInfo'][0]
        for sn in node['pmsInfo']:
            if sn.get('name', '') != '' and sn.get('isZdhkg', True):
                sub_node = sn
                break

        element = None
        switch_bus = None
        if len(neighbors) == 1:
            switch_bus = self._get_adjacent_bus_id(neighbors[0])
            et = 'b'
            element = self._get_adjacent_bus(neighbors[0])
        else:
            for neighbor in neighbors:
                if 'line' in neighbor:
                    switch_bus = self._get_adjacent_bus_id(neighbor)
                    et = 'l'
                    element = neighbor  # 这里需要改成设备id
                    break
                elif 'trafo' in neighbor:
                    switch_bus = self._get_adjacent_bus_id(neighbor)
                    et = 't'
                    element = neighbor  # 这里需要改成设备id
                    break
        if element is None:
            # 如果两端都是bus，就让一侧是bus，一侧是element
            switch_bus = self._get_adjacent_bus_id(neighbors[0])
            et = 'b'
            element = neighbors[1]

        self.nodes_info[node['id']] = {
            # 通用属性
            'id': self.id_counter['switch'],
            'name': sub_node.get('name', ''),
            'type': 'switch',
            'region_name': self.region_name,
            'psr_id': sub_node['psrId'],
            # 特有属性
            'bus': switch_bus,
            'element': element,
            'et': et,
            'closed': node['param']['closed'],
            'is_zdhkg': sub_node['isZdhkg'],
        }
        self.dev_id[node['id']] = self.id_counter['switch']
        self.id_counter['switch'] += 1

    def long_line_pi_equiv(self, r_per_km, x_per_km, c_per_km, f, L_km, g_per_km=0.0):
        w = 2*np.pi*f
        z = (r_per_km + 1j*x_per_km)
        y = g_per_km + 1j*w*c_per_km * 1e-9
        L = L_km
        gamma = np.sqrt(z*y)
        Zc = np.sqrt(z/y)
        A = np.cosh(gamma*L)
        B = Zc*np.sinh(gamma*L)
        Z_series_eq = B/A
        Y_sh_total = 2*(A-1)/B
        r_new = Z_series_eq.real / L_km
        x_new = Z_series_eq.imag / L_km
        c_new = Y_sh_total.imag / (L_km * w) * 1e9
        return r_new, x_new, c_new

    def to_json_file(self, data_path, js_format_dict, relationship, nodes_info):
        """
        Out of date, need to revise
        """
        data_dir = os.path.dirname(data_path)
        data_name = os.path.basename(data_path)
        res_dict = {
            'js_format_dict': js_format_dict,
            'relationship': relationship,
            'nodes_info': nodes_info
        }
        if not os.path.exists(os.path.join(data_dir, 'parsered')):
            os.makedirs(os.path.join(data_dir, 'parsered'))
        with open(os.path.join(data_dir, 'parsered', data_name), 'w', encoding='utf-8') as f:
            json.dump(res_dict, f, ensure_ascii=False, indent=4)

    def _find_closest_time_point(self, start:datetime, end:datetime, count:datetime, ref:datetime):
        # 边界情况处理
        if count <= 1:
            return start, 0
        
        # 1. 计算总时长的秒数 (或者微秒，视精度要求而定)
        total_duration = (end - start).total_seconds()
        
        # 2. 计算每个点之间的步长 (Interval)
        # n个点有 n-1 个间隔
        step_seconds = total_duration / (count - 1)
        
        # 3. 计算 ref_time 距离 start 的秒数
        ref_delta = (ref - start).total_seconds()
        
        # 4. 计算对应的索引 (四舍五入 round 即为寻找最近)
        # 比如落在 2.1 步，最近的是 2；落在 2.8 步，最近的是 3
        estimated_idx = round(ref_delta / step_seconds)
        
        # 5. 限制索引范围 (防止 ref_time 超出 start 或 end)
        idx = max(0, min(estimated_idx, count - 1))
        
        # 6. 根据索引反推准确的时间点
        # 注意：这里重新加一遍是为了保证时间点完全符合均匀分布的定义
        closest_time = start + timedelta(seconds=idx * step_seconds)
        
        return closest_time, idx

    def _get_feeder_info(self, data_dict):
        prefix = data_dict['data']['dkxTz']['d5000LdName'].split('.')[0] + '.'
        corresponding_load_name = data_dict['data']['dkxTz']['d5000LdName'].removeprefix(prefix)
        station = corresponding_load_name.split('/')[0]
        feeder = corresponding_load_name.split('.')[-1]
        return station, feeder, corresponding_load_name