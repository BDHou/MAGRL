from collections import defaultdict
import json
import math
import numpy as np
import pandapower as pp
import os
from datetime import datetime

class TransmissionNetworkParser:
    '''
    Convert the qs_dict to the pp_dict.
    The elements are dictionaries of the pandapower format.
    The nodes_info is a dictionary of dictionaries, each dictionary contains the information of the node.
    The relationship is a list of tuples, each tuple contains the two nodes that are connected.
    '''

    def parse_network(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            qs_dict = json.load(f)
        relationship, nodes_info, orig_timestamp = self.create_pp_dict_with_network_from_qs_dict(qs_dict)
        format_pattern = "%Y%m%d_%H:%M:%S"
        timestamp = datetime.strptime(orig_timestamp, format_pattern)
        # print(timestamp)
        # if save_network:
        #     self.save_network(pp_dict, relationship, nodes_info, region_name)
        return relationship, nodes_info, timestamp
    
    def create_pp_dict_with_network_from_qs_dict(self, qs_dict):
        '''
        Args:
            qs_dict: the dictionary of the qs_dict.
        Returns:
            relationship: the list of the relationship.
            nodes_info: the dictionary of the nodes_info.
            timestamp: the timestamp of the current powerflow_data
        '''
        self.bus_nd_dict = {}
        self.relationship = list()
        self.nodes_info = dict()
        self.element_counter = defaultdict(int)

        if 'Bus' in qs_dict['data']:
            self._get_bus_info(qs_dict['data']['Bus'])
        if 'Load' in qs_dict['data']:
            self._get_load_info(qs_dict['data']['Load'])
        if 'Unit' in qs_dict['data']:
            self._get_gen_info(qs_dict['data']['Unit'])
        if 'Compensator_P' in qs_dict['data']:
            self._get_shunt_info(qs_dict['data']['Compensator_P'])
        if 'ACline' in qs_dict['data']:
            self._get_line_info(qs_dict['data']['ACline'])
        if 'Transformer' in qs_dict['data']:
            self._get_trafo_info(qs_dict['data']['Transformer'])
        if 'Disconnector' in qs_dict['data'] or 'Breaker' in qs_dict['data']:
            self._get_switch_info(qs_dict['data']['Disconnector'], qs_dict['data']['Breaker'])
        # 加上主网标记
        for node in self.nodes_info:
            self.nodes_info[node]['net_type'] = 'transmission'
        return self.relationship, self.nodes_info, qs_dict['data']['time']
    
    def generate_pp_format_json(self, data_name, data_dir):
        '''
        Broken
        '''
        trans_path = os.path.join(data_dir, f"{data_name}.txt")
        with open(trans_path, "r", encoding='utf-8') as f:
            qs_dict = json.load(f)

        pp_dict, relationship, nodes_info = self.create_pp_dict_with_network_from_qs_dict(qs_dict)

        # save pp_dict
        res_path = os.path.join(data_dir, "tmp", f"{data_name}_pp_dict.json")
        os.makedirs(os.path.join(data_dir, "tmp"), exist_ok=True)
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(pp_dict, f, ensure_ascii=False, indent=4)
    
        topo_dict = dict(
            relationship=relationship,
            nodes=nodes_info,
        )

        with open(os.path.join(data_dir, "tmp", f"{data_name}_topo.json"), 'w', encoding='utf-8') as f:
            json.dump(topo_dict, f, ensure_ascii=False, indent=4)

    #########################################################
    # 以下为解析各个设备信息的方法
    #########################################################

    def _get_bus_info(self, data_dict):
        # parse buses
        for bus in data_dict:
            self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                # 通用属性
                psr_id=bus['busid'],
                name=bus['name'],
                nd=bus['nd'],
                type='bus',
                id=self.element_counter["bus"],
                in_service=(bus['off'] == 0),
                # 特有属性
                vn_kv=float(bus['volt']),
                vm_pu=bus['v'] / float(bus['volt']),  # Notice that is p.u. voltage
                va_degree=bus['ang'],
            )
            self.bus_nd_dict[bus['nd']] = self.element_counter["bus"]
            self.element_counter["bus"] += 1

    def _get_load_info(self, data_dict):
        # parse loads
        for load in data_dict:
            if load['nd'] not in self.bus_nd_dict:
                # create virtual bus for load first
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=load['name']+'virtual',
                    nd=load['nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=(load['off'] == 0),
                    # 特有属性
                    vn_kv=float(load['volt']),
                )
                self.bus_nd_dict[load['nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1

            # then attach load to the virtual bus
            self.nodes_info[f'load {self.element_counter["load"]}'] = dict(
                # 通用属性
                psr_id=load['loadid'],
                bus=self.bus_nd_dict[load['nd']],
                name=load['name'],
                nd=load['nd'],
                type='load',
                id=self.element_counter["load"],
                in_service=(load['off'] == 0),
                # 特有属性
                p_mw=load['P'],  # 25-9-27 暂时搞成负的试试看
                q_mvar=load['Q'],  # 25-9-27 暂时搞成负的试试看
                # p_mw=-load['P'],  # 25-9-27 暂时搞成负的试试看
                # q_mvar=-load['Q'],  # 25-9-27 暂时搞成负的试试看
            )
            self.relationship.append((f'load {self.element_counter["load"]}', f'bus {self.bus_nd_dict[load["nd"]]}'))
            self.element_counter["load"] += 1

    def _get_gen_info(self, data_dict):
        # parse gens
        for gen in data_dict:
            if gen['nd'] not in self.bus_nd_dict:
                # create virtual bus for load first
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=gen['name']+'virtual',
                    nd=gen['nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=(gen['off'] == 0),
                    # 特有属性
                    vn_kv=float(gen['V_Rate']),
                )
                self.bus_nd_dict[gen['nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1

            # then attach load to the virtual bus
            if '光伏' in gen['name'] or '风电' in gen['name']:  # 有逆变器的设备应该被建模为sgen
                self.nodes_info[f'sgen {self.element_counter["sgen"]}'] = dict(
                    # 通用属性
                    psr_id=gen['unitid'],
                    bus=self.bus_nd_dict[gen['nd']],
                    name=gen['name'],
                    nd=gen['nd'],
                    type='sgen',
                    id=self.element_counter["sgen"],
                    in_service=(gen['off'] == 0),
                    # 特有属性
                    p_mw=gen['P'],
                    q_mvar=gen['Q'],
                    max_p_mw=gen['P_max'],
                    min_p_mw=gen['P_min'],
                    max_q_mvar=gen['Q_max'],
                    min_q_mvar=gen['Q_min']
                )
                self.relationship.append((f'sgen {self.element_counter["sgen"]}', f'bus {self.bus_nd_dict[gen["nd"]]}'))
                self.element_counter["sgen"] += 1
            elif '线' in gen['name'][-2:]:  # 25-9-25: 线性负载应该被建模为load
                self.nodes_info[f'load {self.element_counter["load"]}'] = dict(
                    # 通用属性
                    psr_id=gen['unitid'],
                    bus=self.bus_nd_dict[gen['nd']],
                    name=gen['name'],
                    nd=gen['nd'],
                    type='load',
                    id=self.element_counter["load"],
                    in_service=(gen['off'] == 0),
                    # 特有属性
                    p_mw=gen['P'],
                    q_mvar=gen['Q']
                )
                self.relationship.append((f'load {self.element_counter["load"]}', f'bus {self.bus_nd_dict[gen["nd"]]}'))
                self.element_counter["load"] += 1
            else:
                self.nodes_info[f'gen {self.element_counter["gen"]}'] = dict(
                    # 通用属性
                    psr_id=gen['unitid'],
                    bus=self.bus_nd_dict[gen['nd']],
                    name=gen['name'],
                    nd=gen['nd'],
                    type='gen',
                    id=self.element_counter["gen"],
                    in_service=(gen['off'] == 0),
                    # 特有属性
                    p_mw=gen['P'],
                    vn_kv=float(gen['V_Rate']),
                    vm_pu=float(gen['Ue'])/float(gen['V_Rate']),
                    max_p_mw=gen['P_max'],
                    min_p_mw=gen['P_min'],
                    max_q_mvar=gen['Q_max'],
                    min_q_mvar=gen['Q_min']
                )
                self.relationship.append((f'gen {self.element_counter["gen"]}', f'bus {self.bus_nd_dict[gen["nd"]]}'))
                self.element_counter["gen"] += 1


    def _get_shunt_info(self, data_dict):
        # parse shunts
        cnt = 0
        for shunt in data_dict:
            if shunt['nd'] in self.bus_nd_dict:
                cnt += 1
            else:
                # create virtual bus for shunt first
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=shunt['name']+'virtual',
                    nd=shunt['nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    # 特有属性
                    vn_kv=float(float(shunt['volt'])),
                )
                self.bus_nd_dict[shunt['nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1

            # then attach load to the virtual bus
            self.nodes_info[f'shunt {self.element_counter["shunt"]}'] = dict(
                # 通用属性
                psr_id=shunt['compensatorid'],
                bus=self.bus_nd_dict[shunt['nd']],
                name=shunt['name'],
                nd=shunt['nd'],
                type='shunt',
                id=self.element_counter["shunt"],
                in_service=(shunt['off'] == 0),
                # 特有属性
                # p_mw=-float(shunt['P']),  # 25-9-27 暂时搞成负的试试看
                # q_mvar=-shunt['Q'],  # 25-9-27 暂时搞成负的试试看                
                p_mw=float(shunt['P']),  # 25-9-27 暂时搞成负的试试看
                q_mvar=-float(shunt['Q']),  # 25-9-29 得搞成负的，pandapower里认为shunt是消耗无功的，qs里是电容器啊
            )
            self.relationship.append((f'shunt {self.element_counter["shunt"]}', f'bus {self.bus_nd_dict[shunt["nd"]]}'))
            self.element_counter["shunt"] += 1


    def _get_line_info(self, data_dict):
        def c_from_x(x_ohm_per_km, f_hz=50.0, zc_ohm=350.0):
            '''
            用x计算c，单位是nF/km
            '''
            L = x_ohm_per_km / (2*math.pi*f_hz)    # H/km
            C = L / (zc_ohm**2)                    # F/km
            return C * 1e9     
        # parse lines
        for line in data_dict:

            i_nd = line['I_nd']
            j_nd = line['J_nd']

            # create virtual bus for each end
            if i_nd not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=line['I_node'],
                    nd=line['I_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=(line['I_off'] == 0),
                    # 特有属性
                    vn_kv=float(float(line['volt'])), 
                    p_mw=line['I_P'],  # 这样的值可能会不准的
                    q_mvar=line['I_Q'],
                )
                self.bus_nd_dict[line['I_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            else:  # 25-9-25 如果bus已经存在，则更新其属性
                self.nodes_info[f'bus {self.bus_nd_dict[i_nd]}'].update(
                    in_service=(line['I_off'] == 0),
                    p_mw=line['I_P'],
                    q_mvar=line['I_Q'],
                )
            if j_nd not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    name=line['J_node'],
                    nd=line['J_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=(line['J_off'] == 0),
                    # 特有属性
                    vn_kv=float(float(line['volt'])),
                    p_mw=line['J_P'],  # 25-9-27 暂时搞成负的试试看, to侧的应该是流出吧
                    q_mvar=line['J_Q'],  # 25-9-27 暂时搞成负的试试看, to侧的应该是流出吧
                )
                self.bus_nd_dict[line['J_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            else:  # 25-9-25 如果bus已经存在，则更新其属性
                self.nodes_info[f'bus {self.bus_nd_dict[j_nd]}'].update(
                    in_service=(line['J_off'] == 0),
                    p_mw=line['J_P'],  # 25-9-27 暂时搞成负的试试看, to侧的应该是流出吧
                    q_mvar=line['J_Q'],  # 25-9-27 暂时搞成负的试试看, to侧的应该是流出吧
                )
            # connect line to the virtual bus
            self.nodes_info[f'line {self.element_counter["line"]}'] = dict(
                # 通用属性
                psr_id=line['AClineid'],
                name=line['name'],
                i_nd=line['I_nd'],
                j_nd=line['J_nd'],
                type='line',
                id=self.element_counter["line"],
                # 特有属性
                from_bus=self.bus_nd_dict[line['I_nd']],
                to_bus=self.bus_nd_dict[line['J_nd']],
                r_ohm_per_km=line['R'] * 1000,  # 25-8-27 原来的值实在太小了，先试试kOhm
                x_ohm_per_km=line['X'],
                c_nf_per_km=c_from_x(line['X']),  # 尝试用x计算c
                max_i_ka=line['Ih'] / 1000,
                length_km=1.0,
                # 参考原始值
                p_from_mw_ref=line['I_P'],  # 25-10-04 用来给边界精准等值
                p_to_mw_ref=line['J_P'],
                q_from_mvar_ref=line['I_Q'],
                q_to_mvar_ref=line['J_Q'],
            )
            self.relationship.append((f'bus {self.bus_nd_dict[line["I_nd"]]}', f'line {self.element_counter["line"]}'))
            self.relationship.append((f'line {self.element_counter["line"]}', f'bus {self.bus_nd_dict[line["J_nd"]]}'))

            self.element_counter["line"] += 1

    def _get_trafo_info(self, data_dict):
        for trafo in data_dict:
            i_nd = trafo['I_nd']
            j_nd = trafo['J_nd']
            k_nd = trafo['K_nd']
            is_3w = not (k_nd == "-1")
            # 如果off为1，则暂时认为该变压器不投入运行
            i_in_service = (trafo['I_off'] == 0) and trafo['Itap_V'] != 0
            j_in_service = (trafo['J_off'] == 0) and trafo['Jtap_V'] != 0
            k_in_service = (trafo['K_off'] == 0) and trafo['Ktap_V'] != 0
            # 暂时把不在线的也保留着
            # if not is_3w:
            #     if (not i_in_service) or (not j_in_service):
            #         continue
            # else:
            #     if (not i_in_service) or (not j_in_service) or (not k_in_service):
            #         continue

            # create virtual bus for each end
            if i_nd not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=trafo['I_node'],
                    nd=trafo['I_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=i_in_service,
                    # 特有属性
                    vn_kv=float(float(trafo['I_Vol'])),
                    p_mw=trafo['I_P'],
                    q_mvar=trafo['I_Q'],
                )
                self.bus_nd_dict[trafo['I_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            else:  # 25-9-25 如果bus已经存在，则更新其属性
                self.nodes_info[f'bus {self.bus_nd_dict[i_nd]}'].update(
                    in_service=i_in_service,
                    p_mw=trafo['I_P'],
                    q_mvar=trafo['I_Q'],
                )
            if j_nd not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=trafo['J_node'],
                    nd=trafo['J_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=j_in_service,
                    # 特有属性
                    vn_kv=float(float(trafo['J_Vol'])),
                    p_mw=trafo['J_P'],  # 25-9-27 暂时搞成负的试试看,中低压侧应该默认是流出吧？不能这么简单的判断，要根据方向来！！！
                    q_mvar=trafo['J_Q'],  # 25-9-27 暂时搞成负的试试看,中低压侧应该默认是流出吧
                )
                self.bus_nd_dict[trafo['J_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            else:  # 25-9-25 如果bus已经存在，则更新其属性
                self.nodes_info[f'bus {self.bus_nd_dict[j_nd]}'].update(
                    in_service=j_in_service,
                    p_mw=trafo['J_P'],
                    q_mvar=trafo['J_Q'],
                )
            if k_nd not in self.bus_nd_dict and is_3w:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    # 通用属性
                    name=trafo['K_node'],
                    nd=trafo['K_nd'],   
                    type='bus',
                    id=self.element_counter["bus"],
                    in_service=k_in_service,
                    # 特有属性
                    vn_kv=float(float(trafo['K_Vol'])),
                    p_mw=trafo['K_P'],
                    q_mvar=trafo['K_Q'],
                )
                self.bus_nd_dict[trafo['K_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            elif is_3w:  # 25-9-25 如果bus已经存在，则更新其属性
                self.nodes_info[f'bus {self.bus_nd_dict[k_nd]}'].update(
                    in_service=k_in_service,
                    p_mw=trafo['K_P'],
                    q_mvar=trafo['K_Q'],
                )


            if not is_3w:
                vk_percent, vkr_percent, pfe_kw, i0_percent = self._get_transformer_params(trafo, _print=False)
                self.relationship.append((f'bus {self.bus_nd_dict[trafo["I_nd"]]}', f'trafo {self.element_counter["trafo"]}'))
                self.relationship.append((f'trafo {self.element_counter["trafo"]}', f'bus {self.bus_nd_dict[trafo["J_nd"]]}'))
                self.nodes_info[f'trafo {self.element_counter["trafo"]}'] = dict(
                    # 通用属性
                    psr_id=trafo['transformerid'],
                    name=trafo['name'],
                    i_nd=trafo['I_nd'],
                    j_nd=trafo['J_nd'],
                    type='trafo',
                    id=self.element_counter["trafo"],
                    # 特有属性
                    hv_bus=self.bus_nd_dict[trafo['I_nd']],
                    lv_bus=self.bus_nd_dict[trafo['J_nd']],
                    sn_mva=trafo['I_S'],
                    vn_hv_kv=float(trafo['Itap_V']),  # 25-9-27 高压侧先用tap_V, 这个确实是额定电压呀
                    # vn_lv_kv=float(trafo['Jtap_V']),  # 25-9-27 低压侧先用tap_V
                    # vn_hv_kv=float(trafo['I_Vol']),  # 25-9-25 更新为I_Vol
                    vn_lv_kv=float(trafo['J_Vol']), 
                    # - need calculation - 
                    vk_percent=vk_percent,
                    vkr_percent=vkr_percent,
                    pfe_kw=pfe_kw,
                    i0_percent=i0_percent,
                    tap_side='hv',  # 暂时先这么设定，后续再改
                    tap_pos=trafo['I_tap'],
                    tap_neutral=trafo['Itap_E'],
                    tap_max=trafo['Itap_H'],
                    tap_min=trafo['Itap_L'],
                    tap_step_percent=trafo['Itap_C'] * 100,  # 25-9-25 单位是percent，所以要乘100
                    # 参考原始值
                    p_hv_mw_ref=trafo['I_P'],
                    p_lv_mw_ref=trafo['J_P'],
                    q_hv_mvar_ref=trafo['I_Q'],
                    q_lv_mvar_ref=trafo['J_Q'],
                    in_service=i_in_service and j_in_service,
                )
                self.element_counter["trafo"] += 1

            else:
                vk_hv_percent, vk_mv_percent, vk_lv_percent, vkr_hv_percent, vkr_mv_percent, vkr_lv_percent, pfe_kw, i0_percent = self._get_transformer3w_params(trafo, _print=False)
                self.relationship.append((f'bus {self.bus_nd_dict[trafo["I_nd"]]}', f'trafo3w {self.element_counter["trafo3w"]}'))
                self.relationship.append((f'trafo3w {self.element_counter["trafo3w"]}', f'bus {self.bus_nd_dict[trafo["J_nd"]]}'))
                self.relationship.append((f'trafo3w {self.element_counter["trafo3w"]}', f'bus {self.bus_nd_dict[trafo["K_nd"]]}'))
                self.nodes_info[f'trafo3w {self.element_counter["trafo3w"]}'] = dict(
                    # 通用属性
                    psr_id=trafo['transformerid'],
                    name=trafo['name'],
                    i_nd=trafo['I_nd'],
                    j_nd=trafo['J_nd'],
                    k_nd=trafo['K_nd'],
                    type='trafo3w',
                    id=self.element_counter["trafo3w"],
                    # 特有属性
                    hv_bus=self.bus_nd_dict[trafo['I_nd']],
                    mv_bus=self.bus_nd_dict[trafo['J_nd']],
                    lv_bus=self.bus_nd_dict[trafo['K_nd']],
                    vn_hv_kv=float(trafo['Itap_V']),  
                    # vn_mv_kv=float(trafo['Jtap_V']),  # 25-9-27 中压侧先用tap_V
                    # vn_lv_kv=float(trafo['Ktap_V']),  # 25-9-27 低压侧先用tap_V
                    # vn_hv_kv=float(trafo['I_Vol']),  # 25-9-25 更新为I_Vol
                    vn_mv_kv=float(trafo['J_Vol']),
                    vn_lv_kv=float(trafo['K_Vol']),
                    sn_hv_mva=trafo['I_S'],
                    sn_mv_mva=trafo['J_S'],
                    sn_lv_mva=trafo['K_S'],
                    # - need calculation - 
                    vk_hv_percent=vk_hv_percent,
                    vk_mv_percent=vk_mv_percent,
                    vk_lv_percent=vk_lv_percent,
                    vkr_hv_percent=vkr_hv_percent,
                    vkr_mv_percent=vkr_mv_percent,
                    vkr_lv_percent=vkr_lv_percent,
                    pfe_kw=pfe_kw,
                    i0_percent=i0_percent,
                    tap_max=trafo['Itap_H'],  # 缺失了Ktap的信息，后续得考虑一下建模了
                    tap_min=trafo['Itap_L'],
                    tap_neutral=trafo['Itap_E'],
                    tap_step_percent=trafo['Itap_C'] * 100,  # 25-9-25 单位是percent，所以要乘100
                    tap_side='hv',  # 暂时先这么设定，后续再改
                    tap_pos=trafo['I_tap'],
                    # 参考原始值
                    p_hv_mw_ref=trafo['I_P'],
                    p_mv_mw_ref=trafo['J_P'],
                    p_lv_mw_ref=trafo['K_P'],
                    q_hv_mvar_ref=trafo['I_Q'],
                    q_mv_mvar_ref=trafo['J_Q'],
                    q_lv_mvar_ref=trafo['K_Q'],
                    in_service=i_in_service and j_in_service and k_in_service,
                )
                self.element_counter["trafo3w"] += 1

    def _get_switch_info(self, disconnector_data_dict, breaker_data_dict):
        self._get_disconnector_info(disconnector_data_dict)
        self._get_breaker_info(breaker_data_dict)

    def _get_disconnector_info(self, data_dict, _print=False):
        for breaker in data_dict:
            if breaker['I_nd'] not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    name=breaker['I_node'],
                    nd=breaker['I_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    # 特有属性
                    vn_kv=float(float(breaker['volt'])), 
                )
                self.bus_nd_dict[breaker['I_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            if breaker['J_nd'] not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    name=breaker['J_node'],
                    nd=breaker['J_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    # 特有属性
                    vn_kv=float(float(breaker['volt'])), 
                )
                self.bus_nd_dict[breaker['J_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1

            i_nd = self.bus_nd_dict[breaker['I_nd']]
            j_nd = self.bus_nd_dict[breaker['J_nd']]

            if _print:
                print(f'breaker {self.element_counter["switch"]}: {i_nd} {j_nd}')

            self.relationship.append((f'bus {i_nd}', f'switch {self.element_counter["switch"]}'))
            self.relationship.append((f'switch {self.element_counter["switch"]}', f'bus {j_nd}'))
            self.nodes_info[f'switch {self.element_counter["switch"]}'] = dict(
                # 通用属性
                psr_id=breaker['disconnectorid'],
                name=breaker['name'],
                i_nd=breaker['I_nd'],
                j_nd=breaker['J_nd'],   
                type='switch',
                id=self.element_counter["switch"],
                # 特有属性
                closed=True if breaker['point'] == 1 else False,
                is_zdhkg=False,
                bus=self.bus_nd_dict[breaker['I_nd']],
                element=self.bus_nd_dict[breaker['J_nd']],  # 不到啊
                et='b',  # 不到啊
            )
            self.element_counter["switch"] += 1

    def _get_breaker_info(self, data_dict):
        _print = False
        for disconnector in data_dict:
            if disconnector['I_nd'] not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    name=disconnector['I_node'],
                    nd=disconnector['I_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    # 特有属性
                    vn_kv=float(float(disconnector['volt'])), 
                )
                self.bus_nd_dict[disconnector['I_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1
            if disconnector['J_nd'] not in self.bus_nd_dict:
                self.nodes_info[f'bus {self.element_counter["bus"]}'] = dict(
                    name=disconnector['J_node'],
                    nd=disconnector['J_nd'],
                    type='bus',
                    id=self.element_counter["bus"],
                    # 特有属性
                    vn_kv=float(float(disconnector['volt'])), 
                )
                self.bus_nd_dict[disconnector['J_nd']] = self.element_counter["bus"]
                self.element_counter["bus"] += 1

            i_nd = self.bus_nd_dict[disconnector['I_nd']]
            j_nd = self.bus_nd_dict[disconnector['J_nd']]

            if _print:
                print(f'disconnector {self.element_counter["switch"]}: {i_nd} {j_nd}')

            self.relationship.append((f'bus {i_nd}', f'switch {self.element_counter["switch"]}'))
            self.relationship.append((f'switch {self.element_counter["switch"]}', f'bus {j_nd}'))
            self.nodes_info[f'switch {self.element_counter["switch"]}'] = dict(
                psr_id=disconnector['breakeid'],
                name=disconnector['name'],
                i_nd=disconnector['I_nd'],
                j_nd=disconnector['J_nd'],   
                type='switch',
                id=self.element_counter["switch"],
                # 特有属性
                closed=True if disconnector['point'] == 1 else False,
                is_zdhkg=True,
                bus=self.bus_nd_dict[disconnector['I_nd']],
                element=self.bus_nd_dict[disconnector['J_nd']],  # 不到啊
                et='b',  # 不到啊
            )
            self.element_counter["switch"] += 1

            
    # parse transformers
    def _get_transformer_params(self, trafo, _print=False):
        un_hv = trafo['Itap_V'] 
        un_lv = trafo['Jtap_V']
        sn = trafo['I_S']
        z_base = un_hv**2 / sn
        r = trafo['Ri']
        x = trafo['Xi']
        if _print:
            print(f'un_hv, sn, z_base, r, x: {un_hv}, {sn}, {z_base}, {r}, {x}')

        r_pu = r / z_base
        x_pu = x / z_base
        z_pu = np.sqrt(r_pu**2 + x_pu**2)
        vk_percent = z_pu * 100
        vkr_percent = r_pu * 100
        if _print:
            print(f'vk_percent, vkr_percent: {vk_percent}, {vkr_percent}')

        g = trafo['G']
        b = trafo['B']
        if _print:
            print(f'g, b: {g}, {b}')

        # 暂时设定un为1 MVA，但是感觉不大合理，后续再改
        un = 1
        pfe_kw = trafo['P0'] * 1000 # 25-9-22 改用P0
        # pfe_kw = 1000 * g * un **2

        i0_percent = 0.5  # 25-9-22 暂时先不要了，和金山变一样
        # i0_percent = abs(100 * b * un ** 2 / sn)  # 25-9-22 这样有可能会算出非常离谱的值
        # if i0_percent > 10:
        #     i0_percent = 10  # 25-9-22 设定一个上限吧，后续再考虑考虑

        if _print:
            print(f'i0_percent: {i0_percent}')

        return vk_percent, vkr_percent, pfe_kw, i0_percent
            
    def _get_transformer3w_params(self, trafo, _print=False):
        """
        根据T型等效电路参数计算并返回pandapower三绕组变压器所需的短路参数。

        核心逻辑：
        1. 识别高(HV)、中(MV)、低(LV)压侧绕组，并提取各自的额定参数。
        2. 提取各绕组在自身电压等级下的物理阻抗（Ohm）。
        3. 设定一个统一的计算基准（通常是高压侧的电压和容量）。
        4. 将中、低压侧的物理阻抗“归算”到高压侧基准下。
        5. 使用归算后的阻抗计算三对绕组（HV-MV, HV-LV, MV-LV）的短路阻抗。
        6. 将短路阻抗转换为pandapower所需的百分比形式。
        7. 提取空载损耗等其他参数。

        Args:
            trafo (dict): 包含变压器原始参数的字典。
            _print (bool, optional): 是否打印计算结果用于调试。 Defaults to False.

        Returns:
            tuple: 包含pandapower所需参数的元组 (vk_hv_percent, vk_mv_percent, 
                vk_lv_percent, vkr_hv_percent, vkr_mv_percent, vkr_lv_percent, 
                pfe_kw, i0_percent)
        """
        # --- 1. 识别绕组并提取基本参数 ---
        vn_hv_kv = trafo['Itap_V']    
        vn_mv_kv = trafo['Jtap_V']   
        vn_lv_kv = trafo['Ktap_V']   

        sn_hv_mva = trafo['I_S']
        # 假设三个绕组容量相同，取HV侧为准
        sn_mva = sn_hv_mva 

        # 提取T型等效电路各支路的物理阻抗（在各自电压等级下）
        r_hv = trafo['Ri']
        x_hv = trafo['Xi']
        r_mv_own_base = trafo['Rj'] 
        x_mv_own_base = trafo['Xj']
        r_lv_own_base = trafo['Rk'] 
        x_lv_own_base = trafo['Xk']

        # --- 2. 设定统一计算基准 (高压侧) ---
        z_base_hv = vn_hv_kv**2 / sn_mva

        # --- 3. 将所有阻抗归算到高压侧 ---
        # HV侧阻抗无需归算
        r_hv_ref = r_hv
        x_hv_ref = x_hv

        # MV侧阻抗归算到HV侧
        n_hv_mv = vn_hv_kv / vn_mv_kv
        r_mv_ref_hv = r_mv_own_base * (n_hv_mv**2)
        x_mv_ref_hv = x_mv_own_base * (n_hv_mv**2)

        # LV侧阻抗归算到HV侧
        n_hv_lv = vn_hv_kv / vn_lv_kv
        r_lv_ref_hv = r_lv_own_base * (n_hv_lv**2)
        x_lv_ref_hv = x_lv_own_base * (n_hv_lv**2)

        # --- 4. 计算短路阻抗并转换为百分比 ---
        # pandapower命名：vk_hv -> HV-MV; vk_mv -> HV-LV; vk_lv -> MV-LV

        # (A) HV-MV 短路参数
        r_hv_mv = r_hv_ref + r_mv_ref_hv
        x_hv_mv = x_hv_ref + x_mv_ref_hv
        z_hv_mv = np.sqrt(r_hv_mv**2 + x_hv_mv**2)
        vkr_hv_percent = (r_hv_mv / z_base_hv) * 100
        vk_hv_percent = (z_hv_mv / z_base_hv) * 100

        # (B) HV-LV 短路参数
        r_hv_lv = r_hv_ref + r_lv_ref_hv
        x_hv_lv = x_hv_ref + x_lv_ref_hv
        z_hv_lv = np.sqrt(r_hv_lv**2 + x_hv_lv**2)
        vkr_mv_percent = (r_hv_lv / z_base_hv) * 100
        vk_mv_percent = (z_hv_lv / z_base_hv) * 100

        # (C) MV-LV 短路参数
        r_mv_lv = r_mv_ref_hv + r_lv_ref_hv
        x_mv_lv = x_mv_ref_hv + x_lv_ref_hv
        z_mv_lv = np.sqrt(r_mv_lv**2 + x_mv_lv**2)
        vkr_lv_percent = (r_mv_lv / z_base_hv) * 100
        vk_lv_percent = (z_mv_lv / z_base_hv) * 100

        # --- 5. 提取其他参数 ---
        pfe_kw = trafo['P0'] * 1000  # P0单位通常是MW
        i0_percent = 0.5  # JSON中无此数据，使用一个典型估价值

        # --- 6. 打印调试信息 ---
        if _print:
            print("--- Transformer Parameter Calculation ---")
            print(f"Base Impedance (HV side): {z_base_hv:.4f} Ohm")
            print("\nReferred T-Model Impedances (to HV side):")
            print(f"  R_hv_ref: {r_hv_ref:.4f} Ohm, X_hv_ref: {x_hv_ref:.4f} Ohm")
            print(f"  R_mv_ref_hv: {r_mv_ref_hv:.4f} Ohm, X_mv_ref_hv: {x_mv_ref_hv:.4f} Ohm")
            print(f"  R_lv_ref_hv: {r_lv_ref_hv:.4f} Ohm, X_lv_ref_hv: {x_lv_ref_hv:.4f} Ohm")
            print("\nCalculated Short-Circuit Parameters:")
            print(f"  HV-MV Side (vk_hv, vkr_hv): {vk_hv_percent:.4f} %, {vkr_hv_percent:.4f} %")
            print(f"  HV-LV Side (vk_mv, vkr_mv): {vk_mv_percent:.4f} %, {vkr_mv_percent:.4f} %")
            print(f"  MV-LV Side (vk_lv, vkr_lv): {vk_lv_percent:.4f} %, {vkr_lv_percent:.4f} %")
            print("\nOther Parameters:")
            print(f"  Iron Losses (pfe_kw): {pfe_kw:.4f} kW")
            print(f"  No-load Current (i0_percent): {i0_percent:.4f} % (assumed)")
            print("-----------------------------------------")

        return (vk_hv_percent, vk_mv_percent, vk_lv_percent,
                vkr_hv_percent, vkr_mv_percent, vkr_lv_percent,
                pfe_kw, i0_percent)

if __name__ == '__main__':
    data_dir = "/home/xwj/yancheng/data"
    data_name = "盐城算例(1)"

    converter = TransmissionNetworkParser()

    converter.generate_pp_format_json(data_name, data_dir)

    # import json
    # import numpy as np
    # import pandapower as pp
    # import os

    # from collections import defaultdict

    # data_dir = "/home/xwj/yancheng/data"
    # data_name = "盐城算例(1)"
    # trans_path = os.path.join(data_dir, f"{data_name}.txt")
    # target_area_name = "丰富"
    # with open(trans_path, "r") as f:
    #     data_dict = json.load(f)
    # data_dict.keys()
