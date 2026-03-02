# MAGRL

dataset generation instructions：
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
