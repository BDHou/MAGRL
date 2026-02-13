import pickle
import numpy as np

samples = pickle.load(open("out_dataset/feeder_case33bw__nocontrol.pkl", "rb"))
s0 = samples[0]

y_node_va = s0["y_node_va"]  # ✅ 改这里：sample -> s0

print("y_node_va shape:", y_node_va.shape)
print("y_node_va min/max:", float(y_node_va.min()), float(y_node_va.max()))

print("va (deg) approx range:", float(np.min(y_node_va)), float(np.max(y_node_va)))
print("va (rad) approx range:",
      float(np.min(y_node_va) * np.pi/180),
      float(np.max(y_node_va) * np.pi/180))

samples = pickle.load(open("out_dataset/feeder_case33bw__nocontrol.pkl","rb"))
edge_attr = samples[0]["edge_attr"]      # shape: (2*nl, 4)

col4 = edge_attr[:, 3]                  # 第4维 = max_i_ka
print("edge_attr shape:", edge_attr.shape)
print("col4 unique count:", np.unique(col4).size)
print("col4 min/max:", float(col4.min()), float(col4.max()))
print("col4 all zero?", np.allclose(col4, 0.0))