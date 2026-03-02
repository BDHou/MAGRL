# scenario/dataset_check_pt.py
import os
import argparse
import numpy as np
import torch

def load_pt(pt_path: str):
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return obj[0], obj[1]  # (data, slices)
    return obj, None

def get_item(data, slices, idx=0):
    if slices is None:
        return data

    out = data.__class__()
    for key in data.keys():   # ✅ 注意这里是 keys()
        item = data[key]
        sl = slices[key]
        start, end = int(sl[idx]), int(sl[idx + 1])

        if torch.is_tensor(item):
            dim = data.__cat_dim__(key, item)
            out[key] = item.narrow(dim, start, end - start)
        else:
            out[key] = item[start:end]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/offline_case33bw",
                    help="must match training --data_root")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n_check", type=int, default=50)
    args = ap.parse_args()

    pt_path = os.path.join(args.data_root, "processed", "data.pt")
    print("Loading:", pt_path)

    data, slices = load_pt(pt_path)
    print("Loaded type:", type(data))
    print("Has slices:", slices is not None)

    s0 = get_item(data, slices, args.idx)

    print("\n=== Sample basic ===")
    print("x:", tuple(s0.x.shape))
    print("edge_index:", tuple(s0.edge_index.shape))
    edge_attr = getattr(s0, "edge_attr", None)
    if edge_attr is None:
        print("edge_attr: None")
        return
    print("edge_attr:", tuple(edge_attr.shape))

    ea = edge_attr.detach().cpu().numpy()

    # -------- CHECK 1 --------
    print("\n--- [CHECK 1] edge_attr per-column stats ---")
    A = ea.shape[1]
    for j in range(A):
        col = ea[:, j]
        uniq = np.unique(col)
        print(f"col[{j}] min/max = {float(col.min()):.6g} / {float(col.max()):.6g} | unique={len(uniq)}")
        if len(uniq) <= 10:
            print("   uniques:", uniq.tolist())

    abs_med = np.median(np.abs(ea), axis=0) + 1e-12
    abs_max = np.max(np.abs(ea), axis=0)
    ratio = abs_max / abs_med
    print("\ncol-wise max(abs)/median(abs):", ratio.tolist())

    # -------- CHECK 2 --------
    print("\n--- [CHECK 2] max_i_ka constant 99999? ---")
    max_i_col = 3  # if [r,x,length,max_i]
    if A > max_i_col:
        max_i = ea[:, max_i_col]
        uniq = np.unique(max_i)
        all_99999 = np.allclose(max_i, 99999.0)
        print(f"max_i_ka col[{max_i_col}] unique={len(uniq)} min/max={float(max_i.min())}/{float(max_i.max())}")
        print("all 99999.0 ?", bool(all_99999))
    else:
        print(f"edge_attr has only {A} cols, no col[{max_i_col}]")

    # -------- CHECK 3 --------
    print("\n--- [CHECK 3] alignment: (u->v) vs (v->u) edge_attr should match ---")
    ei = s0.edge_index.detach().cpu().numpy()
    src = ei[0].astype(int)
    dst = ei[1].astype(int)

    pair2idx = {}
    for i in range(src.shape[0]):
        pair2idx.setdefault((int(src[i]), int(dst[i])), i)

    rng = np.random.default_rng(0)
    all_idx = np.arange(src.shape[0])
    rng.shuffle(all_idx)

    n_check = min(args.n_check, len(all_idx))
    mismatch = 0
    checked = 0

    for i in all_idx[:n_check]:
        u = int(src[i]); v = int(dst[i])
        j = pair2idx.get((v, u), None)
        if j is None:
            continue
        checked += 1
        if not np.allclose(ea[i], ea[j], rtol=1e-6, atol=1e-9):
            mismatch += 1
            print(f"[MISMATCH] ({u}->{v}) idx={i}  vs  ({v}->{u}) idx={j}")
            print("   attr(u->v):", ea[i].tolist())
            print("   attr(v->u):", ea[j].tolist())

    print(f"checked_pairs={checked}, mismatched_pairs={mismatch}")
    if checked == 0:
        print("No reverse pairs found (如果你不是存 2E directed edges，这也可能正常)")
    elif mismatch == 0:
        print("Alignment looks OK on sampled pairs.")
    else:
        print("!! Alignment BUG likely: reverse edges do not share same edge_attr.")

if __name__ == "__main__":
    main()