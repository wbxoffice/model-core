"""Step 1 (run in the OLD env, sklearn 1.1.3): dump RandomForest bundles to version-neutral npz + json.
Usage: python export_rf.py IN.pkl OUT_PREFIX"""
import json
import pickle
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
bundle = pickle.load(open(src, "rb"))
rf = bundle["model"]
meta = {k: v for k, v in bundle.items() if k != "model"}
params = rf.get_params()
arrays = {
    "classes": np.asarray(rf.classes_),
}
trees = []
for i, est in enumerate(rf.estimators_):
    st = est.tree_.__getstate__()
    nodes = st["nodes"]
    for name in nodes.dtype.names:
        arrays[f"t{i}_{name}"] = np.ascontiguousarray(nodes[name])
    arrays[f"t{i}_values"] = np.asarray(st["values"])
    trees.append({"max_depth": int(st["max_depth"]), "node_count": int(st["node_count"]),
                  "max_features_": int(est.max_features_), "random_state": int(est.random_state)})
info = {
    "bundle_meta": meta,
    "rf_params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))},
    "n_features_in_": int(rf.n_features_in_),
    "n_outputs_": int(rf.n_outputs_),
    "n_classes_": int(rf.n_classes_),
    "feature_names_in_": list(map(str, getattr(rf, "feature_names_in_", []))),
    "trees": trees,
}
np.savez_compressed(out + ".npz", **arrays)
json.dump(info, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
print(f"{src}: {len(trees)} trees exported")
