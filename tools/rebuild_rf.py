"""Step 2 (run in the NEW env, sklearn >= 1.3): rebuild RandomForest from npz + json, pickle as a bundle.
Usage: python rebuild_rf.py IN_PREFIX OUT.pkl"""
import json
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import NODE_DTYPE, Tree

prefix, out = sys.argv[1], sys.argv[2]
info = json.load(open(prefix + ".json", encoding="utf-8"))
arr = np.load(prefix + ".npz", allow_pickle=False)
classes = arr["classes"]
n_feat, n_out, n_cls = info["n_features_in_"], info["n_outputs_"], info["n_classes_"]

rf = RandomForestClassifier(**{k: v for k, v in info["rf_params"].items()
                               if k in RandomForestClassifier().get_params()})
estimators = []
for i, t in enumerate(info["trees"]):
    nodes = np.zeros(t["node_count"], dtype=NODE_DTYPE)
    for name in NODE_DTYPE.names:
        key = f"t{i}_{name}"
        if key in arr:
            nodes[name] = arr[key]
        # 'missing_go_to_left' did not exist in 1.1.3; training data had no NaN -> 0
    values = arr[f"t{i}_values"].astype(np.float64)
    # sklearn < 1.4 stored weighted class counts per node; >= 1.4 stores class fractions
    # and predict_proba no longer normalizes. Normalize so leaf probabilities are identical.
    tot = values.sum(axis=2, keepdims=True)
    tot[tot == 0] = 1.0
    values = values / tot
    tree = Tree(n_feat, np.array([n_cls], dtype=np.intp), n_out)
    tree.__setstate__({"max_depth": t["max_depth"], "node_count": t["node_count"],
                       "nodes": nodes, "values": values})
    est = DecisionTreeClassifier(**{k: v for k, v in rf.estimator_params_dict().items()}) \
        if hasattr(rf, "estimator_params_dict") else DecisionTreeClassifier(
            criterion=rf.criterion, max_depth=rf.max_depth, min_samples_split=rf.min_samples_split,
            min_samples_leaf=rf.min_samples_leaf, min_weight_fraction_leaf=rf.min_weight_fraction_leaf,
            max_features=rf.max_features, max_leaf_nodes=rf.max_leaf_nodes,
            min_impurity_decrease=rf.min_impurity_decrease, ccp_alpha=rf.ccp_alpha,
            random_state=t["random_state"])
    est.tree_ = tree
    est.classes_ = classes
    est.n_classes_ = n_cls
    est.n_outputs_ = n_out
    est.n_features_in_ = n_feat
    est.max_features_ = t["max_features_"]
    estimators.append(est)

rf.estimators_ = estimators
rf.estimator_ = DecisionTreeClassifier()
rf.classes_ = classes
rf.n_classes_ = n_cls
rf.n_outputs_ = n_out
rf.n_features_in_ = n_feat
if info["feature_names_in_"]:
    rf.feature_names_in_ = np.array(info["feature_names_in_"], dtype=object)
bundle = dict(info["bundle_meta"])
bundle["model"] = rf
bundle["converted_from"] = "sklearn 1.1.3 pickle via export_rf/rebuild_rf (lossless tree copy)"
pickle.dump(bundle, open(out, "wb"))
print(f"rebuilt {len(estimators)} trees -> {out}")
