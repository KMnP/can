#!/usr/bin/env python3
"""
Ultra-fine Entity Typing experiments
"""
import os
import copy
import pandas as pd
import torch
import multiprocessing
import numpy as np

from collections import defaultdict
from contextlib import contextmanager
from tqdm import tqdm

from .data import get_output, count_labels
from .eval import compute_metrics

from ...rsa.rsa import RSA
from ...rsa.prepare_rsa import entropy, create_fixed_sets_dict
from ...utils import vis
from ...utils.io_utils import save_or_append_df


def run(data_root, model_type="baseline"):
    root = f"{data_root}/data/UltraFineEntity-{model_type}"

    tune(root, "dev", "all")
    get_test(root, "all")


def tune(root, test_type, label_type):
    """spawn num_alphas x num_threshold processes"""
    prior_list = count_labels(root, "dev")
    prior_array_list = [prior_list]
    prior_names = ["train_distribution"]

    fixed_types = [
        "test_confident_lite_stratefy",
    ]

    confused_threshold_choices = [0.75, 0.5, 0.25]

    alpha_list = [i * 0.1 for i in range(1, 10)]
    alpha_list += [i for i in range(1, 31)]
    for f_type in fixed_types:
        temp_out_path = f"{root}/{label_type}.csv"
        for confused_t in tqdm(confused_threshold_choices, desc=f_type):
            for prior, p_name in zip(prior_array_list, prior_names):
                # construct the arg_list
                arg_list = []
                arg_list.extend([(root, test_type, label_type, a, confused_t, f_type, prior) for a in alpha_list])

                # spawn process
                print(f"\t\tgoing to spawn {len(arg_list) // 2 - 1} processes")
                with poolcontext(processes=len(arg_list) // 2 - 1) as pool:
                    df_list = pool.map(get_metric_unpack, iter(arg_list))

                _df = pd.concat(df_list, ignore_index=True)
                _df["prior"] = p_name
                # print out the best results
                vis.inspect_f1(
                    _df,
                    f"{test_type}-{label_type}-{confused_t}-{f_type}-{p_name}"
                )
                save_or_append_df(temp_out_path, _df)


def get_test(root, label_type):
    dev_out_path = f"{root}/{label_type}.csv"
    out_path = f"{root}/{label_type}_best.csv"

    prior_list = count_labels(root, "dev")
    prior_array_list = [prior_list]
    p_names = ["train_distribution"]

    fixed_types = [
        "test_confident_lite_stratefy",
    ]
    pname2list = {}
    for prior, p_name in zip(prior_array_list, p_names):
        pname2list[p_name] = prior


    for f_type in fixed_types:
        best_alpha, best_agent, best_pname, confused_t, best_f1 = get_best_all(
                dev_out_path, f_type)
        print(best_alpha, best_agent, best_pname, confused_t, best_f1)
        prior = pname2list[best_pname]
        for test_type in ["dev", "test"]:
            df = compute_rsa(
                root, test_type, label_type, best_alpha, confused_t,
                f_type, prior, save_probs=True
            )
            df["prior"] = best_pname

            print(df[df["name"].isin(["V0", best_agent])])
            save_or_append_df(
                out_path, df[df["name"].isin(["V0", best_agent])])


def get_metric_unpack(args):
    return compute_rsa(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def compute_rsa(
    root, test_type, label_type, alpha, confused_t,
    fixed_type, prior_list, depth=5, save_probs=False
):
    output = {}
    # get the probs
    all_probs, targets = get_output(root, test_type, "all")

    if label_type == "coarse":
        probs = copy.deepcopy(all_probs[:, :9])
    elif label_type == "fine":
        probs = copy.deepcopy(all_probs[:, 9:130])
    elif label_type == "ultra-fine":
        probs = copy.deepcopy(all_probs[:, 130:])
    elif label_type == "all":
        probs = copy.deepcopy(all_probs)
    level2probs = {"original": copy.deepcopy(all_probs)}

    # find the confused location and construct the fixed set
    X_seen, row_ids, col_ids = create_or_cache_fixedset(
        probs, confused_t, fixed_type, root, test_type, label_type, prior_list)

    output[label_type] = (row_ids, col_ids)

    # do rsa one location at a time
    rsa = RSA(
        2, X_seen.shape[0] + 1,
        alpha=alpha, costs=None,
        prior=prior_list,
        use_lexicon_as_base=True,
        normalize_use_base=False
    )
    for r, c in zip(row_ids, col_ids):
        if label_type == "fine":
            c = c + 9
        elif label_type == "ultra-fine":
            c = c + 130

        single_distribution = np.array([probs[r, c], 1 - probs[r, c]])
        V0 = np.vstack([X_seen, single_distribution[np.newaxis, :]])
        _level2probs = rsa.viewer(V0.T, depth, return_all=True)
        for level, _probs in _level2probs.items():
            prob_value = _probs[0, -1]  # _probs: 2 x num_samples
            if level not in level2probs:
                level2probs[level] = copy.deepcopy(all_probs)
            level2probs[level][r, c] = prob_value

    if save_probs:
        output["probs"] = level2probs
        out_probs_path = os.path.join(
            root, f"{test_type}_{alpha}_{confused_t}_probs.pth")
        torch.save(output, out_probs_path)
        print(f"saved output to {out_probs_path}")

    # compute metrics for all levels
    level2metrics = defaultdict(dict)
    for level, probs in level2probs.items():
        metric = compute_metrics(probs, targets, root, label_type)
        for m_name, v in metric.items():
            level2metrics[level][m_name] = v

        level2metrics[level]["alpha"] = alpha
        level2metrics[level]["fixed_type"] = fixed_type
        level2metrics[level]["confused_threshold"] = confused_t

    # return the df
    df = _get_df(level2metrics)
    df["test_type"] = test_type
    df["label_type"] = label_type
    return df


def create_or_cache_fixedset(
    probs, confused_threshold, fixed_type,
    data_dir, eval_type, label_type, prior_list
):
    cache_file = os.path.join(
        data_dir, f"cached_{confused_threshold}_{eval_type}_{label_type}.pth")
    if os.path.exists(cache_file):
        if fixed_type.startswith("test") and eval_type == "test":
            val_cache_file = os.path.join(data_dir,
                f"cached_{confused_threshold}_dev_{label_type}.pth")
            val_fixed = torch.load(val_cache_file)["fixed_set"][fixed_type]
            data = torch.load(cache_file)
            return val_fixed, data["row_ids"], data["col_ids"]
        else:
            data = torch.load(cache_file)
            return data["fixed_set"][fixed_type], data["row_ids"], data["col_ids"]

    print("Creating fixed set and locations from the given probability")
    if prior_list is None:
        prior_list = count_labels(data_dir, "dev")
    row_ids = []
    col_ids = []
    fixed_set = []
    num_rows, num_cols = probs.shape
    for r in range(num_rows):
        for c in range(num_cols):
            single_distribution = np.array([probs[r, c], 1 - probs[r, c]])
            # no need for topk since there are only 2 classes
            ent = entropy(single_distribution, base=2)

            if ent >= confused_threshold:
                row_ids.append(r)
                col_ids.append(c)
            else:
                fixed_set.append(single_distribution[np.newaxis, :])
    fixed_set = np.vstack(fixed_set)

    # create the fixed set according to different types
    dummy_probs_confused = np.zeros((len(row_ids), 2))
    fixed_set_dict = create_fixed_sets_dict(
        fixed_set, dummy_probs_confused, prior_list)
    torch.save(
        {"fixed_set": fixed_set_dict, "row_ids": row_ids, "col_ids": col_ids},
        cache_file
    )
    print(f"\tnumber of confused cases: {len(row_ids)}/{probs.size}")
    return fixed_set_dict[fixed_type], row_ids, col_ids


def _get_df(level2metrics):
    """organize the result, produce a readable table (pd.DataFrame)"""
    pd_dict = defaultdict(list)
    for level, metrics in level2metrics.items():
        pd_dict["name"].append(level)
        for name, v in metrics.items():
            pd_dict[name].append(v)

    df = pd.DataFrame(pd_dict)
    df = df.sort_values(['name'])
    return df


def get_best_all(dev_out_path, f_type):
    result_df = pd.read_csv(dev_out_path)
    result_df = result_df.replace(np.nan, '', regex=True)
    for c in result_df.columns:
        if c.startswith("Unnamed"):
            del result_df[c]

    if f_type.startswith("dev"):
        f_type = f_type[4:]

    df_filter = result_df[result_df["fixed_type"] == f_type]
    # get the best parameters using dev set
    df_filter = df_filter[df_filter["test_type"] == "dev"]
    df_filter = df_filter[df_filter["prior"].isin(
        ["uniform", "train_distribution"])]
    df_filter.reset_index(drop=True, inplace=True)

    m = "f1"
    row_idx = np.argmax(df_filter[m])

    best_f1 = df_filter[m][row_idx]

    best_agent = df_filter["name"][row_idx]
    best_alpha = df_filter["alpha"][row_idx]
    best_prior = df_filter["prior"][row_idx]
    best_t = float(df_filter["confused_threshold"][row_idx])

    return best_alpha, best_agent, best_prior, best_t, best_f1
