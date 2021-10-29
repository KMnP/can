#!/usr/bin/env python3
"""
Dialogue RE experiments
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
from .eval import get_all_metrics, get_dev_metrics

from ...rsa.rsa import RSA
from ...rsa.prepare_rsa import entropy, create_fixed_sets_dict
from ...utils import vis
from ...utils.io_utils import save_or_append_df


def run(data_root):
    root = f"{data_root}/data/DialRE"
    # total~8 hours
    for lang in ["cn", "en"]:
        for bert_type in ["bert", "berts"]:
            for run_type in ["", "_2", "_3", "_4", "_5"]:
                tune_dev(root, lang, bert_type, run_type)  # ~36 min
                get_testresults(root, lang, bert_type, run_type)

    # get the saved probs for visluzations
    # get_testresults(root, "en", "berts", "_2", save_probs=True)
    # get_testresults(root, "cn", "berts", "", save_probs=True)


def get_testresults(root, lang, bert_type, model_type, save_probs=False):
    prior_list = count_labels(root, lang)
    prior_array_list = [prior_list]
    prior_names = ["train_distribution"]

    fixed_types = [
        "dev_test_confident_lite_stratefy",
    ]

    confused_threshold_choices = [0.25, 0.5, 0.75]
    for f_type in tqdm(fixed_types):
        out_path = f"{root}/" + "_".join([lang, bert_type, "test"]) + ".csv"
        dev_out_path = f"{root}/" + "_".join([lang, bert_type]) + ".csv"
        for confused_t in confused_threshold_choices:
            for prior, p_name in zip(prior_array_list, prior_names):
                best_alpha, best_agent, best_t2, best_t2_0 = get_best(
                    dev_out_path, f_type, model_type,
                    p_name, confused_t, False
                )
                best_alpha_c, best_agent_c, best_t2_c, _ = get_best(
                    dev_out_path, f_type, model_type,
                    p_name, confused_t, True
                )
                df = compute_test_rsa(
                    root, lang, bert_type, model_type,
                    best_alpha, confused_t, f_type, prior, best_agent,
                    best_alpha, prior, best_agent,
                    best_t2, best_t2, best_t2_0, save_probs
                )
                df["prior"] = p_name
                df["prior_c"] = p_name
                df["hp_type"] = "same_as_standard"
                print(df)
                if not save_probs:
                    save_or_append_df(out_path, df)


def tune_dev(root, lang, bert_type, model_type):
    prior_list = count_labels(root, lang)
    prior_array_list = [prior_list]
    p_names = ["train_distribution"]

    fixed_types = [
        "test_confident_lite_stratefy",
    ]

    confused_threshold_choices = [0.25, 0.5, 0.75]

    alpha_list = [i * 0.1 for i in range(1, 10)]
    alpha_list += [i for i in range(1, 20)]
    for f_type in tqdm(fixed_types):
        out_path = f"{root}/" + "_".join([lang, bert_type]) + ".csv"
        for confused_t in confused_threshold_choices:
            for prior, p_name in zip(prior_array_list, p_names):
                # construct the arg_list
                arg_list = []
                arg_list.extend([(
                    root, lang, bert_type, model_type, a,
                    confused_t, f_type, prior) for a in alpha_list
                ])

                # spawn process
                print(f"\tgoing to spawn {len(arg_list)} processes")
                with poolcontext(processes=len(arg_list)) as pool:
                    df_list = pool.map(get_metric_unpack, iter(arg_list))

                _df = pd.concat(df_list, ignore_index=True)
                _df["prior"] = p_name
                # print out the best results
                vis.inspect_f1(
                    _df,
                    f"dev-{lang}-{bert_type}-{model_type}" +
                    f"-{confused_t}-{f_type}-{p_name}",
                    ["dev_f1_macro", "dev_f1", "dev_c_f1"]
                )
                save_or_append_df(out_path, _df)


def get_metric_unpack(args):
    return compute_dev_rsa(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def compute_test_rsa(
    root, lang, bert_type, run_type,
    alpha, confused_t, fixed_type, prior_list, agent,
    alpha_c, prior_list_c, agent_c, best_t2, best_t2_c, best_t2_0,
    save_probs=False, depth=5,
):
    """get the metrics for each alpha, prior_list, fixed_type"""
    eval_type = "test"
    targets, probs = get_output(root, eval_type, lang, bert_type, run_type)
    _, probs_c = get_output(root, eval_type, lang, bert_type, run_type + "c")

    targets_dev, probs_dev = get_output(root, "dev", lang, bert_type, run_type)
    _, probs_c_dev = get_output(root, "dev", lang, bert_type, run_type + "c")

    level2probs = {
        # can't reproduce the paper results using this setting, since the best_t2, best_t2_c are tuned for
        "original": (
            copy.deepcopy(probs), copy.deepcopy(probs_c),
            copy.deepcopy(probs_dev), copy.deepcopy(probs_c_dev),
        ),
        "best": (
            copy.deepcopy(probs), copy.deepcopy(probs_c),
            copy.deepcopy(probs_dev), copy.deepcopy(probs_c_dev),
        ),
    }
    output = {}

    # find the confused location and construct the fixed set
    for probs_tuple, eval_type, offset in zip(
        [(probs, probs_c), (probs_dev, probs_c_dev)], ["test", "dev"], [0, 2]
    ):
        probs, probs_c = probs_tuple

        if eval_type == "dev" and fixed_type.startswith("dev"):
            _fixed_type = fixed_type[4:]
        else:
            _fixed_type = fixed_type

        X_seen, row_ids, col_ids = create_or_cache_fixedset(
            probs, confused_t, _fixed_type, root,
            eval_type, "_".join([lang, bert_type, run_type]), prior_list)
        X_seen_c, row_ids_c, col_ids_c = create_or_cache_fixedset(
            probs_c, confused_t, _fixed_type, root,
            eval_type, "_".join([lang, bert_type, run_type + "c"]),
            prior_list_c)

        if eval_type == "dev" and save_probs:
            output["confused_cases"] = (row_ids, col_ids, row_ids_c, col_ids_c)

        # do rsa one location at a time
        rsa = RSA(
            2, X_seen.shape[0] + 1,
            alpha=alpha, costs=None,
            prior=prior_list,
            use_lexicon_as_base=True,
            normalize_use_base=False
        )
        rsa_c = RSA(
            2, X_seen_c.shape[0] + 1,
            alpha=alpha_c, costs=None,
            prior=prior_list_c,
            use_lexicon_as_base=True,
            normalize_use_base=False,
        )
        for r, c in zip(row_ids, col_ids):
            single_distribution = np.array([probs[r, c], 1 - probs[r, c]])
            V0 = np.vstack([X_seen, single_distribution[np.newaxis, :]])
            _level2probs = rsa.viewer(V0.T, depth, return_all=True)
            prob_value = _level2probs[agent][0, -1]  # _probs: 2 x num_samples
            level2probs["best"][0 + offset][r, c] = prob_value

        for r_c, c_c in zip(row_ids_c, col_ids_c):
            single_distribution_c = np.array([
                probs_c[r_c, c_c], 1 - probs_c[r_c, c_c]])
            V0_c = np.vstack([X_seen_c, single_distribution_c[np.newaxis, :]])
            _level2probs_c = rsa_c.viewer(V0_c.T, depth, return_all=True)
            prob_value_c = _level2probs_c[agent_c][0, -1]
            level2probs["best"][1 + offset][r_c, c_c] = prob_value_c

    if save_probs:
        output.update({"probs": level2probs, "dev_targets": targets_dev})
        out_probs_path = os.path.join(
            root, f"{lang}_{bert_type}_{run_type}_{confused_t}_probs.pth")
        torch.save(output, out_probs_path)
        print(f"saved output to {out_probs_path}")

    # return the df
    df = create_metrics(
        level2probs, (targets, targets_dev),
        best_t2, best_t2_c, best_t2_0
    )
    # add dev metrics
    df["confused_threshold"] = confused_t
    df["fixed_type"] = fixed_type
    df["language"] = lang
    df["run"] = run_type
    df["alpha"] = alpha
    df["alpha_c"] = alpha_c
    df["agent"] = agent
    df["agent_c"] = agent_c
    return df


def compute_dev_rsa(
    root, lang, bert_type, run_type,
    alpha, confused_t, fixed_type, prior_list, depth=5,
):
    """get the metrics for each alpha, prior_list, fixed_type"""
    eval_type = "dev"
    targets, probs = get_output(root, eval_type, lang, bert_type, run_type)
    _, probs_c = get_output(root, eval_type, lang, bert_type, run_type + "c")

    level2probs = {"original": (probs, probs_c)}

    # find the confused location and construct the fixed set
    X_seen, row_ids, col_ids = create_or_cache_fixedset(
        probs, confused_t, fixed_type, root,
        eval_type, "_".join([lang, bert_type, run_type]), prior_list)
    X_seen_c, row_ids_c, col_ids_c = create_or_cache_fixedset(
        probs_c, confused_t, fixed_type, root,
        eval_type, "_".join([lang, bert_type, run_type + "c"]), prior_list)

    # do rsa one location at a time
    rsa = RSA(
        2, X_seen.shape[0] + 1,
        alpha=alpha, costs=None,
        prior=prior_list,
        use_lexicon_as_base=True,
        normalize_use_base=False
    )
    for r, c in zip(row_ids, col_ids):
        single_distribution = np.array([probs[r, c], 1 - probs[r, c]])
        V0 = np.vstack([X_seen, single_distribution[np.newaxis, :]])
        _level2probs = rsa.viewer(V0.T, depth, return_all=True)
        for level, _probs in _level2probs.items():
            prob_value = _probs[0, -1]  # _probs: 2 x num_samples
            if level not in level2probs:
                level2probs[level] = (
                    copy.deepcopy(probs), copy.deepcopy(probs_c))

            level2probs[level][0][r, c] = prob_value

    for r_c, c_c in zip(row_ids_c, col_ids_c):
        single_distribution_c = np.array([
            probs_c[r_c, c_c], 1 - probs_c[r_c, c_c]])
        V0_c = np.vstack([X_seen_c, single_distribution_c[np.newaxis, :]])
        _level2probs_c = rsa.viewer(V0_c.T, depth, return_all=True)
        for level, _probs in _level2probs_c.items():
            prob_value_c = _probs[0, -1]
            level2probs[level][1][r_c, c_c] = prob_value_c

    # return the df
    df = create_metrics(level2probs, targets)
    df["alpha"] = alpha
    df["fixed_type"] = fixed_type
    df["confused_threshold"] = confused_t
    df["test_type"] = eval_type
    df["language"] = lang
    df["bert_type"] = bert_type
    df["run"] = run_type
    return df


def create_or_cache_fixedset(
    probs, confused_threshold, fixed_type,
    data_dir, eval_type, model_type, prior_list
):
    cache_file = os.path.join(
        data_dir, f"cached_{confused_threshold}_{eval_type}_{model_type}.pth")
    if os.path.exists(cache_file):
        data = torch.load(cache_file)
        return data["fixed_set"][fixed_type], data["row_ids"], data["col_ids"]

    print("Creating fixed set and locations from the given probability")
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

    if eval_type == "test":
        val_cache_file = os.path.join(
            data_dir, f"cached_{confused_threshold}_dev_{model_type}.pth")
        val_cache = torch.load(val_cache_file)
        fixed_set_dict["dev_test_confident_lite_random"] = val_cache["fixed_set"]["test_confident_lite_random"]
        fixed_set_dict["dev_test_confident_lite_stratefy"] = val_cache["fixed_set"]["test_confident_lite_stratefy"]
    torch.save(
        {"fixed_set": fixed_set_dict, "row_ids": row_ids, "col_ids": col_ids},
        cache_file
    )
    print(f"\tnumber of confused cases: {len(row_ids)}/{probs.size}")
    return fixed_set_dict[fixed_type], row_ids, col_ids


def create_metrics(
    level2probs, targets, best_t2=None, best_t2_c=None, best_t2_0=None
):
    """organize the result, produce a readable table (pd.DataFrame)"""
    # compute metrics for all levels
    level2metrics = defaultdict(dict)
    for level, probs_tuple in level2probs.items():
        # evaluate the probs
        if best_t2 is None:
            probs, probs_c = probs_tuple
            metric = get_dev_metrics(list(probs), list(probs_c), targets)
        else:
            targets_test, targets_dev = targets
            probs, probs_c, probs_dev, probs_c_dev = probs_tuple
            if level == "original":
                metric = get_all_metrics(
                    list(probs_dev), list(probs_c_dev),
                    list(probs), list(probs_c),
                    targets_dev, targets_test, best_t2_0, best_t2_0)
            else:
                metric = get_all_metrics(
                    list(probs_dev), list(probs_c_dev),
                    list(probs), list(probs_c),
                    targets_dev, targets_test, best_t2, best_t2_c)
        for m_name, v in metric.items():
            level2metrics[level][m_name] = v
    # return the df
    df = _get_df(level2metrics)
    return df


def _get_df(level2metrics):
    pd_dict = defaultdict(list)
    for level, metrics in level2metrics.items():
        pd_dict["name"].append(level)
        for name, v in metrics.items():
            pd_dict[name].append(v)

    df = pd.DataFrame(pd_dict)
    return df


def get_best(
    dev_out_path, f_type, run_type, prior_name, confused_t, is_convert=False
):
    result_df = pd.read_csv(dev_out_path)
    result_df = result_df.replace(np.nan, '', regex=True)
    for c in result_df.columns:
        if c.startswith("Unnamed"):
            del result_df[c]

    if f_type.startswith("dev"):
        f_type = f_type[4:]
    df_filter = result_df[result_df["confused_threshold"] == confused_t]
    df_filter = df_filter[df_filter["fixed_type"] == f_type]
    df_filter = df_filter[df_filter["run"] == run_type]
    df_filter = df_filter[df_filter["prior"] == prior_name]
    df_filter.reset_index(drop=True, inplace=True)

    # print(df_filter.shape)  # # alpha * 2 priors * 7

    if is_convert:
        m = "dev_c_f1"
    else:
        m = "dev_f1"
        # m = "dev_f1_macro"
    row_idx = np.argmax(df_filter[m])
    # print(df_filter.loc[[row_idx]])

    best_agent = df_filter["name"][row_idx]
    best_alpha = df_filter["alpha"][row_idx]
    best_t2 = df_filter["best_T2"][row_idx]
    best_t2_0 = list(df_filter[df_filter["name"] == "V0"]["best_T2"])[0]

    if best_agent == "original":
        best_agent = "V0"

    return best_alpha, best_agent, best_t2, best_t2_0

