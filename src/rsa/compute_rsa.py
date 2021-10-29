#!/usr/bin/env python3
"""
rsa-related functions
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .evaluation import (
    top_n_accuracy, topk_success, aggregate_success, topk_accuracy_all)
from .rsa import RSA
from .prepare_rsa import (
    get_confused_instances_topk,
    get_fixed_set,
    filter_logits,
)


def tune(
    probs, targets,
    fixed_type="test_confident", prior_list=None,
    confused_threshold_choices=[0.01, 0.1, 0.5], depth=5, maxk=None,
    verbose=False
):
    """
    Tuning for alpha, pragmatic agents, and threshold,
    based on top1 accuracy for confused cases
    Args:
        probs: np.ndarray of shape (n_samples, n_classes)
               used as literal agent (viewer)
        targets: list or np.ndarray of (n_samples, )
        fixed_type: str, type of fixed set to produce
        prior_list: list or np.ndarray of (n_classes, )
                 if None, will assume uniform distribution.
        confused_threshold_choices: list of float
        verbose: bool
    Returns:
        result_df: pandas.dataframe, contains eval results
    """
    if verbose:
        table_headers = ["threshold", "V0", "best result", "alpha", "agent"]
        row_format ="{:>15}" * len(table_headers)
        print(row_format.format(*table_headers))
    agents = [f"V{i}" for i in range(1, 5)]
    level2metric_all = {}
    for confused_t in confused_threshold_choices:
        if verbose:
            conf_ids, highconf_ids = get_confused_instances_topk(
                probs, confused_t, maxk)
            print(len(conf_ids))
            print(len(highconf_ids))
        best_alpha = -1
        best_top1 = -1
        best_agent, best_level2metric = None, None

        alpha_list = [i * 0.1 for i in range(1, 10)] + \
            [i for i in range(1, 10)]
        for alpha in tqdm(alpha_list, desc=f"threshold={confused_t:.2f}"):
            level2metrics = get_rsa(
                probs, targets, alpha,
                confused_t, fixed_type, prior_list, depth
            )
            top1_list = [
                float(level2metrics[a]["top1"]) for a in agents
            ]
            best_idx = np.argmax(top1_list)
            c_top1 = np.max(top1_list)
            if c_top1 >= best_top1:
                best_alpha = alpha
                best_top1 = c_top1
                best_agent = agents[best_idx]
                best_level2metric = level2metrics
        if verbose:
            v0_top1 = float(level2metrics["V0"]["top1"])
            print(row_format.format(
                confused_t, round(v0_top1, 2), round(best_top1, 2),
                round(best_alpha, 1), best_agent)
            )
        level2metric_all[confused_t] = best_level2metric
    return _get_df(level2metric_all)


def get_rsa(
    probs, targets, alpha, confused_threshold,
    fixed_type="test_confident", prior_list=None, depth=5, efficient=True
):
    if efficient:
        return _get_rsa_save(
                    probs, targets, alpha, confused_threshold,
                    fixed_type, prior_list, depth
                )
    return _get_rsa(
                probs, targets, alpha, confused_threshold,
                fixed_type, prior_list, depth
            )


def _get_rsa(
    probs, targets, alpha, confused_threshold,
    fixed_type="test_confident", prior_list=None, depth=5
):
    """
    Compute pragmatic resutls
    Args:
        probs: np.ndarray of shape (n_samples, n_classes)
               used as literal agent (viewer)
        targets: list or np.ndarray of (n_samples, )
        alpha: float, a parameter that controls how "rational" the agent is
        confused_threshold: float, controls how to split the probs
        fixed_type: str, type of fixed set to produce
        prior_list: list or np.ndarray of (n_classes, )
                 if None, will assume uniform distribution.
    Returns:
        level2metrics: dict of {agent: metric_dict}
    """
    conf_ids, highconf_ids = get_confused_instances_topk(
        probs, confused_threshold)
    probs_unseen, targets_unseen = filter_logits(probs, targets, conf_ids)
    probs_confi, targets_confi = filter_logits(probs, targets, highconf_ids)

    all_targets = targets_unseen + targets_confi

    probs_fixed = get_fixed_set(
        probs_unseen, probs_confi, prior_list, fixed_type)
    level2probs = attach_rsa(prior_list, alpha, probs_fixed, probs_unseen, depth)
    level2metrics = defaultdict(dict)
    for level, _probs in level2probs.items():
        if level.startswith("C"):  # only look for viewer
            continue
        for k in [1, 3, 5]:
            level2metrics[level]["alpha"] = alpha
            level2metrics[level]["fixed_type"] = fixed_type
            level2metrics[level]["confused_threshold"] = confused_threshold

            acc = top_n_accuracy(_probs.T, targets_unseen, k)
            level2metrics[level][f"top{k}"] = round(acc * 100, 2)

        all_probs = np.hstack([_probs, probs_confi.T])
        for k in [1, 3, 5]:
            acc = top_n_accuracy(all_probs.T, all_targets, k)
            level2metrics[level][f"all-top{k}"] = round(acc * 100, 2)

    return level2metrics


def _get_rsa_save(
    probs, targets, alpha, confused_threshold,
    fixed_type="test_confident", prior_list=None, depth=5
):
    """
    Compute pragmatic resutls
    Args:
        probs: np.ndarray of shape (n_samples, n_classes)
               used as literal agent (viewer)
        targets: list or np.ndarray of (n_samples, )
        alpha: float, a parameter that controls how "rational" the agent is
        confused_threshold: float, controls how to split the probs
        fixed_type: str, type of fixed set to produce
        prior_list: list or np.ndarray of (n_classes, )
                 if None, will assume uniform distribution.
    Returns:
        level2metrics: dict of {agent: metric_dict}
    """
    conf_ids, highconf_ids = get_confused_instances_topk(
        probs, confused_threshold)
    probs_unseen, targets_unseen = filter_logits(probs, targets, conf_ids)
    probs_confi, targets_confi = filter_logits(probs, targets, highconf_ids)

    probs_fixed = get_fixed_set(
        probs_unseen, probs_confi, prior_list, fixed_type)
    level2successes = attach_rsa_save(
        prior_list, alpha, probs_fixed, probs_unseen, targets_unseen, depth)
    level2metrics = defaultdict(dict)
    for level, _successes in level2successes.items():
        for k_idx, k in enumerate([1, 3, 5]):
            level2metrics[level]["alpha"] = alpha
            level2metrics[level]["fixed_type"] = fixed_type
            level2metrics[level]["confused_threshold"] = confused_threshold

            acc, tp_counts = aggregate_success(_successes[k_idx, :])
            level2metrics[level][f"top{k}"] = round(acc * 100, 2)
            level2metrics[level][f"tp{k}"] = tp_counts
            level2metrics[level][f"confused_{k}"] = _successes.shape[1]

        for k_idx, k in enumerate([1, 3, 5]):
            acc = topk_accuracy_all(
                probs_confi, targets_confi, _successes[k_idx, :], k)
            level2metrics[level][f"all-top{k}"] = round(acc * 100, 2)

    return level2metrics


def attach_rsa_save(prior_list, alpha, X_seen, X_unseen, y_unseen, depth=5):
    """
    compute rsa results using "split-and-attach" method.
    args:
        prior_list: None or a list
        alpha: float, for rsa
        X_seen: numpy.array of shape n_samples1 x n_labels
        X_unseen: numpy.array of shape n_samples2 x n_labels
    returns
        level2probs: dict, map agent name to the probability distribution
    """
    num_unseen, num_classes = X_unseen.shape
    level2success_unseen = defaultdict(list)

    rsa = RSA(
        num_classes, X_seen.shape[0] + 1,
        alpha=alpha, costs=None,
        prior=prior_list, use_lexicon_as_base=True,
        normalize_use_base=False
    )

    for s_id in range(num_unseen):
        V0 = np.vstack([X_seen, X_unseen[s_id, :][np.newaxis, :]])
        level2probs = rsa.viewer(V0.T, depth, return_all=True)
        for level, _probs in level2probs.items():
            # compute the success rate of the unseen instance
            success_list = np.array([
                topk_success(_probs[:, -1], y_unseen[s_id], k)
                for k in [1, 3, 5]
            ])[:, np.newaxis]
            level2success_unseen[level].append(success_list)
    for level, probs_list in level2success_unseen.items():
        # (num_samples, 3)
        level2success_unseen[level] = np.hstack(probs_list)
    return level2success_unseen


def attach_rsa(prior_list, alpha, X_seen, X_unseen, depth=5):
    """
    compute rsa results using "split-and-attach" method.
    args:
        prior_list: None or a list
        alpha: float, for rsa
        X_seen: numpy.array of shape n_samples1 x n_labels
        X_unseen: numpy.array of shape n_samples2 x n_labels
    returns
        level2probs: dict, map agent name to the probability distribution
    """
    num_unseen, num_classes = X_unseen.shape
    level2probslist_unseen = defaultdict(list)

    rsa = RSA(
        num_classes, X_seen.shape[0] + 1,
        alpha=alpha, costs=None,
        prior=prior_list, use_lexicon_as_base=True,
        normalize_use_base=False
    )
    for s_id in range(num_unseen):
        V0 = np.vstack([X_seen, X_unseen[s_id, :][np.newaxis, :]])
        level2probs = rsa.viewer(V0.T, depth, return_all=True)
        for level, _probs in level2probs.items():
            # append the last probs which is the unseen instance
            # compute the success rate of
            level2probslist_unseen[level].append(_probs[:, -1][:, np.newaxis])
    level2probs_unseen = {}
    for level, probs_list in level2probslist_unseen.items():
        level2probs_unseen[level] = np.hstack(probs_list)
    return level2probs_unseen


def _get_df(level2metrics_all):
    """organize the result, produce a readable table (pd.DataFrame)"""
    pd_dict = defaultdict(list)
    for confused_t, level2metrics in level2metrics_all.items():
        for level, metrics in level2metrics.items():
            pd_dict["threshold"].append(confused_t)
            pd_dict["name"].append(level)
            for name, v in metrics.items():
                pd_dict[name].append(v)

    df = pd.DataFrame(pd_dict)
    df = df.sort_values(["threshold", 'name'])
    return df
