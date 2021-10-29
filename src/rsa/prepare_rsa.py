#!/usr/bin/env python3
"""
some functions to prepare for the RSA model
"""
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


# ==============================
# split the evaluation instances
# ==============================
def entropy(probs, base):
    """probs: np.array, (n_classes,), sum to 1"""
    exponent = np.log(sanitycheck_probs(probs)) / np.log(base)
    return - np.multiply(exponent, probs).sum()


def topk_entropy(probs, k):
    best_k = np.sort(sanitycheck_probs(probs))[-k:]
    best_k = best_k / best_k.sum()
#     print(best_k)
    return entropy(best_k, k)


def sanitycheck_probs(probs):
    # check if there is any 0 values, if so, add a eps to that position
    probs = np.array(probs)
    return probs + (probs == 0) * 1e-16


def get_confused_instances_topk(probs, threshold=0.9, max_k=None):
    """
    split logits into confident cases and ambigious cases
    a ambigious/confused case is defined as:
        given a probs of length n, the topk entropy of probs > threshold,f
        or all k in the range of (2, n)
    for a distribution with n_classes, entropy in [0, ln(n_classes)].
    options for threshold:
    - set the base of log to n_classes, so the range of entropy is [0, 1]. the threshold is a scalar from 0 - 1
    """
    n_samples, n_labels = probs.shape

    if max_k is None or max_k > n_labels + 1:
        max_k = n_labels + 1
    confused_ids = set()
    confidence_ids = []
    for k in range(2, max_k):
        for i in range(n_samples):
            ent = topk_entropy(probs[i, :], k)
            if ent >= threshold:
                confused_ids.add(i)
    confidence_ids = [i for i in range(n_samples) if i not in confused_ids]
    confused_ids = list(confused_ids)
    return confused_ids, confidence_ids


def get_confused_instances(probs, threshold=0.5):
    """
    split logits into confident cases and ambigious cases
    A confident case is defined as entropy <= some threshold.
    for a distribution with n_classes, entropy in [0, ln(n_classes)].
    options for threshold:
     - set the base of log to n_classes, so the range of entropy is [0, 1]. the threshold is a scalar from 0 - 1
    """
    n_labels = probs.shape[1]

    confused_ids = []
    confidence_ids = []
    for i in range(probs.shape[0]):
        ent = entropy(probs[i, :], base=n_labels)
        if ent >= threshold:
            confused_ids.append(i)
        else:
            confidence_ids.append(i)
    return confused_ids, confidence_ids


def split_seenunseen(tgts, probs):
    """split test set into seen and unseen"""
    X_seen, X_unseen, y_seen, y_unseen = train_test_split(
        probs, tgts, test_size=0.5, random_state=3, stratify=tgts)
    return X_seen, X_unseen, y_seen, y_unseen


def filter_logits(probs, targets, selected_ids):
    return probs[selected_ids, :], [targets[i] for i in selected_ids]


# ==============================
# fixed set
# ==============================
def create_fixed_sets_dict(fixed_set, confused_set, prior_list):
    """
    create the all the fixed set at once
    """
    fixed_set_dict = {}
    fixed_types = [
        "test_confident_lite_stratefy",
        "identity",
    ]
    for fixed_type in fixed_types:
        fixed_set_dict[fixed_type] = get_fixed_set(
            confused_set, fixed_set, prior_list, fixed_type)
    return fixed_set_dict


def get_fixed_set(probs_confuse, probs_confident, prior_list, fixed_type):
    """
    get fixed set: n_fixed x n_classes
    fixed_type choice:
    "test_confident", "identity", "random",
    "eye_prior", "eye_uniform", "random_peak_prior", "random_peaklite_prior",
    "random_peak_uniform", "random_peaklite_uniform"
    """
    np.random.seed(32)  # set random seed

    n_confused, n_classes = probs_confuse.shape

    n_samples = max([n_classes * 5, 1000])

    if fixed_type == "test_confident":
        ## choice 1: use confident cases in test set
        # if there is no confident case, use identity matrix
        n_confident = probs_confident.shape[0]
        if n_confident == 0:
            return np.eye(n_classes)

        return probs_confident

    if fixed_type == "test_confident_lite_stratefy":
        ## choice 1: use a subset of confident cases in test set
        n_confident = probs_confident.shape[0]
        if n_confident == 0:
            return np.eye(n_classes)
        elif n_confident > n_samples:
            # train_test_split not working for some classes with less than 2
            # filter out classes that only have 1 instances.
            predicted_labels = np.argmax(probs_confident, axis=1)
            rare_classes = []
            for cls_idx, count in Counter(predicted_labels).items():
                if count < 2:
                    rare_classes.append(cls_idx)
            rare_idxes = [
                idx for idx, l in enumerate(predicted_labels)
                if l in rare_classes
            ]
            if len(rare_idxes) > 0:
                rare_probs = probs_confident[rare_idxes, :]
                probs_confident = probs_confident[
                    [i for i in range(n_confident) if i not in rare_idxes], :]
            else:
                rare_probs = None

            _, probs_confident = train_test_split(
                probs_confident, test_size=n_samples,
                random_state=3, stratify=np.argmax(probs_confident, axis=1)
            )
            if rare_probs is not None:
                probs_confident = np.vstack([probs_confident, rare_probs])
        return probs_confident

    if fixed_type == "identity":
        ## choice 2: a fixed identity matrix
        return np.eye(n_classes)

    if fixed_type == "random":
        # choice 3: random matrix
        probs_fixed = np.random.rand(n_samples, n_classes)
        probs_fixed = probs_fixed / np.sum(probs_fixed, 1, keepdims=True)
        return probs_fixed

