#!/usr/bin/env python3
"""
Heavily borrowed from https://github.com/nlpdata/dialogre/blob/master/bert/evaluate.py
"""
import numpy as np
import os
import json

from ...utils.io_utils import read_json


def get_output(data_dir, eval_type, lang, bert_type, m_type):
    """
    Args:
        data_dir: path to the data
        eval_type: "dev" or "test"
        lang: "cn" or "en"
        bert_type: "bert", "berts"
        m_type: "", "c", "_2" "_2c"
    Returns:
        gold: dict
        probs: numpy.ndarray
    """
    gold = get_gold(os.path.join(
        data_dir, "data_v2", lang, "data", f"{eval_type}.json"))

    if lang == "en":
        lang = "enu"
    model_folder = f"{bert_type}_f1_{lang}_finetuned{m_type}"
    probs = get_probs(os.path.join(
         data_dir, "dialogRElogits(v2)",
         model_folder, f"logits_{eval_type}.txt")
    )
    return gold, probs


def get_gold(filename):
    with open(filename, "r", encoding='utf8') as f:
        gold = json.load(f)
    for i in range(len(gold)):
        for j in range(len(gold[i][1])):
            for k in range(len(gold[i][1][j]["rid"])):
                gold[i][1][j]["rid"][k] -= 1

    return gold


def get_probs(filename):
    """read logits and convert to probabilities"""
    result = []
    with open(filename, "r") as f:
        l = f.readline()
        while l:
            l = l.strip().split()
            for i in range(len(l)):
                l[i] = float(l[i])
            result += [l]
            l = f.readline()
    result = np.asarray(result)
    return 1 / (1 + np.exp(-result))


def count_labels(root, lang):
    """
    in this multilabel cls setting, the positive / negative ratio is similar
    lang: train /  dev / test
    cn:    5.58 / 5.85 / 6.06
    en     6.74 / 7.07 / 7.4
    """
    filename = os.path.join(root, "data_v2", lang, "data", "train.json")
    gold_dev = get_gold(filename)
    correct_gt = 0
    for i in range(len(gold_dev)):
        for j in range(len(gold_dev[i][1])):
            for id in gold_dev[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
    total = len(gold_dev) * 36
    prior = np.array([correct_gt, total - correct_gt])
    return prior / prior.sum()
