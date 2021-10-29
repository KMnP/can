#!/usr/bin/env python3
"""
Utility functions for dataset
"""
import numpy as np
import os
import time
import torch
from collections import Counter


def get_output(folder, data_type="val"):
    pth_file = os.path.join(folder, f"{data_type}.pth")
    start = time.time()
    o_dict = torch.load(pth_file)
    end = time.time()
    elapse = end - start
    if elapse > 5:
        print("\tLoading {} takes {:.2f} seconds.".format(pth_file, elapse))
    return o_dict["logits"].numpy(), o_dict["targets"].numpy()


def imagenet_prior(train_results_folder):
    logits, targets = get_output(train_results_folder, "train")
    n_classes = logits.shape[1]
    prior = np.zeros((n_classes, ))
    for cls_id, count in Counter(targets).items():
        prior[cls_id] = count

    return prior / prior.sum()   # (n_classes, )
