import os
import pickle
import numpy as np

from collections import Counter


def get_output(root, test_type, label_type="all"):
    if root.endswith("denoise"):
        test_type = test_type + "_probs"
    with (open(f"{root}/best_model/{test_type}.p", "rb")) as openfile:
        data = pickle.load(openfile)

    logits = []
    for out_b in data['pred_dist']:
        logits.append(out_b[np.newaxis, :])
    logits = np.vstack(logits)

    labels = []
    for l in data['gold_id_array']:
        labels.append(l[np.newaxis, :])
    labels = np.vstack(labels).astype(int)

    if label_type == "all":
        return logits, labels

    if label_type == "coarse":
        return logits[:, :9], labels[:, :9]
    if label_type == "fine":
        return logits[:, 9:130], labels[:, 9:130]
    if label_type == "ultra-fine":
        return logits[:, 130:], labels[:, 130:]

    raise ValueError(f"Unsupported label type: {label_type}")


def count_labels(root, test_file):
    if root.endswith("denoise"):
        test_file = test_file + "_probs"
    with (open(f"{root}/best_model/{test_file}.p", "rb")) as openfile:
        data = pickle.load(openfile)
    prior = np.zeros((2, ))
    for l in data['gold_id_array']:
        for _l, c in Counter(l).items():
            prior[int(_l)] += c
    # the first class should be the possitive one,
    # and second is the negative one
    return prior[::-1] / prior.sum()   # (n_classes, )


def to_binary(a):
    """
    rearrange probabilities from multilabel cls to binary classifications
    (num_samples, num_labels) -> (num_samples * num_labels, 2)
    Args:
        a: np.ndarray of shape (num_samples, num_labels)
    returns:
        c: np.ndarray of shape (num_samples * num_labels, 2)
    """
    b = 1 - a
    c = np.concatenate(
        [a[:, :, np.newaxis], b[:, :, np.newaxis]], -1
    ).reshape(-1, 2)
    return c


def to_multilabel(a, num_samples):
    """
    rearrange the probability back to the original multilabel format
    """
    return a[:, 0].reshape(num_samples, -1)
