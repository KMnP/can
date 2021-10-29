#!/usr/bin/env python3
"""
Functions for evaluations for single-label multiclass tasks
"""
import sklearn.metrics as metrics
import numpy as np
from collections import defaultdict


def get_level2metric(level2probs, target):
    level2metrics = {}
    for k, _probs in level2probs.items():
        if k.startswith("C"):  # only look for viewer
            continue
        if k not in level2metrics:
            level2metrics[k] = defaultdict(list)

        top1 = top_n_accuracy(_probs.T, target, 1)
        top5 = top_n_accuracy(_probs.T, target, 5)

        level2metrics[k]["top1"].append(top1 * 100)
        level2metrics[k]["top5"].append(top5 * 100)

    return level2metrics


def accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    accuracy = metrics.accuracy_score(y_true, y_preds) * 100
    error = 100 - accuracy
    return accuracy, error


def top_n_accuracy(y_probs, truths, n=1):
    # y_prob: (num_images, num_classes)
    # truth: (num_images, num_classes) multi/one-hot encoding
    best_n = np.argsort(y_probs, axis=1)[:, -n:]
    if isinstance(truths, np.ndarray) and truths.shape == y_probs.shape:
        ts = np.argmax(truths, axis=1)
    else:
        # a list of GT class idx
        ts = truths

    num_input = y_probs.shape[0]
    successes = 0
    for i in range(num_input):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / num_input


def topk_success(y_prob, truth, k):
    """
    Calculate the number of success case given top k predictions
    Args:
        y_prob: np.ndarray of shape (n_classes, )
        truth: int
    Returns:
        sucess: integer 1 or 0, 1 denotes success
    """
    best_n = np.argsort(y_prob)[-k:]
    if truth in best_n:
        return 1
    return 0


def aggregate_success(successes):
    """
    compute the accurarcy given _successes
    args:
        successes: list of np.ndarray of (n_samples,)
    returns:
        accurarcy: float
    """
    return sum(successes) / len(successes), sum(successes)


def topk_accuracy_all(y_probs, targets, successes, k):
    """
    compute topk accuracy given a list of sucesses indicator for one subset of input, and probs and targets for the rest of the input
    """
    best_k = np.argsort(y_probs, axis=1)[:, -k:]
    num_input = y_probs.shape[0]
    num_successes = 0
    for i in range(num_input):
        if targets[i] in best_k[i, :]:
            num_successes += 1
    return (num_successes + sum(successes)) / (len(successes) + num_input)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


#########################
# calibration-related
#########################
def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece


def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))

    return max(cal_errors)


def get_bin_info(conf, pred, true, bin_size = 0.1):

    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)


    return accuracies, confidences, bin_lengths
