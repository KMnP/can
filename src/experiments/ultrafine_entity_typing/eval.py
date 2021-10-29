#!/usr/bin/env python3
"""
Utility functions, including evals for ultra-fine entity typing task
Heavily borrowed from https://github.com/uwnlp/open_type/blob/master/scorer.py
and https://github.com/uwnlp/open_type/blob/master/eval_metric.py
"""
import copy
import os
import numpy as np


def compute_metrics(probs, y, root, label_type):
    if label_type == "all":

        sep_probs = copy.deepcopy(probs)
        goal = load_vocab_dict(root, label_type)
        output_index = get_output_index(probs)
        gold_pred = get_gold_pred_str(output_index, y, goal)
        # mrr_val = mrr(probs, y.tolist())
        count, pred_count, avg_pred_count, p, r, f1 = macro(gold_pred)
        _, _, _, p_micro, r_micro, f1_micro = micro(gold_pred)
        p_strict, r_strict, f1_strict = strict(gold_pred)
        metric_dict = dict(
            # mrr=mrr_val,
            average_pred_count=avg_pred_count,
            pred_count=pred_count,
            p_macro=p * 100, r_macro=r * 100, f1=f1 * 100,
            p_micro=p_micro * 100, r_micro=r_micro * 100,
            f1_micro=f1_micro * 100,
            p_strict=p_strict * 100,
            r_strict=r_strict * 100, f1_strict=f1_strict * 100,
        )
        for l_type in ["coarse", "fine", "ultra-fine"]:
            _m = fg_metrics(sep_probs, y, root, l_type)
            for k, v in _m.items():
                metric_dict[k+"-" + l_type] = v
        return metric_dict
    return fg_metrics(probs, y, root, label_type)


def stratify(all_labels, types):
    """
    Divide label into three categories.
    """
    coarse = types[:9]
    fine = types[9:130]
    return (
        [l for l in all_labels if l in coarse],
        [l for l in all_labels if ((l in fine) and (not l in coarse))],
        [l for l in all_labels if (not l in coarse) and (not l in fine)]
    )


def fg_metrics(probs, y, root, label_type):
    types = load_vocab_dict(root, "all")
    output_index = get_output_index(probs)
    gold_pred = get_gold_pred_str(output_index, y, types)

    coarse_true_and_predictions = []
    fine_true_and_predictions = []
    finer_true_and_predictions = []
    for k, v in gold_pred:
        coarse_gold, fine_gold, finer_gold = stratify(k, types)
        coarse_pred, fine_pred, finer_pred = stratify(v, types)
        coarse_true_and_predictions.append((coarse_gold, coarse_pred))
        fine_true_and_predictions.append((fine_gold, fine_pred))
        finer_true_and_predictions.append((finer_gold, finer_pred))

    if label_type == "coarse":
        true_and_predictions = coarse_true_and_predictions
    elif label_type == "fine":
        true_and_predictions = fine_true_and_predictions
    elif label_type == "ultra-fine":
        true_and_predictions = finer_true_and_predictions

    count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
    _, _, _, p_micro, r_micro, f1_micro = micro(true_and_predictions)
    p_strict, r_strict, f1_strict = strict(true_and_predictions)
    metric_dict = dict(
        # mrr=-1,
        average_pred_count=avg_pred_count,
        pred_count=pred_count,
        p_macro=p * 100, r_macro=r * 100, f1=f1 * 100,
        p_micro=p_micro * 100, r_micro=r_micro * 100,
        f1_micro=f1_micro * 100,
        p_strict=p_strict * 100,
        r_strict=r_strict * 100, f1_strict=f1_strict * 100,
    )

    return metric_dict


def compute_metrics_old(probs, y, root, label_type):
    goal = load_vocab_dict(root, label_type)
    output_index = get_output_index(probs)
    gold_pred = get_gold_pred_str(output_index, y, goal)

    if label_type == "all":
        mrr_val = mrr(probs, y.tolist())
    else:
        mrr_val = -1
    # print('mrr_value: ', mrr_val)

    count, pred_count, avg_pred_count, p, r, f1 = macro(gold_pred)
#     perf_total = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(
#         count, avg_pred_count, p * 100, r * 100, f1 * 100)
#     print(perf_total)
    return dict(
        mrr=mrr_val, average_pred_count=avg_pred_count, pred_count=pred_count,
        precision=p * 100, recall=r * 100, f1=f1 * 100
    )


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    precision = recall = correct_num / num_entities
    return precision, recall, f1(precision, recall)


def macro(true_and_prediction):
    # loose macro score from hinton
    num_examples = len(true_and_prediction)
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_label_count > 0:
        recall = r / gold_label_count
    avg_elem_per_pred = pred_label_count / pred_example_count
    return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


def micro(true_and_prediction):
    num_examples = len(true_and_prediction)
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    pred_example_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    if pred_example_count == 0:
        return num_examples, 0, 0, 0, 0, 0
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    avg_elem_per_pred = num_predicted_labels / pred_example_count
    return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


def mrr(dist_list, gold):
    """
    dist_list: list of list of label probability for all labels.
    gold: list of gold indexes.
    Get mean reciprocal rank. (this is slow, as have to sort for 10K vocab)
    """
    mrr_per_example = []
    dist_arrays = np.array(dist_list)
    dist_sorted = np.argsort(-dist_arrays, axis=1)
    for ind, gold_i in enumerate(gold):
        gold_i_where = [i for i in range(len(gold_i)) if gold_i[i] == 1]
        rr_per_array = []
        sorted_index = dist_sorted[ind, :]
        for gold_i_where_i in gold_i_where:
            for k in range(len(sorted_index)):
                if sorted_index[k] == gold_i_where_i:
                    rr_per_array.append(1.0 / (k + 1))
        mrr_per_example.append(np.mean(rr_per_array))
    return sum(mrr_per_example) * 1.0 / len(mrr_per_example)


def load_vocab_dict(root, label_type):
    vocab_file_name = os.path.join(root, "release/ontology/types.txt")
    with open(vocab_file_name) as f:
        types = [x.strip() for x in f.readlines()]
    if label_type == "coarse":
        types = types[:9]
    elif label_type == "fine":
        types = types[9:130]
    elif label_type == "ultra-fine":
        types = types[130:]
    return types
    # file_content = dict(zip(range(0, len(types)), types))
    # return file_content


def get_output_index(outputs):
    """
    Given outputs from the decoder, generate prediction index.
    :param outputs:
    :return:
    """
    pred_idx = []
    for single_dist in outputs:
        single_dist = single_dist[:]
        arg_max_ind = np.argmax(single_dist)
        pred_id = [arg_max_ind]
        pred_id.extend(
          [i for i in range(len(single_dist)) if single_dist[i] > 0.5 and i != arg_max_ind])
        pred_idx.append(pred_id)
    return pred_idx


def get_gold_pred_str(pred_idx, gold, goal):
    """
    Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
    """
    id2word_dict = goal
    gold_strs = []
    for gold_i in gold:
        gold_i = list(gold_i)
        gold_strs.append([id2word_dict[i] for i in range(len(gold_i)) if gold_i[i] == 1])
    pred_strs = []
    for pred_idx1 in pred_idx:
        pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
    return list(zip(gold_strs, pred_strs))
