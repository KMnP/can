#!/usr/bin/env python3
"""
Utility functions, including evals for dialogue-based RE task
Heavily borrowed from https://github.com/nlpdata/dialogre/blob/master/bert/evaluate.py
"""
import copy
import numpy as np


def get_dev_metrics(dev_probs, dev_probs_c, datadev):
    # get best T2
    bestT2 = find_best_t2(dev_probs, datadev)
    metric_dict = {}
    metric_dict["best_T2"] = bestT2

    devp = get_predict(dev_probs, T2=bestT2)
    precision, recall, f_1 = evaluate(devp, datadev)
    metric_dict["dev_p"] = precision * 100
    metric_dict["dev_r"] = recall * 100
    metric_dict["dev_f1"] = f_1 * 100

    precision, recall, f_1 = evaluate_macro(devp, datadev)
    metric_dict["dev_p_macro"] = precision * 100
    metric_dict["dev_r_macro"] = recall * 100
    metric_dict["dev_f1_macro"] = f_1 * 100

    devp = get_predict(dev_probs_c, T2=bestT2)
    precision, recall, f_1c = evaluate_f1c(devp, datadev)
    metric_dict["dev_c_p"] = precision * 100
    metric_dict["dev_c_r"] = recall * 100
    metric_dict["dev_c_f1"] = f_1c * 100
    return metric_dict


def get_all_metrics(
    dev_probs, dev_probs_c, test_probs, test_probs_c,
    datadev, datatest, bestT2, bestT2_c
):
    metric_dict = {}
    metric_dict["best_T2"] = bestT2
    devp = get_predict(dev_probs, T2=bestT2)
    precision, recall, f_1 = evaluate(devp, datadev)
    metric_dict["dev_p"] = precision * 100
    metric_dict["dev_r"] = recall * 100
    metric_dict["dev_f1"] = f_1 * 100

    precision, recall, f_1 = evaluate_macro(devp, datadev)
    metric_dict["dev_p_macro"] = precision * 100
    metric_dict["dev_r_macro"] = recall * 100
    metric_dict["dev_f1_macro"] = f_1 * 100

    testp = get_predict(test_probs, T2=bestT2)
    precision, recall, f_1 = evaluate(testp, datatest)
    metric_dict["test_p"] = precision * 100
    metric_dict["test_r"] = recall * 100
    metric_dict["test_f1"] = f_1 * 100
    precision, recall, f_1 = evaluate_macro(testp, datatest)
    metric_dict["test_p_macro"] = precision * 100
    metric_dict["test_r_macro"] = recall * 100
    metric_dict["test_f1_macro"] = f_1 * 100

    metric_dict["best_c_T2"] = bestT2_c
    devp = get_predict(dev_probs_c, T2=bestT2_c)
    precision, recall, f_1c = evaluate_f1c(devp, datadev)
    metric_dict["dev_c_p"] = precision * 100
    metric_dict["dev_c_r"] = recall * 100
    metric_dict["dev_c_f1"] = f_1c * 100
    # metric_dict["dev_c_f1_macro"] = evaluate_f1c_macro(devp, datadev) * 100

    testp = get_predict(test_probs_c, T2=bestT2_c)
    precision, recall, f_1c = evaluate_f1c(testp, datatest)
    metric_dict["test_c_p"] = precision * 100
    metric_dict["test_c_r"] = recall * 100
    metric_dict["test_c_f1"] = f_1c * 100
    # metric_dict["test_c_f1_macro"] = evaluate_f1c_macro(testp, datatest) * 100
    return metric_dict


def get_metrics(
    dev_probs, dev_probs_c, test_probs, test_probs_c, datadev, datatest
):
    # get best T2
    bestT2 = find_best_t2(dev_probs, datadev)
    return get_all_metrics(
        dev_probs, dev_probs_c, test_probs,
        test_probs_c, datadev, datatest, bestT2, bestT2
    )


def find_best_t2(dev_probs, datadev):
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        # have to copy the probs since get_predict() modifies the input
        devp = get_predict(copy.deepcopy(dev_probs), T2=T2/100.)
        precision, recall, f_1 = evaluate(devp, datadev)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.
    return bestT2


def get_predict(result, T1=0.5, T2=0.4):
    for i in range(len(result)):
        r = []
        maxl, maxj = -1, -1
        for j in range(len(result[i])):
            if result[i][j] > T1:
                r += [j]
            if result[i][j] > maxl:
                maxl = result[i][j]
                maxj = j
        if len(r) == 0:
            if maxl <= T2:
                r = [36]
            else:
                r += [maxj]
        result[i] = r
    return result


def evaluate(devp, data):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[index]:
                        correct_sys += 1
            for id in devp[index]:
                if id != 36:
                    all_sys += 1
            index += 1

    precision = correct_sys/all_sys if all_sys != 0 else 1
    recall = correct_sys/correct_gt if correct_gt != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1


def evaluate_f1c(devp, data):
    index = 0
    precisions = []
    recalls = []

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0

            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["rid"])):
                if data[i][1][j]["rid"][k] != 36:
                    t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["rid"]) - set([36])

            ex, ey = False, False
            et = {}
            for r in range(36):
                et[r] = r not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set([36])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    for r in range(36):
                        et[r] = True
                for r in range(36):
                    if r in t:
                        if t[r] != "" and t[r] in data[i][0][k].lower():
                            et[r] = True
                    if ex and ey and et[r]:
                        e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            precisions += [correct_sys/all_sys if all_sys != 0 else 1]
            recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1


def evaluate_macro(devp, data, num_classes=36):
    index = 0
    correct_sys_list = [0] * num_classes
    all_sys_list = [0] * num_classes
    correct_gt_list = [0] * num_classes

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt_list[id] += 1
                    if id in devp[index]:
                        correct_sys_list[id] += 1
            for id in devp[index]:
                if id != 36:
                    all_sys_list[id] += 1
            index += 1

    f1_list = []
    p_list = []
    r_list = []
    for correct_sys, all_sys, correct_gt in zip(
        correct_sys_list, all_sys_list, correct_gt_list
    ):
        precision = correct_sys/all_sys if all_sys != 0 else 1
        recall = correct_sys/correct_gt if correct_gt != 0 else 0
        f_1 = 2 * precision * recall / (precision + recall) \
            if precision+recall != 0 else 0
        f1_list.append(f_1)
        p_list.append(precision)
        r_list.append(recall)

    return np.mean(p_list), np.mean(r_list), np.mean(f1_list)


def evaluate_f1c_macro(devp, data):
    f1_list = []
    for i in range(36):
        f1_list.append(_evaluate_f1c_macro(devp, data, i)[-1])
    # print(f1_list)
    return np.mean(f1_list)


def _evaluate_f1c_macro(devp, data, class_idx=0):
    index = 0
    precisions = []
    recalls = []

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0

            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["rid"])):
                # if data[i][1][j]["rid"][k] != 36:
                if data[i][1][j]["rid"][k] == class_idx:
                    t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["rid"]) - set([36]) \
                - set([i for i in range(36) if i != class_idx])

            ex, ey = False, False
            et = {}
            et[class_idx] = class_idx not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set([36]) - set([i for i in range(36) if i != class_idx])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    et[class_idx] = True

                r = class_idx
                if r in t:
                    if t[r] != "" and t[r] in data[i][0][k].lower():
                        et[r] = True
                if ex and ey and et[r]:
                    e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            if correct_sys != 0 or all_sys != 0 or correct_gt != 0:
                precisions += [correct_sys/all_sys if all_sys != 0 else 1]
                recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]

    precision = sum(precisions) / len(precisions) \
        if len(precisions) != 0 else 1
    recall = sum(recalls) / len(recalls)  if len(recalls) != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1
