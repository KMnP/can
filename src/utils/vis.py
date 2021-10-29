#!/usr/bin/env python3
"""
Visualize results in the paper
"""
import numpy as np


def inspect(
    result_df, prefix_msg="Inspect best results", metric_names=["top1"]):
    """print out the best the result"""
    print("=" * 80)
    print(prefix_msg)
    print("=" * 80)
    table_headers = [
        "threshold", "metric", "V0", "best top1", "V0-alltop1", "V?-alltop1", "alpha", "agent", "fixed_type"
    ]
    row_format = "{:>12}" * (len(table_headers) - 1 ) + "{:>35}"
    print(row_format.format(*table_headers))

    thresholds = list(set(list(result_df["confused_threshold"])))
    for t in thresholds:
        df_filter = result_df[result_df["confused_threshold"] == t]
        df_filter.reset_index(drop=True, inplace=True)

        for m in metric_names:
            row_idx = np.argmax(df_filter[m])
            best_top1 = np.max(df_filter[m])

            v0_top1 = list(df_filter[df_filter["name"] == "V0"][m])[0]
            best_agent = df_filter["name"][row_idx]
            best_alpha = df_filter["alpha"][row_idx]
            best_f_type = df_filter["fixed_type"][row_idx]

            v0_alltop1 = df_filter[df_filter["name"] == "V0"][f"all-{m}"][0]
            best_p_alltop1 = df_filter[f"all-{m}"][row_idx]
            print(row_format.format(
                t, m, round(v0_top1, 2), round(best_top1, 2),
                round(v0_alltop1, 2), round(best_p_alltop1, 2),
                round(best_alpha, 1), best_agent, best_f_type
            ))


def inspect_f1(
    result_df, prefix_msg="Inspect best results", metric_names=["f1", "mrr"]
):
    """print out the best the result"""
    print("=" * 80)
    print(prefix_msg)
    print("=" * 80)
    table_headers = [
        "threshold", "metric", "V0", "best f1", "alpha", "agent", "fixed_type"
    ]
    row_format = "{:>12}" * (len(table_headers) - 1 ) + "{:>35}"
    print(row_format.format(*table_headers))

    thresholds = list(set(list(result_df["confused_threshold"])))
    for t in thresholds:
        df_filter = result_df[result_df["confused_threshold"] == t]
        df_filter.reset_index(drop=True, inplace=True)
        for m in metric_names:
            row_idx = np.argmax(df_filter[m])
            best_top1 = np.max(df_filter[m])
            if best_top1 == -1:
                continue
            v0_top1 = list(df_filter[df_filter["name"] == "V0"][m])[0]
            best_agent = df_filter["name"][row_idx]
            best_alpha = df_filter["alpha"][row_idx]
            best_f_type = df_filter["fixed_type"][row_idx]

            print(row_format.format(
                t, m, round(v0_top1, 2), round(best_top1, 2),
                round(best_alpha, 1), best_agent, best_f_type
            ))
