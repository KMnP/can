#!/usr/bin/env python3
"""
ImageNet experiments
"""
import pandas as pd
import multiprocessing

from collections import defaultdict
from contextlib import contextmanager

from ..rsa.compute_rsa import get_rsa
from ..utils import imagenet_utils as data_utils
from ..utils.io_utils import save_or_append_df


def _get_df(level2metrics):
    """organize the result, produce a readable table (pd.DataFrame)"""
    pd_dict = defaultdict(list)
    for level, metrics in level2metrics.items():
        pd_dict["name"].append(level)
        for name, v in metrics.items():
            pd_dict[name].append(v)

    df = pd.DataFrame(pd_dict)
    df = df.sort_values(["confused_threshold", 'name'])
    return df


def get_metric(probs, targets, alpha, confused_t, fixed_type, prior_list):
    level2metrics = get_rsa(
        probs, targets, alpha,
        confused_t, fixed_type, prior_list, depth=5
    )
    return _get_df(level2metrics)


def get_metric_unpack(args):
    return get_metric(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def tune_imagenet_blur(root, image_folder):
    """spawn num_alphas processes"""
    arch = image_folder.split("/")[-1]
    prior_list = data_utils.imagenet_prior(image_folder)

    alpha_list = [i * 0.1 for i in range(1, 10)]
    alpha_list += [i for i in range(1, 10)]
    fixed_type = "test_confident_lite_stratefy"
    confused_t = 0.75

    arg_list = []
    for img_type_out in [
        "original", "blur2", "blur4", "blur8", "blur16", "blur32"
    ]:
        probs, targets = data_utils.get_output(
            image_folder + "/" + img_type_out, "val")

        # construct the arg_list
        arg_list.extend([
            (probs, targets, a,  confused_t, fixed_type, prior_list) \
            for a in alpha_list
        ])

    # spawn process
    print(f"\t\tgoing to spawn {len(arg_list)} processes")
    with poolcontext(processes=len(arg_list)) as pool:
        df_list = pool.map(get_metric_unpack, iter(arg_list))

    df = pd.concat(df_list, ignore_index=True)
    out_path = f"{root}/imagenet-{arch}-blur.csv"
    save_or_append_df(out_path, df)


def main(data_root):
    root = f"{data_root}/visual"
    # do different blur effect
    # resnet 50, "test_confident_lite_stratefy"
    tune_imagenet_blur(root, "resnet50")

