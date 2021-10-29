#!/usr/bin/env python3
"""
randomly construct some arrays and save it first.
A: (NUM_SIMULATIONS, n-1, n) for each n, where last dim of A always have some entropy range (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)
b: (NUM_SIMULATIONS, NUM_B, n) for each n, where each b has topk entropy of range: (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1),
prior: (NUM_SIMULATIONS, n)


Simulate experiments
construct L with size n x n, and given alpha and a random prior.
range
- n: 2 - 50
- A: high entropy / low entropy, need 1 class
- b: high entropy / low entropy, (for each A, generate 100 different b)
- alpha: 0 - 10
- random prior: uniform - slightly lower entropy than uniform
For each situation, generate 200 times
"""
import os
import functools
import multiprocessing
import numpy as np

from contextlib import contextmanager
from tqdm import tqdm, trange

from ..rsa.rsa import RSA
from ..rsa.prepare_rsa import get_confused_instances_topk
from ..utils.io_utils import read_npy_dict
# np.random.seed(32)  # set random seed
NUM_SIMULATIONS = 200
# given a fixed A, the number of randomly generated B of any condition
NUM_B = 100

THRESHOLDS2LEVELS = {0.75: "high2", 0.5: "high1", 0.25: "low1", 0: "low2"}
# {0: (0.5, 0.75), 1: (0.75, 0.9), 2: (0.9, 0.99), 4: (0.999, 1)}
ENTROPY_THRESHOLDS = [0.75, 0.5, 0.25, 0.0]
FG_THRESHODS = [0.999, 0.99, 0.9, None]


#===================== generators related methods ======================
def generate_b(n, num_peaks=2, confident_offset=100):
    """
    # SINGLE PEAK, NOT m PEAKS
    generate b of size NUM_SIMULATIONS x NUM_B x n
    Args:
        n: int, length of b
        confident_offset: float, so the peak position value is  higher than the others.

        num_peaks: how many peaks are in the distribution.
    Returns:
        b_array: np.ndarray of size (NUM_SIMULATIONS, NUM_B, n), each generated distribution has high topk entropy (k = num_peaks)
    """
    # normalize by n so each n will have a similar range entropy
    assert num_peaks <= n and num_peaks > 0
    confident_offset = confident_offset  # * n
    offset = 10

    # b_array = np.random.rand(NUM_SIMULATIONS, NUM_B, n) / offset + confident_offset
    b_array = np.random.rand(NUM_SIMULATIONS, NUM_B, n) / confident_offset
    if num_peaks > 1:
        m_clses = np.zeros((NUM_SIMULATIONS, NUM_B, num_peaks), dtype=int)
        for i in range(NUM_SIMULATIONS):
            for j in range(n-1):
                m_clses[i, j, :] = np.random.choice(
                    n, size=num_peaks, replace=False)
    else:
        m_clses = np.random.choice(n, size=(NUM_SIMULATIONS, NUM_B, num_peaks))

    # print(m_clses)
    for i in range(NUM_SIMULATIONS):
        for j in range(NUM_B):
            b_array[i, j, m_clses[i, j]] = confident_offset

    b_array = b_array / np.sum(b_array, 2, keepdims=True)
    return b_array


def generate_A(n, confident_offset=0):
    """
    generate NUM_SIMULATIONS A of size (n-1, n).
    Each row of A has a higher
    The classes of each row of A is randomly decided.
    Args:
        n: int, size of A
        confident_offset: float, of range (0, 1) so the peak position value is 1/offset higher than the others.
            Higher the value of confident_offset,
            higher the entropies of each row of A.
    Returns:
        A: np.ndarray of size (NUM_SIMULATIONS, n-1, n)
    """
    # normalize by n so each n will have a similar range entropy
    offset = confident_offset  # / n

    A = np.random.rand(NUM_SIMULATIONS, n-1, n) * offset + 1
    # random choose? or use prior?
    m_clses = np.random.choice(n, size=(NUM_SIMULATIONS, n-1))
    # print(m_clses)
    for i in range(NUM_SIMULATIONS):
        for j in range(n-1):
            # A[i, j, m_clses[i, j]] = 2  #1 / offset
            # A[i, j, m_clses[i, j]] = n / offset
            A[i, j, m_clses[i, j]] = n / 4  # 10

    A = A / np.sum(A, 2, keepdims=True)
    return A


def generate_prior(n):
    """
    generate prior distribution q that has minimum entropy of something
    Args:
        n: int, length of q
        is_confident: bool, if True
    Returns:
        prior: np.ndarray of size (NUM_SIMULATIONS, n)
    """
    # some small value to pertube the uniform distribution
    delta = np.random.rand(NUM_SIMULATIONS, n) / n

    prior = np.ones((NUM_SIMULATIONS, n)) / n + delta
    prior = prior / np.sum(prior, 1, keepdims=True)
    return prior


def generate_alpha():
    alpha_list = [i * 0.1 for i in range(1, 10)]
    alpha_list += [i for i in range(1, 10)]
    return alpha_list


def check_full(entropy2array, max_num, default_thresholds=True):
    if default_thresholds:
        threshold_list = ENTROPY_THRESHOLDS
    else:
        threshold_list = FG_THRESHODS[:-1]
    if len(entropy2array) < len(threshold_list):
        return False
    a = [entropy2array[l].shape[0] for l in threshold_list if l in entropy2array]
    for c in a:
        if c < max_num:
            return False
    return True


def _update_buckets(new_array, entropy2array, entropy_level):
    if entropy_level in entropy2array:
        entropy2array[entropy_level] = np.vstack(
            [new_array, entropy2array[entropy_level]])
    else:
        entropy2array[entropy_level] = new_array
    return entropy2array


def update_buckets(b_array, entropy2array, default_thresholds=True):
    if default_thresholds:
        threshold_list = ENTROPY_THRESHOLDS
    else:
        threshold_list = FG_THRESHODS

    n1, n2, _ = b_array.shape
    b_array = np.reshape(b_array, (n1 * n2, -1))
    n_rows, n_labels = b_array.shape
    max_k = min(10, n_labels)

    for threshold in threshold_list[:-1]:
        if b_array is None:
            break
        entropy_level = threshold
        # from high to low
        confused_ids, confidence_ids = get_confused_instances_topk(b_array, threshold, max_k)

        # put confused_ids to buckets
        if len(confused_ids) > 0 and len(entropy2array.get(entropy_level, [])) < n_rows:
            entropy2array = _update_buckets(b_array[confused_ids, :], entropy2array, entropy_level)
        if len(confidence_ids) > 0:
            b_array = b_array[confidence_ids, :]
        else:
            b_array = None
    if default_thresholds:
        low_entropy_level = threshold_list[-1]
        if b_array is not None and len(entropy2array.get(low_entropy_level, [])) < n_rows:
            entropy2array = _update_buckets(b_array, entropy2array, low_entropy_level)
    return entropy2array


def generate_b_buckets(n, default_thresholds=True):
    """
    generate a list of b of size NUM_SIMULATIONS x NUM_B x n
    Args:
        n: int, length of b
    Returns:
        entropy2array: np.ndarray of size (NUM_SIMULATIONS, NUM_B, n), each generated distribution has high topk entropy (k = num_peaks)
    """
    max_num = NUM_SIMULATIONS * NUM_B
    # generate a dict
    entropy2array = {}
    # for _ in trange(1000):
    #     if check_full(entropy2array, max_num, default_thresholds):
    #         break
    #     # random generate distribution for some times
    #     b_array = np.random.rand(NUM_SIMULATIONS, NUM_B, n)
    #     b_array = b_array / np.sum(b_array, 2, keepdims=True)
    #     entropy2array = update_buckets(
    #         b_array, entropy2array, default_thresholds)

    offset_list = [0.001, 0.01, 0.0001, 0.000001, 0.1, 1, 10, 100, 1000, 10000]
    offset_list = [
        0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.2, 0.3, 2, 3, 4, 5, 6]
    # for _ in trange(200):
    for _ in trange(2):
        if check_full(entropy2array, max_num, default_thresholds):
            break
        for n_peaks in range(1, min(10, n) + 1):
            for offset in offset_list:
                if check_full(entropy2array, max_num, default_thresholds):
                    break
                # generate bunch of bs
                b_array = generate_b(n, n_peaks, offset)
                # filter the resulting b into different buckets
                # then update the buckets
                entropy2array = update_buckets(
                    b_array, entropy2array, default_thresholds)

    # reshape entropy2array
    for l, b_list in entropy2array.items():
        n_rows = b_list.shape[0]
        if n_rows > max_num:
            ids = np.random.choice(n_rows, size=max_num, replace=False)
            entropy2array[l] = b_list[ids, :]
            entropy2array[l] = b_list[ids, :].reshape(-1, NUM_B, n)
        elif n_rows == max_num:
            entropy2array[l] = b_list.reshape(-1, NUM_B, n)
    return entropy2array


def generate_A_buckets(n):
    """
    generate a list of b of size NUM_SIMULATIONS x n-1 x n
    Args:
        n: int, length of b
    Returns:
        entropy2array: np.ndarray of size (NUM_SIMULATIONS, NUM_B, n), each generated distribution has high topk entropy (k = num_peaks)
    """
    max_num = NUM_SIMULATIONS * (n-1)
    # generate a dict
    entropy2array = {}
    # for _ in trange(2000):
    #     if check_full(entropy2array, max_num):
    #         break
    #     # random generate distribution for some times
    #     b_array = np.random.rand(NUM_SIMULATIONS, n-1, n)
    #     b_array = b_array / np.sum(b_array, 2, keepdims=True)
    #     entropy2array = update_buckets(b_array, entropy2array)

    # offset_list = [0.001, 0.01, 0.0001, 0.000001, 0.1, 1, 10, 100, 1000, 10000]
    # offset_list = [0.001, 0.000001, 0.0000001, 0.0001, 0.01, 0.1, 1]  # 0.0=0.25
    offset_list = [
        # 0.000001, 0.0000001, 0.0001, 0.1, 1,
        # 0.01, 0.02, 0.003, 0.004, 0.05, 0.06, 0.07,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8,
        0.001, 0.002, 0.03, 0.04, 0.005, 0.006, 0.007,
    ]  # 0.5
    # offset_list = [1, 2, 3, 1.5] # 0.75
    for _ in trange(2):
        if check_full(entropy2array, max_num):
            break
        for offset in offset_list:
            if check_full(entropy2array, max_num):
                break
            # generate bunch of arrays
            b_array = generate_A(n, offset)
            # filter the resulting b into different buckets
            # then update the buckets
            entropy2array = update_buckets(b_array, entropy2array)

    # for _ in trange(10000):
    #     if check_full(entropy2array, max_num):
    #         break
    #     # random generate distribution for some times
    #     b_array = np.random.rand(NUM_SIMULATIONS, n-1, n)
    #     b_array = b_array / np.sum(b_array, 2, keepdims=True)
    #     entropy2array = update_buckets(b_array, entropy2array)

    # for _ in trange(1000):
    #     if check_full(entropy2array, max_num):
    #         break
    #     for offset in offset_list:
    #         if check_full(entropy2array, max_num):
    #             break
    #         # generate bunch of arrays
    #         b_array = generate_A(n, offset)
    #         # filter the resulting b into different buckets
    #         # then update the buckets
    #         entropy2array = update_buckets(b_array, entropy2array)
    # reshape entropy2array
    for l, b_list in entropy2array.items():
        n_rows = b_list.shape[0]
        if n_rows > max_num:
            ids = np.random.choice(n_rows, size=max_num, replace=False)
            entropy2array[l] = b_list[ids, :].reshape(-1, n-1, n)
        elif n_rows == max_num:
            entropy2array[l] = b_list.reshape(-1, n-1, n)
    return entropy2array


def generate_n(n, data_root):
    b = {}
    # b = generate_b_buckets(n, False)
    # print("fine-grained b")
    # for k, v in b.items():
    #     print(f"{k}: {len(v)}")

    a = generate_A_buckets(n)
    print(f"A: {n-1} x {n}")
    for k, v in a.items():
        print(f"{k}: {len(v)}")

    b.update(generate_b_buckets(n, True))
    print(f"b: {n}")
    for k, v in b.items():
        print(f"{k}: {len(v)}")

    prior = generate_prior(n)

    out_path = os.path.join(data_root, f"{n}_more_more_more.npy")
    np.save(out_path, {"A": a, "b": b, "prior": prior})


def generate_A_levels(n, data_root):
    a = generate_A_buckets(n)
    print(f"A: {n-1} x {n}")
    for k, v in a.items():
        print(f"{k}: {len(v)}")

    out_path = os.path.join(data_root, f"{n}_A_more.npy")
    np.save(out_path, {"A": a})
    print("Saved to {}".format(out_path))


#============================= simulate ==============================
def compute_pragmatic(A, b, prior, alpha):
    """
    Args
        A: (n-1, n)
        b: (n, )
        prior: (n, )
        alpha: scalar
    Returns:
        pragmatic_b: (n, )
    """
    n_classes = A.shape[1]
    rsa = RSA(
        n_classes, n_classes,
        alpha=alpha, prior=prior,
        use_lexicon_as_base=True,
        normalize_use_base=False
    )
    V0 = np.vstack([A, b[np.newaxis, :]]).T
    level2probs = rsa.viewer(V0, 3, return_all=True)
    pragmatic_b1 = level2probs["V1"][:, -1]
    pragmatic_b2 = level2probs["V2"][:, -1]
    return pragmatic_b1, pragmatic_b2


def EAG(b, pragmatic_b, prior):
    """
    compute expected accuracy gain
    Args
        b: (n, )
        pragmatic_b: (n, )
    return:
        eag: float
    """
    n_classes = b.shape[0]
    return np.dot(pragmatic_b - b, prior) / n_classes


def relative_EAG(b, pragmatic_b, prior):
    """
    compute expected accuracy gain
    Args
        b: (n, )
        pragmatic_b: (n, )
    return:
        eag: float
    """
    return np.dot(pragmatic_b - b, prior) / np.dot(b, prior)


def change_prediction(b, pragmatic_b):
    """if the prediction changes, return 1, else 0"""
    return int(np.argmax(b) != np.argmax(pragmatic_b))


def compute_EAG(A, b_array, prior, alpha):
    pragmatic_bs = [compute_pragmatic(A, b, prior, alpha) for b in b_array]
    # accu_gains = [
    #     EAG(b, pragmatic_b, prior) \
    #     for b, pragmatic_b in zip(b_array, pragmatic_bs)
    # ]  # [1, NUM_B]
    relative_accu_gains = [
        relative_EAG(b, pragmatic_b[0], prior) \
        for b, pragmatic_b in zip(b_array, pragmatic_bs)
    ]  # [1, NUM_B]
    flips = [
        change_prediction(b, pragmatic_b[0]) \
        for b, pragmatic_b in zip(b_array, pragmatic_bs)
    ]  # [1, NUM_B]

    relative_accu_gains_l2 = [
        relative_EAG(b, pragmatic_b[1], prior) \
        for b, pragmatic_b in zip(b_array, pragmatic_bs)
    ]  # [1, NUM_B]
    flips_l2 = [
        change_prediction(b, pragmatic_b[1]) \
        for b, pragmatic_b in zip(b_array, pragmatic_bs)
    ]  # [1, NUM_B]
    return (
        np.array(relative_accu_gains), np.array(flips),
        np.array(relative_accu_gains_l2), np.array(flips_l2)
    )


def simulate(n, A_array, b_array, prior):
    """
    simulate given a n x n L_{d-1} matrix and value of alpha
    Returns:
        accu_gains: shape (num_alphas, NUM_SIMULATIONS, NUM_Bs)
    """
    # generate A, b, q
    alpha_list = generate_alpha()

    relative_accu_array = np.zeros((len(alpha_list), NUM_SIMULATIONS, NUM_B))
    flip_array = np.zeros((len(alpha_list), NUM_SIMULATIONS, NUM_B))
    relative_accu_array_l2 = np.zeros(
        (len(alpha_list), NUM_SIMULATIONS, NUM_B))
    flip_array_l2 = np.zeros((len(alpha_list), NUM_SIMULATIONS, NUM_B))
    for a_idx, alpha in enumerate(alpha_list):
        for i in range(NUM_SIMULATIONS):
            relative_accu_array[a_idx, i, :], flip_array[a_idx, i, :], relative_accu_array_l2[a_idx, i, :], flip_array_l2[a_idx, i, :] = compute_EAG(
                A_array[i, :, :], b_array[i, :, :], prior[i, :], alpha)
    return (
        relative_accu_array, flip_array,
        relative_accu_array_l2, flip_array_l2
    )


def summary(accu_array, flip_array):
    """evaluate how many are positive accuracy gain"""
    n_alpha, n_sim, n_b = accu_array.shape
    gain_probs = np.zeros((n_alpha, n_sim))
    flipped_gain_probs = np.zeros((n_alpha, n_sim))
    nonflipped_gain_probs = np.zeros((n_alpha, n_sim))
    for a in range(n_alpha):
        for s in range(n_sim):
            _sum = np.sum(np.logical_and(
                accu_array[a, s, :] > 0, flip_array[a, s, :] > 0))
            flipped_gain_probs[a, s] = _sum / n_b

            _sum = np.sum(np.logical_and(
                accu_array[a, s, :] > 0, flip_array[a, s, :] == 0))
            nonflipped_gain_probs[a, s] = _sum / n_b

            gain_probs[a, s] = np.sum(accu_array[a, s, :] > 0) / n_b
    return gain_probs, flipped_gain_probs, nonflipped_gain_probs


def simulate_and_save(n, a_entropy, b_entropy, root):
    input_data = read_npy_dict(
        os.path.join(root, "constructed_arrays", f"{n}.npy"))

    A_array = input_data["A"][a_entropy]
    b_array = input_data["b"][b_entropy]
    prior = input_data["prior"]

    data = {}
    data[f"{a_entropy}_{b_entropy}"] = simulate(n, A_array, b_array, prior)
    return data


def simulate_unpack(args):
    return simulate_and_save(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def main(root, step, generate_all=True):
    root = os.path.join(root, "simulation_results")
    print("=" * 80)
    print("Simulation experiments")
    print("=" * 80)
    if step == 1:
        print("Step 1: generate A, b, priors first")
        data_root = os.path.join(root, "constructed_arrays")
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        n_list = [i for i in range(2, 11)]
        n_list += [i * 10 for i in range(2, 11)]
        print(f"\tspawn {len(n_list)} processes")
        pool = multiprocessing.Pool(processes=len(n_list))
        pool.map(functools.partial(generate_n, data_root=data_root), n_list)

    elif step == 2:
        n_list = [i for i in range(2, 11)]
        n_list += [i * 10 for i in range(2, 11)]

        print(
            f"Step 2: construct the expected accuracy gain with n = {n_list}")
        for n in tqdm(reversed(n_list), total=len(n_list)):
            output_data = {}
            outpath = os.path.join(root, f"{n}x{n}.npy")
            A_entropy_levels = [0.75, 0.5, 0.25, 0.0]
            b_entropy_levels = [0.75, 0.5, 0.25, 0.0]
            # spawn below processes
            args_list = []
            for b_entropy in b_entropy_levels:
                for a_entropy in A_entropy_levels:
                    args_list.append((n, a_entropy, b_entropy, root))

            print(f"\tgoing to spawn {len(args_list)} processes n = {n}")
            with poolcontext(processes=len(args_list)) as pool:
                data_list = pool.map(simulate_unpack, iter(args_list))

            for d in data_list:
                output_data.update(d)

            np.save(outpath, output_data)
            print(f"Saved the computed expected accuracy gain at {outpath}")
