#!/usr/bin/env python3
"""
RSA model
"""
import numpy as np


class RSA:
    """Implementation of the core Rational Speech Acts model.
    Support both speaker-base and listerner-base model
    Parameters
    ----------
    lexicon : `np.array`. (num_classes x num_images)
        motives along the rows, states along the columns.
    prior : array-like
        Same length as the number of rows in `lexicon`.
        if None: will create a uniform distribution over motive
    costs : array-like
        Same length as the number of columns in `lexicon`.
        if None, will create a list of 0s
    alpha : float
        The temperature parameter. Default: 1.0
    use_lexicon_as_base [deprecated]: bool, if to use the lexicon as a base agent
    normalize_use_base [deprecated]: bool, if true, use base agent to normalize
        each reaser and creator
    """
    def __init__(
        self, num_labels, num_images, prior=None, costs=None,
        alpha=1.0, use_lexicon_as_base=True, normalize_use_base=False,
    ):
        # num_labels, num_images = self.lexicon.shape

        if prior is None:
            self.prior = np.array([1 / num_labels] * num_labels)
        else:
            self.prior = prior
        if costs is None:
            self.costs = np.array([0.0] * num_images)
        else:
            self.costs = np.array(costs)
        self.alpha = alpha
        self.use_lexicon_as_base = use_lexicon_as_base
        self.normalize_use_base = normalize_use_base

    def _viewer(self, x):
        """
        Equivalent to listener in traditional RSA game setting.
        normalized across motives, column normalization
        Inputs:
            x: an array (num_classes x num_images)
        Returns:
            x: an array (num_classes x num_images)
        """
        return rownorm(x.T * self.prior).T

    def _creator(self, x):
        """
        Equivalent to speaker in traditional RSA game setting.
        normalized across images, row normalization
        Inputs:
            x: an array (num_classes x num_images)
        Returns:
            x: (num_classes x num_images)
        """
        utilities = self.alpha * (safelog(x) + self.costs)
        # across motives
        return rownorm(np.exp(utilities))

    def literal_viewer(self, lexicon):
        """Literal viewer predictions, which corresponds intuitively
        to truth conditions with priors.
        """
        if self.use_lexicon_as_base:
            return lexicon
        return self._viewer(self.lexicon)

    def literal_creator(self, lexicon):
        if self.use_lexicon_as_base:
            return lexicon
        return self._creator(self.lexicon)

    def _return_results(self, all_results, return_all, literal_creator=True):
        """return the last instance to save space"""
        if not return_all:
            return all_results[-1][:, -1][:, np.newaxis]

        out = {}
        if literal_creator:
            key = "C"
        else:
            key = "V"
        for i, result_array in enumerate(all_results):
            out[f"{key}{i}"] = result_array[:, -1][:, np.newaxis]
        return out

    def viewer(self, lexicon, level, return_all=True):
        """Returns a matrix of pragmatic viewer predictions.
        if return all: return r0, c1, r2, c3, ...
        Returns
        -------
        np.array: (num_classes x num_images)
        """
        all_r = []

        v0 = self.literal_viewer(lexicon)
        all_r.append(v0)

        for _ in range(level):
            v = self._viewer(self._creator(v0))
            v0 = v
            all_r.append(v)
        return self._return_results(all_r, return_all, False)

    def creator(self, lexicon, level, return_all=True):
        """Returns a matrix of pragmatic creator predictions.

        Returns
        -------
        np.array: (num_classes x num_images)
        """
        all_r = []

        c0 = self.literal_creator(lexicon)
        all_r.append(c0)

        for _ in range(level):
            r = self._viewer(c0)
            c = self._creator(r)

            c0 = c
            all_r.append(r)
            all_r.append(c)
        return self._return_results(all_r, return_all, True)


def rownorm(mat):
    """Row normalization of np.array"""
    return (mat.T / (mat.sum(axis=1) + 0.0000000001)).T


def safelog(vals):
    """Silence distracting warnings about log(0)."""
    with np.errstate(divide='ignore'):
        return np.log(vals)
