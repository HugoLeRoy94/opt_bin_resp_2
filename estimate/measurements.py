"""Evaluation-time measurements. Name -> dict of recorded numbers.

These are what a test script records into test_<label>.json. They are
separate from the training
objective: the loss optimizes one bound, these record several, so a run can be
judged by an estimator it was not trained on.

Every entry takes (activity,) or (activity, loss_fn) and returns a dict. The
runner dispatches on that signature, so adding a measurement means adding a
function and a REGISTRY line.

Acronyms: MI = mutual information. KT = Kolchinsky-Tracey. MM = Miller-Madow
(a bias correction for plug-in entropy, +(K-1)/(2 B ln2) with K distinct
observed symbols and B samples).
"""
import math

import torch

from estimate.counting import entropy_from_counts
from estimate.entropy import (KT_EPS, compute_kt_entropy, compute_kt_upper_entropy,
                              compute_response_conditional_entropy)
from estimate.grouping import GroupedResponseCounter
from estimate.losses import (GroupedCellMutualInformationLoss,
                             GroupedKTMutualInformationLoss)


@torch.no_grad()
def _measure_entropy(loss_fn, act, entropy_type):
    """One entropy estimator on the soft activity, in bits.

    entropy_type=None means the loss's native estimator.
    """
    if entropy_type == 'kt':
        if isinstance(loss_fn, GroupedKTMutualInformationLoss):
            return loss_fn.bound(act).item()
        return compute_kt_entropy(loss_fn.compute_soft_assignment(act)).item()
    if entropy_type == 'kt_upper':
        if isinstance(loss_fn, GroupedKTMutualInformationLoss):
            return loss_fn.bound(act, upper=True).item()
        return compute_kt_upper_entropy(loss_fn.compute_soft_assignment(act)).item()
    kw = {} if entropy_type is None else {'entropy_type': entropy_type}
    return loss_fn.compute_entropy(act, **kw).item()


def full_array_entropy(activity, loss_fn):
    """H(Y) under the estimator the loss itself trains on."""
    return {'full_array_entropy': _measure_entropy(loss_fn, activity.detach(), None)}


def entropy_kt(activity, loss_fn):
    """Opt-in: KT lower bound on H(Y)."""
    return {'full_array_entropy_kt': _measure_entropy(loss_fn, activity.detach(), 'kt')}


def entropy_kt_upper(activity, loss_fn):
    """Opt-in: KT upper bound on H(Y). Pairs with entropy_kt to bracket it."""
    return {'full_array_entropy_kt_upper':
            _measure_entropy(loss_fn, activity.detach(), 'kt_upper')}


def conditional_entropy_response(activity):
    """H(Y|X) from the Bernoulli outputs; X includes the sampled observation noise."""
    return compute_response_conditional_entropy(activity).item()


def mutual_information_kt(activity, loss_fn):
    """KT lower bound on I(Y;X) for the complete sampled input."""
    if isinstance(loss_fn, GroupedKTMutualInformationLoss):
        return loss_fn.bound(activity, return_mi=True).item()
    return compute_kt_entropy(loss_fn.compute_soft_assignment(activity),
                              return_mi=True).item()


def mutual_information_kt_upper(activity, loss_fn):
    """KT upper bound on I(Y;X)."""
    if isinstance(loss_fn, GroupedKTMutualInformationLoss):
        return loss_fn.bound(activity, upper=True, return_mi=True).item()
    return compute_kt_upper_entropy(loss_fn.compute_soft_assignment(activity),
                                    return_mi=True).item()


def miller_madow_entropy(activity):
    """Plug-in and MM entropy of the HARD codewords (activity thresholded at 0.5)."""
    B, R = activity.shape
    codes = (activity > 0.5).long()
    _, counts = torch.unique(codes, dim=0, return_counts=True)
    H_plugin, H_MM, K_hat, _, missing = entropy_from_counts(counts)
    return H_plugin, H_MM, K_hat, math.log2(B), math.ldexp(float(K_hat), -R), missing


def _counting_metrics(plugin, mm, k_hat, log2_b, missing, n_samples, h_cond):
    return {
        'response_entropy_plugin': plugin,
        'response_entropy_mm': mm,
        'mutual_information_counting_plugin': plugin - h_cond,
        'mutual_information_counting_mm': mm - h_cond,
        'conditional_entropy_response': h_cond,
        'response_counting_K_hat': float(k_hat),
        'response_counting_unique_fraction': k_hat / n_samples,
        'response_counting_missing_mass': missing,
        'response_counting_samples': int(n_samples),
        'response_counting_log2B': log2_b,
    }


def response_counting_metrics(codes, h_cond):
    """Sampled-output H minus analytic H(Y|X), with no population-bound claim.

    Negative estimates are kept. Finite-sample bias and Monte Carlo error must
    not be hidden by clamping, and K_hat/B is what diagnoses sparse coverage.
    """
    plugin, mm, k_hat, log2_b, _, missing = miller_madow_entropy(codes)
    return _counting_metrics(plugin, mm, k_hat, log2_b, missing, codes.shape[0], h_cond)


def response_counting_metrics_from_counts(counts, n_samples, h_cond):
    """Same metrics from a merged frequency table, without holding the code matrix."""
    plugin, mm, k_hat, log2_b, missing = entropy_from_counts(counts)
    return _counting_metrics(plugin, mm, k_hat, log2_b, missing, n_samples, h_cond)


@torch.no_grad()
def mutual_information_counting(activity):
    """Draw one Bernoulli output per sniff, count codewords, subtract H(Y|X)."""
    a = activity.clamp(KT_EPS, 1.0 - KT_EPS)
    return response_counting_metrics(torch.bernoulli(a).bool(),
                                     compute_response_conditional_entropy(a).item())


@torch.no_grad()
def codeword_entropy(activity):
    """Hard-codeword plug-in and MM entropy; activity binarised at 0.5.

    K_frac = K_hat / 2^R is the fraction of the alphabet ever observed, and
    missing_mass is the Good-Turing estimate of the probability never sampled.
    Both say whether the plug-in number is trustworthy.
    """
    H_plugin, H_MM, K_hat, log2_B, K_frac, missing = miller_madow_entropy(activity.detach())
    return {
        'codeword_entropy_plugin': H_plugin,
        'codeword_entropy_mm':     H_MM,
        'codeword_entropy_log2B':  log2_B,
        'codeword_entropy_K_hat':  float(K_hat),
        'codeword_entropy_K_frac': K_frac,
        'codeword_entropy_missing_mass': missing,
    }


@torch.no_grad()
def grouped_information(activity, loss_fn):
    """Exact count entropies and cell MI, by enumerating joint count states."""
    return {k: v.item() for k, v in loss_fn.compute_metrics(activity).items()}


@torch.no_grad()
def grouped_counting(activity, loss_fn):
    """Sampled integer group counts, no joint-state enumeration."""
    counter = GroupedResponseCounter(loss_fn.grouping)
    counter.update(activity)
    return counter.metrics()


REGISTRY = {
    "full_array_entropy":           full_array_entropy,
    "entropy_kt":                   entropy_kt,
    "entropy_kt_upper":             entropy_kt_upper,
    "codeword_entropy":             codeword_entropy,
    "conditional_entropy_response": conditional_entropy_response,
    "mutual_information_kt":        mutual_information_kt,
    "mutual_information_kt_upper":  mutual_information_kt_upper,
    "mutual_information_counting":  mutual_information_counting,
    "grouped_information":          grouped_information,
    "grouped_counting":             grouped_counting,
}
