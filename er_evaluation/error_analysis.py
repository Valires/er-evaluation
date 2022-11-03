import pandas as pd
import numpy as np
from scipy.special import comb


def proportion_extra_links(prediction, sample):
    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        # Number of elements within sampled cluster split across predicted clusters:
        p = pd.value_counts(sample_cluster)
        # Number of elements within predicted clusters (restricted to current sampled cluster):
        u = outer.prediction.value_counts()[p.index].values

        n_links = np.sum(p * (u - p)) + np.sum(comb(u, 2))

        if n_links == 0:
            return 0
        else:
            return np.sum(p * (u - p)) / n_links

    result = outer.groupby("sample").apply(lambd)
    result.rename("proportion_extra_links", inplace=True)

    return result


def proportion_missing_links(prediction, sample):
    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        p = pd.value_counts(sample_cluster)
        n = np.sum(p)

        n_links = comb(n, 2)
        if n_links == 0:
            return 0
        else:
            return np.sum(p * (n - p)) / (2 * n_links)

    result = outer.groupby("sample").apply(lambd)
    result.rename("proportion_extra_links", inplace=True)

    return result


def splitting_entropy(prediction, sample, alpha=1):
    # Index of elements with a predicted cluster intersecting sampled clusters:
    I = prediction.isin(prediction[prediction.index.isin(sample.index)])
    relevant_predictions = prediction[I]

    outer = pd.concat(
        {"prediction": relevant_predictions, "sample": sample},
        axis=1,
        copy=False,
        join="outer",
    )

    def lambd(sample_cluster):
        u = pd.value_counts(sample_cluster, normalize=True).values

        if len(u) <= 1:
            return 1
        if alpha == 0:
            return len(u)
        if alpha == 1:
            return np.exp(-np.sum(u * np.log(u)))
        else:
            return (np.sum(u**alpha)) ** (1 / (1 - alpha))

    result = outer.groupby("sample").apply(lambd)
    result.rename("proportion_extra_links", inplace=True)

    return result
