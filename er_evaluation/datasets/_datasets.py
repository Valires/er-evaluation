from importlib import resources

import pandas as pd

DATA_MODULE = "er_evaluation.datasets.raw_data"


def load_tsv(module, filename, dtype=str):
    with resources.open_text(module, filename) as f:
        data = pd.read_csv(f, sep="\t", dtype=dtype)

    return data


def _load_rldata(filename, sample_prop=0.2, sample_type="uniform", random_state=1):
    module = DATA_MODULE + ".rldata"

    rldata = load_tsv(module, filename, dtype=str)
    prediction = rldata["fname_c1"].astype(str) + " " + rldata["lname_c1"].astype(str) + rldata["by"].astype(str)
    reference = load_tsv(module, "identity." + filename).iloc[:, 0]

    if sample_type == "uniform":
        ref_ids = reference.drop_duplicates()
    elif sample_type == "cluster_size":
        ref_ids = reference
    else:
        raise ValueError("Invalid sample_type value. Should be one of 'uniform' or 'cluster_size'.")
    sampled_cluster_ids = ref_ids.sample(frac=sample_prop, random_state=random_state)
    sample = reference[reference.isin(sampled_cluster_ids)]

    return rldata, prediction, reference, sample


def load_rldata500(data_only=False, memberships_only=False, sample_prop=0.2, random_state=1):
    data, prediction, reference, sample = _load_rldata(
        "RLdata500.tsv", sample_prop=sample_prop, random_state=random_state
    )

    if data_only:
        return data
    if memberships_only:
        return prediction, reference, sample
    else:
        return data, prediction, reference, sample


def load_rldata10000(data_only=False, memberships_only=False, sample_prop=0.2, random_state=1):
    data, prediction, reference, sample = _load_rldata(
        "RLdata10000.tsv", sample_prop=sample_prop, random_state=random_state
    )

    if data_only:
        return data
    if memberships_only:
        return prediction, reference, sample
    else:
        return data, prediction, reference, sample


# TODO: Add PatentsView inventor mentions dataset for all blocks that intersects Binette's 2022 inventors benchmark
