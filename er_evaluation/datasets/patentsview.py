import pandas as pd

from er_evaluation.utils import load_module_parquet

DATA_MODULE = "er_evaluation.datasets.raw_data"
PV_DATA_MODULE = DATA_MODULE + ".patentsview"


def load_pv_data():
    """
    Load PatentsView dataset.

    This is based on a subset of the "g_inventor_not_disambiguated.tsv" file from PatentsView's `bulk data downloads <https://patentsview.org/download/data-download-tables>`_. The dataset has been subsetted to only contain inventor mentions for blocks which intersect Binette's 2022 inventors benchmark [1]. Following PatentsView's disambiguation methodology, a block is defined by an inventor mention's full last name and the first two letters of its first name. Therefore, this dataset contains all inventor mentions for which the last name and first two letters of the first name match those found in Binette's 2022 inventors benchmark.

    A number of features have been added, such as inventor mention name, location, patent title, abstract, filing date, assignees, attorneys, CPC codes, and co-inventors list. The code used to produce this dataset is located in "er_evaluation/datasets/raw_data/patentsview/reproduce.ipynb".

    Refer to :meth:`er_evaluation.datasets.load_pv_disambiguations` in order to access Binette's 2022 inventors benchmark and PatentsView's predicted disambiguations.

    Returns:
        pandas DataFrame

    References:
        1. Binette, Olivier, Sarvo Madhavan, Jack Butler, Beth Anne Card, Emily Melluso and Christina Jones. 2023. **PatentsView-Evaluation: Evaluation Datasets and Tools to Advance Research on Inventor Name Disambiguation**. arXiv e-prints: arxiv:2301.03591. Available online at https://arxiv.org/abs/2301.03591
        2. Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). **Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org**. arXiv e-prints: arxiv:2210.01230
    """
    return load_module_parquet(PV_DATA_MODULE, "pv-data.parquet")


def load_pv_disambiguations():
    """
    Load reference disambiguation and predicted disambiguations for the PatentsView dataset.

    See :meth:`er_evaluation.datasets.load_pv_data` for more information on the PatentsView dataset.

    The reference disambiguation corresponds to Binette's 2022 inventors benchmark. It does not cover the entirety of the PatentsView dataset. It is a sample of 400 inventor clusters with sampling probabilities proportional to cluster size.

    Predicted disambiguations correspond to inventor disambiguations released by PatentsView between 2017 and 2022. The data has been restricted to inventor mentions for which the last name and first two letters of the first name match those found in Binette's 2022 inventors benchmark.

    Returns:
        tuple ``(predictions, reference)`` where ``reference`` is the ground truth disambiguation and ``predictions`` is a dictionary of predicted disambiguations.

    Examples:

        Estimate pairwise precision for PatentsView's 2021/12/30 disambiguation:

        >>> predictions, reference = load_pv_disambiguations()
        >>> from er_evaluation.estimators import pairwise_precision_design_estimate
        >>> import pandas as pd
        >>> prediction = predictions[pd.Timestamp('2021-12-30 00:00:00')]
        >>> pairwise_precision_design_estimate(prediction, reference, weights="cluster_size")
        (0.9131787709880134, 0.018619907220335144)

    References:
        1. Binette, Olivier, Sarvo Madhavan, Jack Butler, Beth Anne Card, Emily Melluso and Christina Jones. 2023. **PatentsView-Evaluation: Evaluation Datasets and Tools to Advance Research on Inventor Name Disambiguation**. arXiv e-prints: arxiv:2301.03591. Available online at https://arxiv.org/abs/2301.03591
        2. Binette, Olivier, Sokhna A York, Emma Hickerson, Youngsoo Baek, Sarvo Madhavan, Christina Jones. (2022). **Estimating the Performance of Entity Resolution Algorithms: Lessons Learned Through PatentsView.org**. arXiv e-prints: arxiv:2210.01230
    """
    return _load_pv_predictions(), _load_pv_reference()


def _load_pv_reference():
    return load_module_parquet(PV_DATA_MODULE, "pv-reference.parquet").set_index("mention_id")["unique_id"]


def _load_pv_predictions():
    predictions_table = load_module_parquet(PV_DATA_MODULE, "pv-predictions.parquet").set_index("mention_id")

    cols = [
        "disamb_inventor_id_20170808",
        "disamb_inventor_id_20171003",
        "disamb_inventor_id_20171226",
        "disamb_inventor_id_20180528",
        "disamb_inventor_id_20181127",
        "disamb_inventor_id_20190312",
        "disamb_inventor_id_20190820",
        "disamb_inventor_id_20191008",
        "disamb_inventor_id_20191231",
        "disamb_inventor_id_20200331",
        "disamb_inventor_id_20200630",
        "disamb_inventor_id_20200929",
        "disamb_inventor_id_20201229",
        "disamb_inventor_id_20211230",
        "disamb_inventor_id_20220630",
    ]

    return {pd.to_datetime(col.lstrip("disamb_inventor_id_")): predictions_table[col] for col in cols}
