from er_evaluation.utils import load_module_parquet

DATA_MODULE = "er_evaluation.datasets.raw_data"
PV_DATA_MODULE = DATA_MODULE + ".patentsview"


def load_pv_data():
    """
    Load PatentsView dataset.

    This dataset contains inventor mentions for all blocks which intersect Binette's 2022 inventors benchmark. Features such as inventor mention name, location, patent title, abstract, filing date, assignees, attorneys, CPC codes, and co-inventors have been added.

    Returns:
        pandas DataFrame
    """
    return load_module_parquet(PV_DATA_MODULE, "pv-data.parquet")


def load_pv_disambiguations():
    """
    Load reference disambiguation and predicted disambiguations for the PatentsView dataset.

    See :meth:`er_evaluation.datasets.load_pv_data` for more information on the PatentsView dataset.

    The reference disambiguation corresponds to Binette's 2022 inventors benchmark. It does not cover the entirety of the PatentsView dataset. It is a sample of 400 inventor clusters with sampling probabilities proportional to cluster size.

    Predicted disambiguations correspond to inventor disambiguations released by PatentsView between 2017 and 2022. The predicted disambiguations have been restricted to blocks which intersect Binette's 2022 inventors benchmark.

    Returns:
        tuple ``(reference, predictions)`` where ``reference`` is the ground truth disambiguation and ``predictions`` is a dictionary of predicted disambiguations.

    Examples:

        Estimate pairwise precision for PatentsView's 2021/12/30 disambiguation:

        >>> reference, predictions = load_pv_disambiguations()
        >>> from er_evaluation.estimators import pairwise_precision_design_estimate
        >>> pairwise_precision_design_estimate(predictions["disamb_inventor_id_20211230"], reference, weights="cluster_size")
        (0.9138044762074499, 0.018549986866583837)
    """
    return _load_pv_reference(), _load_pv_predictions()


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

    return {col: predictions_table[col] for col in cols}
