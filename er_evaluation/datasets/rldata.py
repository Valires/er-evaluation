from collections import namedtuple
from er_evaluation.utils import load_module_tsv

DATA_MODULE = "er_evaluation.datasets.raw_data"
RLDATA_MODULE = DATA_MODULE + ".rldata"

RLPredictions = namedtuple("RLPredictions", ["name", "name_by", "name_bm", "name_bd"])


def _make_rldisambiguations(rldata):
    """
    Create four predicted disambiguations for RLdata.

    These are toy disambiguations meant to showcase features of the ER-Evaluation package. None of them are very accurate.

    The four disambiguations are:

    * **name**: Disambiguation based on exact matching first name and last name.
    * **name_by**: Disambiguation based on exact matching first name, last name, and birth year.
    * **name_bm**: Disambiguation based on exact matching first name, last name, and birth month.
    * **name_bd**: Disambiguation based on exact matching first name, last name, and birth day.

    Args:
        rldata (DataFrame): RLdata500 or RLdata10000 dataframe (see :meth:`er_evaluation.datasets.load_rldata500`).

    Returns:
        RLPredictions: Named tuple with the four disambiguations.
    """
    disamb_name = rldata["fname_c1"] + " " + rldata["lname_c1"]
    disamb_name_by = disamb_name + " " + rldata["by"]
    disamb_name_bm = disamb_name + " " + rldata["bm"]
    disamb_name_bd = disamb_name + " " + rldata["bd"]

    return RLPredictions(disamb_name, disamb_name_by, disamb_name_bm, disamb_name_bd)


def load_rldata500():
    """
    Load RLdata500 dataset.

    Dataset with 500 rows, including 50 noisy duplicate records, from the `RecordLinkage <https://cran.r-project.org/web/packages/RecordLinkage/RecordLinkage.pdf>`_ R package.

    Unique identifiers for each row can be obtained from :meth:`er_evaluation.datasets.load_rldata500_disambiguations`.

    Columns are:

    * **fname_c1**: First name, first component.
    * **fname_c2**: First name, second component.
    * **lname_c1**: Last name, first component.
    * **lname_c2**: Last name, second component.
    * **by**: Year of birth.
    * **bm**: Month of birth.
    * **bd**: Day of birth.

    Returns:
        DataFrame: RLdata500 dataset.
    """
    rldata500 = load_module_tsv(RLDATA_MODULE, "RLdata500.tsv", dtype=str)

    return rldata500


def load_rldata500_disambiguations():
    """
    Load reference and predicted disambiguations for the RLdata500 dataset.

    The reference disambiguation is the series of true unique identifiers for RLdata500. 
    
    Predicted disambiguations are a set of four toy disambiguations meant to showcase and test features of this package. The four predicted disambiguations are:

    * **name**: Disambiguation based on exact matching first name and last name.
    * **name_by**: Disambiguation based on exact matching first name, last name, and birth year.
    * **name_bm**: Disambiguation based on exact matching first name, last name, and birth month.
    * **name_bd**: Disambiguation based on exact matching first name, last name, and birth day.

    These are returned in a named tuple called "RLPredictions" with the above named elements.

    Returns:
        tuple: tuple of the form ``(reference, predictions)``, where ``reference`` is the ground truth disambiguation and ``predictions`` is a named tuple with four toy disambiguations.
    
    Examples:

        Load ground truth and the set of four toy predictions:

        >>> reference, predictions = load_rldata500_disambiguations()

        Compute pairwise precision for each prediction:

        >>> from er_evaluation.metrics import pairwise_precision
        >>> pairwise_precision(predictions.name, reference)
        0.4523809523809524

        >>> pairwise_precision(predictions.name_by, reference)
        1.0

        >>> pairwise_precision(predictions.name_bm, reference)
        0.7619047619047619

        >>> pairwise_precision(predictions.name_bd, reference)
        1.0
    """
    rldata500 = load_rldata500()
    reference = load_module_tsv(RLDATA_MODULE, "identity.RLdata500.tsv").iloc[:,0]

    disambiguations = _make_rldisambiguations(rldata500)

    return reference, disambiguations


def load_rldata10000():
    """
    Load RLdata10000 dataset.

    Dataset with 10000 rows, including 1000 noisy duplicate records, from the `RecordLinkage <https://cran.r-project.org/web/packages/RecordLinkage/RecordLinkage.pdf>`_ R package.

    Unique identifiers for each row can be obtained from :meth:`er_evaluation.datasets.load_rldata500_disambiguations`.

    Columns are:

    * **fname_c1**: First name, first component.
    * **fname_c2**: First name, second component.
    * **lname_c1**: Last name, first component.
    * **lname_c2**: Last name, second component.
    * **by**: Year of birth.
    * **bm**: Month of birth.
    * **bd**: Day of birth.

    Returns:
        DataFrame: RLdata500 dataset.
    """
    rldata10000 = load_module_tsv(RLDATA_MODULE, "RLdata10000.tsv", dtype=str)

    return rldata10000


def load_rldata10000_disambiguations():
    """
    Load reference and predicted disambiguations for the RLdata10000 dataset.

    The reference disambiguation is the series of true unique identifiers for RLdata10000. 
    
    Predicted disambiguations are a set of four toy disambiguations meant to showcase and test features of this package. The four predicted disambiguations are:

    * **name**: Disambiguation based on exact matching first name and last name.
    * **name_by**: Disambiguation based on exact matching first name, last name, and birth year.
    * **name_bm**: Disambiguation based on exact matching first name, last name, and birth month.
    * **name_bd**: Disambiguation based on exact matching first name, last name, and birth day.

    These are returned in a named tuple called "RLPredictions" with the above named elements.

    Returns:
        tuple: tuple of the form ``(reference, predictions)``, where ``reference`` is the ground truth disambiguation and ``predictions`` is a named tuple with four toy disambiguations.
    
    Examples:

        Load ground truth and the set of four toy predictions:

        >>> reference, predictions = load_rldata10000_disambiguations()

        Compute pairwise precision for each prediction:

        >>> from er_evaluation.metrics import pairwise_precision
        >>> pairwise_precision(predictions.name, reference)
        0.04653923780125846

        >>> pairwise_precision(predictions.name_by, reference)
        0.7028571428571428

        >>> pairwise_precision(predictions.name_bm, reference)
        0.3076086956521739

        >>> pairwise_precision(predictions.name_bd, reference)
        0.501937984496124
    """
    rldata10000 = load_rldata10000()
    reference = load_module_tsv(RLDATA_MODULE, "identity.RLdata10000.tsv").iloc[:,0]

    disambiguations = _make_rldisambiguations(rldata10000)

    return reference, disambiguations
