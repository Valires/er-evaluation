__version__ = "2.2.1"

import er_evaluation.data_structures
import er_evaluation.datasets
import er_evaluation.error_analysis
import er_evaluation.estimators
import er_evaluation.metrics
import er_evaluation.plots
import er_evaluation.summary
import er_evaluation.utils
from er_evaluation.data_structures import *
from er_evaluation.datasets import *
from er_evaluation.error_analysis import *
from er_evaluation.estimators import *
from er_evaluation.metrics import *
from er_evaluation.plots import *
from er_evaluation.summary import *
from er_evaluation.utils import *

__all__ = (
    er_evaluation.data_structures.__all__
    + er_evaluation.datasets.__all__
    + er_evaluation.error_analysis.__all__
    + er_evaluation.estimators.__all__
    + er_evaluation.metrics.__all__
    + er_evaluation.plots.__all__
    + er_evaluation.utils.__all__
    + er_evaluation.summary.__all__
    + ["__version__"]
)
