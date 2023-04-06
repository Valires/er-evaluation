import ethnicolr
import pandas as pd
from statistics import mode
import numpy as np

from er_evaluation.datasets import load_pv_data, load_pv_disambiguations

predictions, reference = load_pv_disambiguations()
pred = predictions[pd.Timestamp('2021-12-30 00:00:00')]

def flatten_mode(x):
    return mode(np.concatenate(x.apply(lambda x: np.unique(x)).values))

pv_data = load_pv_data()
race_df = ethnicolr.pred_census_ln(pv_data, 'raw_inventor_name_first', 2010)
features_df = (
    pv_data
    .merge(pv_data["block"].value_counts().rename("block_size"), left_on="block", right_index=True)
    .assign(num_coauthors=pv_data["coinventor_sequence"].apply(len))
    .assign(year_first=pv_data["filing_date"].apply(lambda x: float(str(x).split("-")[0]) if isinstance(x, str) else np.nan))
    .assign(year_last=pv_data["filing_date"].apply(lambda x: float(str(x).split("-")[0]) if isinstance(x, str) else np.nan))
    .assign(ethnicity=np.where(race_df["race"] == "api", "API", "Other"))
    .merge(reference.rename("reference"), left_on="mention_id", right_index=True)
    .groupby("reference")
    .agg({
        "raw_inventor_name_first": mode,
        "raw_inventor_name_last": mode,
        "patent_id": "count",
        "raw_country": mode,
        "patent_type": mode,
        "num_coauthors": "mean",
        "block_size": "mean",
        "cpc_section": flatten_mode,
        "year_first": min,
        "year_last": max,
        "ethnicity": mode,
    })
    .rename(columns={
        "raw_inventor_name_first": "name_first",
        "raw_inventor_name_last": "name_last",
        "patent_id": "prolificness",
        "raw_country": "country",
        "num_coauthors": "avg_coauthors",
    })
)

numerical_features = [
    "prolificness",
    "avg_coauthors",
    "block_size",
    "year_first",
    "year_last",
]
categorical_features = [
    "country",
    "patent_type",
    "cpc_section",
    "ethnicity"
]