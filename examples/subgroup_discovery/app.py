import streamlit as st
from er_evaluation.summary import cluster_sizes
from er_evaluation.error_analysis import error_indicator, expected_relative_extra, expected_relative_missing
from er_evaluation.error_analysis import fit_dt_regressor
from er_evaluation.plots import plot_dt_regressor_sunburst
from er_evaluation.estimators._utils import _parse_weights

from data_prep import features_df, numerical_features, categorical_features, pred, reference

def make_plot(prediction, reference, weights, error_function, features_df, numerical_features, categorical_features, **kwargs):
    y = error_function(prediction, reference)
    weights = _parse_weights(reference, weights)

    model = fit_dt_regressor(
        features_df, 
        y, 
        numerical_features, 
        categorical_features, 
        sample_weights=weights,
        **kwargs)


    dt_regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [x.split("__")[1] for x in preprocessor.get_feature_names_out()]
    X = preprocessor.transform(features_df)

    return plot_dt_regressor_sunburst(dt_regressor, X, y, feature_names, label="Error Rate")

fig = make_plot(pred, reference, "cluster_size", features_df, error_indicator, numerical_features, categorical_features, max_depth=3)

st.plotly_chart(fig)