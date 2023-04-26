import streamlit as st
import numpy as np
import plotly.graph_objects as go

from er_evaluation.error_analysis import error_indicator, expected_relative_extra, expected_relative_missing
from er_evaluation.plots import make_dt_regressor_plot
from er_evaluation.estimators._utils import _parse_weights

from data_prep import features_df, numerical_features, categorical_features, pred, reference

targets = {
    "Error Indicator": error_indicator,
    "E_relative_extra": expected_relative_extra,
    "E_relative_missing": expected_relative_missing,
}

with st.sidebar:
    target_key = st.selectbox("Target Error Metric:", list(targets.keys()), index=0)
    threshold = st.number_input("Threshold:", value=np.nan)

    with st.expander("Tree Options:", expanded=True):
        criterion = st.selectbox("Criterion:", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0)
        use_weights = st.checkbox("Use Weights", value=True)

        num_select = st.multiselect("Numerical Features:", numerical_features, default=numerical_features)
        cat_select = st.multiselect("Categorical Features:", categorical_features, default=categorical_features)
        max_depth = st.slider("Max Depth:", 1, 10, 3)
        min_samples_leaf = st.number_input("Min Samples Leaf:", value=30, min_value=1)

        cmin = st.number_input("Color Scale Min:", value=np.nan)
        cmax = st.number_input("Color Scale Max:", value=np.nan)

"""
# Error Tree Visualization

Identify subgroups of inventors with higher or lower error rates, given a target error metric and a set of relevant features. We use a decision tree regressor with a squared error criterion to fit the data and present the tree using an interactive sunburst/treemap/tree chart visualization, with the following node attributes:

- **color**: the average target value ("Avg. Error") of the subgroup,
- **arc angle/size**: the number of inventors in the subgroup.
"""

y = targets[target_key](pred, reference)
if not np.isnan(threshold):
    y = y > threshold
if use_weights:
    weights = "cluster_size"
else:
    weights = "uniform"
weights = _parse_weights(reference, weights)
weights = len(y) * weights / weights.sum()

if len(num_select) + len(cat_select) == 0:
    st.write("Select at least one feature")
    st.stop()
else:
    tabs = st.tabs(["Sunburst", "Treemap", "Tree"])
    tab_plots = ["sunburst", "treemap", "tree"]
    for i, tab in enumerate(tabs):
        with tab:
            fig = make_dt_regressor_plot(
                y,
                weights,
                features_df,
                num_select,
                cat_select,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                type=tab_plots[i],
                criterion=criterion,
            )
            fig.update_traces(marker=dict(cmin=cmin, cmax=cmax))
            st.plotly_chart(fig, theme="streamlit")

with st.expander("More information"):
    st.info(
        "The decision tree regressor utilizes a squared error criterion to evaluate the quality of each split. At each node, the model selects the feature and the threshold that minimize the sum of squared errors within the resulting subgroups. The process continues until a stopping criterion is met, such as reaching a predefined maximum depth or a minimum number of samples in a leaf node. Once the tree is fully grown, it models the target value for each subgroup by calculating the mean target value of the samples within the corresponding leaf nodes. The resulting decision tree is then visualized using an interactive sunburst/treemap/tree chart, enabling users to explore the hierarchy and gain insights into subgroups of inventors with higher or lower error rates based on the target error metric and relevant features."
    )

joined = y.reset_index().merge(features_df, left_on=y.index.name, right_on="reference")
joined.set_index("reference", inplace=True)
"""
## Marginal Relationships
"""

feature_names = categorical_features + numerical_features
tabs = st.tabs(feature_names)
for i, tab in enumerate(tabs):
    with tab:
        feature = feature_names[i]
        is_categorical = feature in categorical_features
        weighted_y = joined[y.name] * weights

        plot_container = st.empty()

        if is_categorical:
            weighted_grouped = joined.groupby(feature).agg({y.name: lambda x: (x * weights[x.index]).sum()})
            hist_normalized = weighted_grouped[y.name].values / weighted_grouped[y.name].sum()
            bin_centers = weighted_grouped.index.values
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("Number of bins:")
            with col2:
                nbins = st.slider("Number of bins:", 1, 100, 20, key=f"nbins_slider_{i}", label_visibility="collapsed")
            bin_edges = np.linspace(joined[feature].min(), joined[feature].max(), num=nbins)
            hist, _ = np.histogram(joined[feature], bins=bin_edges, weights=weighted_y)
            hist_weights, _ = np.histogram(joined[feature], bins=bin_edges, weights=weights)
            hist_normalized = hist / hist_weights
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = go.Figure(
            go.Bar(
                x=bin_centers,
                y=hist_normalized,
                width=np.diff(bin_edges) if not is_categorical else None,
                hovertemplate="%{y:.2f} at %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            title_text=f"{'Weighted ' if use_weights else ''}Average by {feature} bin",
            xaxis_title_text=feature,
            yaxis_title_text=f"{'Weighted ' if use_weights else ''}avg of {y.name}",
            bargap=0.2,
            bargroupgap=0.1,
        )

        plot_container.plotly_chart(fig)


"""
## Raw data
"""
st.write(joined)

st.download_button(
    "Download raw data",
    data=joined.to_csv(index=False),
    file_name="error_tree_data.csv",
    mime="text/csv",
    use_container_width=True,
)
