import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from er_evaluation.estimators import pairwise_f_estimator
from er_evaluation.utils import expand_grid


def plot_performance_disparities(
    prediction,
    reference,
    weights,
    protected_feature,
    estimator=pairwise_f_estimator,
    estimator_name="Pairwise F-score",
    max_subgroups=10,
):
    """
    Plot largest performance disparities among predefined subgroups.

    Args:
        prediction (Series): Predicted clustering.
        reference (Series): Reference clustering.
        weights (str or Series): Weights for sampled clusters, or one of "uniform" or "cluster_size".
        protected_feature (Series): Series index by reference cluster IDs and with values corresponding to group assignment.
        estimator: Function to use for performance estimation. Defaults to pairwise_f_design_estimate.
        estimator_name (str, optional): Name of the estimator to use in the plot labels. Defaults to "Pairwise F-score".
        max_subgroups (int, optional): Number of subgroups to display. Defaults to 10.

    Returns:
        plotly Figure
    """

    if protected_feature.name is None:
        protected_feature.name = "Protected attribute"

    scores_df = _make_scores_df(
        prediction, reference, weights, protected_feature.name, protected_feature, estimator_name, estimator
    )

    fig = _make_largest_difference_figure(
        scores_df,
        estimator_name,
        protected_feature=protected_feature.name,
        max_subgroups_per_control_cat_to_display=max_subgroups,
    )
    return fig


def _make_scores_df(prediction, reference, weights, protected_feature, protected_data, estimator_name, estimator):
    estimators = {estimator_name: estimator}
    table = expand_grid(**{protected_feature: protected_data.unique()}, _scorer=estimators)
    table.groupby(protected_feature)

    def apply_estimator(x):
        return estimators[x["_scorer"]](
            prediction,
            reference[reference.isin(protected_data[protected_data == x[protected_feature]].index)],
            weights=weights,
        )

    table[["_score", "_std"]] = table.apply(lambda x: apply_estimator(x), axis=1).to_list()
    table["_count"] = table.apply(
        lambda x: reference[reference.isin(protected_data[protected_data == x[protected_feature]].index)].nunique(),
        axis=1,
    )
    table[["_baseline", "_baseline_std"]] = table.apply(
        lambda x: estimators[x["_scorer"]](prediction, reference, weights=weights), axis=1
    ).to_list()
    table["_baseline_count"] = table.apply(lambda x: reference.nunique(), axis=1)
    table["_diff"] = table["_score"] - table["_baseline"]

    return table


def _add_differences_traces(
    sub_visual_df, fig, protected_feature, max_subgroups_per_control_cat_to_display, row=1, col=1
):
    """
    This function is adapted from Deepcheck's PerformanceBias module (https://deepchecks.com/) released under the AGPL-3 license.
    """
    sub_visual_df = sub_visual_df.sort_values("_diff").head(max_subgroups_per_control_cat_to_display)
    sub_visual_df = sub_visual_df.sort_values("_diff", ascending=False)
    for _, df_row in sub_visual_df.iterrows():
        subgroup = df_row[protected_feature]
        baseline = df_row["_baseline"]
        score = df_row["_score"]
        stds = [df_row["_std"], df_row["_baseline_std"]]
        color = "orangered" if df_row["_diff"] < 0 else "limegreen"
        legendgroup = "Negative differences" if df_row["_diff"] < 0 else "Positive differences"
        extra_label = "<extra></extra>"  # Hide extra label in hover

        fig.add_trace(
            go.Scatter(
                x=[score, baseline],
                y=[subgroup, subgroup],
                hovertemplate=[
                    "%{y}: %{x} (group size: " + str(df_row["_count"]) + ")" + extra_label,
                    "baseline: %{x} (group size: " + str(df_row["_baseline_count"]) + ")" + extra_label,
                ],
                marker=dict(
                    color=["white", "#222222"], symbol=0, size=6, line=dict(width=[2, 2], color=[color, color])
                ),
                legendgroup=legendgroup,
                line=dict(color=color, width=8),
                opacity=1,
                showlegend=False,
                mode="lines+text+markers",
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )
        # Error bars
        fig.add_trace(
            go.Scatter(
                x=[score, baseline],
                y=[subgroup, subgroup],
                hoverinfo="skip",
                error_x=dict(array=stds, color="black", thickness=0.75),
                marker=dict(color=["white", "#222222"], symbol=0, size=6),
                legendgroup=legendgroup,
                opacity=1,
                showlegend=False,
                mode="markers",
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )


def _add_legend(fig):
    """
    This function is adapted from Deepcheck's PerformanceBias module (https://deepchecks.com/) released under the AGPL-3 license.
    """
    for outline, title in [("orangered", "Negative differences"), ("limegreen", "Positive differences")]:
        for color, label in [("white", "subgroup score"), ("#222222", "baseline score")]:
            fig.add_traces(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=label,
                    legendgroup=title,
                    legendgrouptitle=dict(text=title),
                    marker=dict(color=color, symbol=0, size=6, line=dict(width=2, color=outline)),
                )
            )

    return fig


def _make_largest_difference_figure(
    scores_df: pd.DataFrame,
    scorer_name: str,
    protected_feature,
    control_feature=None,
    max_control_cat_to_display=3,
    max_subgroups_per_control_cat_to_display=3,
):
    """
    Create 'largest performance disparity' figure.

    This function is adapted from Deepcheck's PerformanceBias module (https://deepchecks.com/) released under the AGPL-3 license.

    Parameters
    ----------
    scores_df : DataFrame
        Dataframe of performance scores, as returned by `_make_scores_df()`, disaggregated by
        feature and control_feature, and with average scores for each control_feature level.
        Columns named after `feature` and (optionally) `control_feature` are expected, as
        well as columns named '_scorer', '_score', '_baseline', and '_count'.

    Returns
    -------
    Figure
        Figure showing subgroups with the largest performance disparities.
    """
    visual_df = scores_df.copy().dropna()
    if len(visual_df) == 0:
        return "No scores to display."

    has_control = control_feature is not None
    has_model_classes = "_class" in visual_df.columns.values

    subplot_grouping = []
    if has_control:
        subplot_grouping += [control_feature]
    if has_model_classes:
        subplot_grouping += ["_class"]
    # Get distinct subplot categories with the largest observed differences
    if len(subplot_grouping) > 0:
        subplots_categories = (
            visual_df.sort_values("_diff", ascending=True)[subplot_grouping]
            .drop_duplicates()
            .head(max_control_cat_to_display)
        )
        rows = len(subplots_categories)
    else:
        subplots_categories = None
        rows = 1

    subplot_titles = ""
    if has_control:
        subplot_titles += f"{control_feature}=" + subplots_categories[control_feature]
    if has_control and has_model_classes:
        subplot_titles += ", model_class=" + subplots_categories["_class"]
    if has_model_classes and not has_control:
        subplot_titles = "model_class=" + subplots_categories["_class"]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles.values if isinstance(subplot_titles, pd.Series) else None,
        vertical_spacing=0.7 / rows**1.5,
    )

    if subplots_categories is not None:
        i = 0
        for _, cat in subplots_categories.iterrows():
            i += 1
            if has_control and not has_model_classes:
                subset_i = visual_df[control_feature] == cat[control_feature]
            elif has_model_classes and not has_control:
                subset_i = visual_df["_class"] == cat["_class"]
            elif has_control and has_model_classes:
                subset_i = (visual_df[control_feature] == cat[control_feature]) & (visual_df["_class"] == cat["_class"])
            else:
                raise RuntimeError("Cannot use subplot categories without control_feature or model classes.")

            sub_visual_df = visual_df[subset_i]
            _add_differences_traces(
                sub_visual_df, fig, protected_feature, max_subgroups_per_control_cat_to_display, row=i, col=1
            )
    else:
        _add_differences_traces(
            visual_df, fig, protected_feature, max_subgroups_per_control_cat_to_display, row=1, col=1
        )

    title = "Largest performance differences"
    if has_control and not has_model_classes:
        title += f" within {control_feature} categories"
    elif has_model_classes and not has_control:
        title += " model_class categories"
    if has_control and has_model_classes:
        title += f" within {control_feature} and model_class categories"

    n_subgroups = len(visual_df[protected_feature].unique())
    n_subgroups_shown = min(n_subgroups, max_subgroups_per_control_cat_to_display)
    title += f"<br><sup>(Showing {n_subgroups_shown}/{n_subgroups} {protected_feature} categories"
    n_cat = 1
    if has_control or has_model_classes:
        n_cat = len(visual_df[subplot_grouping].drop_duplicates())
        title += f" per subplot and {rows}/{n_cat} "
        if has_control and not has_model_classes:
            title += f"{control_feature}"
        elif has_model_classes and not has_control:
            title += "model_classes"
        else:
            title += f"({control_feature}, model_classes)"
        title += " categories"
    title += ")</sup>"

    fig.update_layout(title_text=title)
    fig.update_annotations(x=0, xanchor="left", font_size=12)
    fig.update_layout({f"xaxis{rows}_title": f"{scorer_name} score"})
    fig.update_layout({f"yaxis{i}_title": protected_feature for i in range(1, rows + 1)})
    fig.update_layout({f"yaxis{i}_tickmode": "linear" for i in range(1, rows + 1)})

    fig.update_layout(height=150 + 50 * rows + 20 * rows * n_subgroups_shown)

    _add_legend(fig)

    return fig
