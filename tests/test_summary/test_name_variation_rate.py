from er_evaluation.summary import name_variation_rate


def test_name_variation_rate_basic(pred_one, pred_singleton, names_unique, names_common):
    assert name_variation_rate(pred_one, names_unique) == 1
    assert name_variation_rate(pred_one, names_common) == 0
    assert name_variation_rate(pred_singleton, names_unique) == 0
    assert name_variation_rate(pred_singleton, names_common) == 0
