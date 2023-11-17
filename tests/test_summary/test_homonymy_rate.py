from er_evaluation.summary import homonymy_rate


def test_homonymy_rate_basic(pred_one, pred_singleton, names_unique, names_common):
    assert homonymy_rate(pred_one, names_unique) == 0
    assert homonymy_rate(pred_one, names_common) == 0
    assert homonymy_rate(pred_singleton, names_unique) == 0
    assert homonymy_rate(pred_singleton, names_common) == 1
