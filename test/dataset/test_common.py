from pts.dataset import FieldName


def test_dataset_fields():
    assert (
        "feat_static_cat" == FieldName.FEAT_STATIC_CAT
    ), "Error in the FieldName 'feat_static_cat'."
    assert (
        "feat_static_real" == FieldName.FEAT_STATIC_REAL
    ), "Error in the FieldName 'feat_static_real'."
    assert (
        "feat_dynamic_cat" == FieldName.FEAT_DYNAMIC_CAT
    ), "Error in the FieldName 'feat_dynamic_cat'."
    assert (
        "feat_dynamic_real" == FieldName.FEAT_DYNAMIC_REAL
    ), "Error in the FieldName 'feat_dynamic_real'."
