import numpy as np

from scripts.audit_v9_qrel_sensitivity import qrel_array


def test_qrel_array_reproduces_frozen_combination() -> None:
    labels = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    facts = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
    result = qrel_array(labels, facts, label_weight=0.6, fact_weight=0.4)
    assert np.allclose(result, [0.4, 0.5, 0.6])


def test_qrel_array_rejects_non_unit_weights() -> None:
    values = np.asarray([0.5], dtype=np.float32)
    try:
        qrel_array(values, values, label_weight=0.6, fact_weight=0.5)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid qrel weights were accepted.")
