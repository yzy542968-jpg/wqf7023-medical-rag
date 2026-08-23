from medical_rag.similar_case.v10_split import (
    PARTITION_ORDER,
    assign_clusters,
    build_duplicate_clusters,
    canonical_fingerprint,
    normalized_report_text,
    report_index_spectrum,
)


def test_normalization_and_spectrum_are_deterministic() -> None:
    assert normalized_report_text("  No  EDEMA. ", " NORMAL\nCHEST ") == "no edema. normal chest"
    assert report_index_spectrum(" Normal ") == "normal"
    assert report_index_spectrum("NO INDEXING") == "indeterminate"
    assert report_index_spectrum("Cardiomegaly") == "abnormal"


def test_duplicate_clusters_join_exact_near_and_exact_image_cases() -> None:
    case_ids = ["A", "B", "C", "D", "E"]
    texts = [
        "there is mild cardiomegaly without edema",
        "there is mild cardiomegaly without edema",
        "there is mild cardiomegaly without edema.",
        "lungs are clear",
        "",
    ]
    result = build_duplicate_clusters(
        case_ids,
        texts,
        cosine_threshold=0.90,
        image_sha256_by_case={"C": ["same"], "D": ["same"]},
    )
    groups = [set(cluster) for cluster in result.clusters]
    assert {"A", "B", "C", "D"} in groups
    assert {"E"} in groups
    assert result.exact_text_edges == 1
    assert result.exact_image_edges == 1


def test_cluster_assignment_is_disjoint_complete_and_reproducible() -> None:
    clusters = [("A", "B"), ("C",), ("D",), ("E",), ("F",), ("G",), ("H",), ("I",)]
    spectrum = {
        "A": "normal", "B": "normal", "C": "normal", "D": "normal",
        "E": "abnormal", "F": "abnormal", "G": "abnormal", "H": "indeterminate",
        "I": "normal",
    }
    fractions = {"train": 0.65, "calibration": 0.10, "validation": 0.10, "test": 0.15}
    first = assign_clusters(clusters, spectrum, fractions, domain="test", seed=7040)
    second = assign_clusters(clusters, spectrum, fractions, domain="test", seed=7040)
    assert first == second
    assert tuple(first) == PARTITION_ORDER
    assert set().union(*(set(values) for values in first.values())) == set(spectrum)
    assert sum(len(values) for values in first.values()) == len(spectrum)
    membership = {case_id: partition for partition, values in first.items() for case_id in values}
    assert membership["A"] == membership["B"]


def test_fingerprint_is_order_invariant() -> None:
    assert canonical_fingerprint(["B", "A"]) == canonical_fingerprint(["A", "B"])
