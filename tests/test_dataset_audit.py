from log_triage.dataset_audit import DATASET_PATH, MANIFEST_PATH


def test_dataset_audit_paths_exist() -> None:
    assert DATASET_PATH.is_file()
    assert MANIFEST_PATH.is_file()
