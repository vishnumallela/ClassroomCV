"""Trivial smoke tests so the scaffold's pytest run is green."""

from app.config import get_settings


def test_settings_load():
    s = get_settings()
    assert s.database_url.startswith("postgres://")
    # device + model are now device-aware ('auto'); assert the RESOLVED values.
    from app import detector

    assert detector.get_device() in ("mps", "cpu", "cuda")
    resolved = detector.resolve_model_name()
    assert resolved.endswith((".pt", ".engine"))
    # the 'auto' default is a YOLO pose weight
    assert "pose" in resolved


def test_app_importable():
    from app.main import app

    assert app.title == "Classroom Surveillance ML Service"


def test_lapjv_shim_matches_and_respects_cost_limit():
    """The 'lap' shim used for BoT-SORT matching assigns like lapjv."""
    import numpy as np

    from app.detector import _lapjv_shim

    cost = np.array([[0.1, 0.9, 0.9], [0.9, 0.2, 0.9]])
    total, x, y = _lapjv_shim(cost, extend_cost=True, cost_limit=0.5)
    assert x.tolist() == [0, 1]  # row assignments
    assert y.tolist() == [0, 1, -1]  # column 2 unassigned
    assert abs(total - 0.3) < 1e-9

    # everything above the cost limit stays unmatched
    total, x, y = _lapjv_shim(cost, extend_cost=True, cost_limit=0.05)
    assert x.tolist() == [-1, -1]
    assert y.tolist() == [-1, -1, -1]
    assert total == 0.0
