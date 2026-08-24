"""Trivial smoke tests so the scaffold's pytest run is green."""

from app.config import get_settings


def test_settings_load():
    s = get_settings()
    assert s.database_url.startswith("postgres://")
    from app import detector

    assert detector.get_device() in ("mps", "cpu", "cuda")
    # Thresholds must sit in the plateau the sweep measured, not at an edge.
    assert 0.2 <= s.teacher_conf <= 0.6
    assert 0.0 < s.zone_conf <= 1.0


def test_app_importable():
    from app.main import app

    assert app.title == "Classroom Surveillance ML Service"


def test_class_ids_are_the_documented_order():
    """The class ids are the whole contract between the model and every KPI.

    detector._check_class_order enforces this against a real checkpoint at load
    time; this catches an accidental edit of the constants themselves.
    """
    from app import models

    assert (models.CLASS_DOOR, models.CLASS_SCREEN, models.CLASS_TEACHER) == (0, 1, 2)
    assert (models.CLASS_POINTING, models.CLASS_WRITING) == (3, 4)
    assert models.CLASS_NAMES[models.CLASS_TEACHER] == "teacher"
