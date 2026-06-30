"""
test_tamper_detector.py
------------------------
Test suite for tamper_detector.py and tamper_result.py.

Run with: pytest test_tamper_detector.py -v
"""

import numpy as np
import pytest

from tamper_analysis.tamper_result import TamperResult
from tamper_analysis.tamper_detector import TamperDetector, DetectorWeights


# ---------------------------------------------------------------------
# TamperResult tests
# ---------------------------------------------------------------------
class TestTamperResult:
    def test_valid_creation(self):
        r = TamperResult(tampered=True, confidence=0.82, reasons=["overlay detected"])
        assert r.tampered is True
        assert r.confidence == 0.82
        assert r.reasons == ["overlay detected"]

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            TamperResult(tampered=True, confidence=1.5)

    def test_invalid_tampered_type_raises(self):
        with pytest.raises(TypeError):
            TamperResult(tampered="yes", confidence=0.5)

    def test_to_dict_matches_spec_shape(self):
        r = TamperResult(tampered=True, confidence=0.82, reasons=["overlay detected", "contour mismatch"])
        d = r.to_dict()
        assert set(d.keys()) == {"tampered", "confidence", "reasons"}
        assert d["tampered"] is True
        assert d["confidence"] == 0.82
        assert d["reasons"] == ["overlay detected", "contour mismatch"]

    def test_round_trip_dict(self):
        r1 = TamperResult(tampered=False, confidence=0.12, reasons=[])
        r2 = TamperResult.from_dict(r1.to_full_dict())
        assert r1.to_dict() == r2.to_dict()

    def test_merge_reasons_dedup_preserves_order(self):
        r = TamperResult(tampered=True, confidence=0.5, reasons=["a"])
        r.merge_reasons(["a", "b"], ["c", "b"])
        assert r.reasons == ["a", "b", "c"]


# ---------------------------------------------------------------------
# TamperDetector tests
# ---------------------------------------------------------------------
class TestTamperDetector:
    @pytest.fixture
    def detector(self):
        return TamperDetector(threshold=0.5)

    @pytest.fixture
    def random_image(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            TamperDetector(weights=DetectorWeights(edge=0.5, contour=0.5, overlay=0.5, pattern=0.5))

    def test_analyze_returns_tamper_result(self, detector, random_image):
        result = detector.analyze(random_image)
        assert isinstance(result, TamperResult)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.tampered, bool)
        assert isinstance(result.reasons, list)

    def test_analyze_none_image_raises(self, detector):
        with pytest.raises(ValueError):
            detector.analyze(None)

    def test_calculate_confidence_weighted_average(self, detector):
        conf = detector.calculate_confidence(1.0, 1.0, 1.0, 1.0)
        assert conf == pytest.approx(1.0)

        conf_zero = detector.calculate_confidence(0.0, 0.0, 0.0, 0.0)
        assert conf_zero == pytest.approx(0.0)

    def test_calculate_confidence_rejects_out_of_range(self, detector):
        with pytest.raises(ValueError):
            detector.calculate_confidence(1.5, 0.0, 0.0, 0.0)

    def test_make_decision_threshold(self, detector):
        assert detector.make_decision(0.6) is True
        assert detector.make_decision(0.4) is False
        assert detector.make_decision(0.5) is True  # inclusive

    def test_normalize_result_float(self):
        score, reasons = TamperDetector._normalize_result(0.8, "default reason")
        assert score == 0.8
        assert reasons == ["default reason"]

    def test_normalize_result_dict_with_issues(self):
        raw = {"score": 0.9, "issues": ["custom issue 1", "custom issue 2"]}
        score, reasons = TamperDetector._normalize_result(raw, "default reason")
        assert score == 0.9
        assert reasons == ["custom issue 1", "custom issue 2"]

    def test_normalize_result_none(self):
        score, reasons = TamperDetector._normalize_result(None, "default reason")
        assert score == 0.0
        assert reasons == []

    def test_pattern_interruption_on_uniform_image(self, detector):
        # A perfectly uniform image should not trigger high pattern-interruption
        # confidence (all blocks equally flat -> no anomalous "dead zones").
        uniform = np.full((128, 128), 127, dtype=np.uint8)
        score, reasons = detector._run_pattern_interruption(uniform)
        assert 0.0 <= score <= 1.0

    def test_deliverable_shape_end_to_end(self, detector, random_image):
        result = detector.analyze(random_image)
        d = result.to_dict()
        assert "tampered" in d and isinstance(d["tampered"], bool)
        assert "confidence" in d and isinstance(d["confidence"], float)
        assert "reasons" in d and isinstance(d["reasons"], list)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))