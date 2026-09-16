import pytest

from drosomath.malecns.presence_graded import PresenceMasteryConfig, mastery_passed


def test_mastery_gate_uses_requested_threshold():
    assert mastery_passed(0.85, 0.85)
    assert mastery_passed(0.91, 0.85)
    assert not mastery_passed(0.849, 0.85)


def test_presence_mastery_defaults_allow_two_retraining_cycles():
    config = PresenceMasteryConfig()
    assert config.max_attempts_per_phase == 3
    assert config.fixed_full_trials == 2048
    assert config.varied_full_trials == 3072
    assert config.varied_dropout_trials == 3072
    assert config.mastery_accuracy == 0.85


def test_presence_mastery_rejects_invalid_attempt_count():
    with pytest.raises(ValueError):
        PresenceMasteryConfig(max_attempts_per_phase=0)
