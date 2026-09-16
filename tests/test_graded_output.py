from drosomath.malecns.graded_output import (
    BETWEEN_LEVELS,
    NO_OUTPUT,
    GradedPopulationCode,
    GradedRateCodeConfig,
)


def test_fixed_graded_population_levels_and_silence():
    code = GradedPopulationCode(
        100,
        GradedRateCodeConfig(
            base_rate_hz=1.0,
            level_step_hz=1.5,
            tolerance_hz=0.5,
            max_level=7,
        ),
    )

    level0 = code.observe(100, duration_ms=1000.0, target_level=0)
    assert level0.predicted_level == 0
    assert level0.status == "LEVEL_0"
    assert level0.correct
    assert level0.teaching_signal == 0.0

    level1 = code.observe(250, duration_ms=1000.0, target_level=1)
    assert level1.predicted_level == 1
    assert level1.status == "LEVEL_1"
    assert level1.correct
    assert level1.teaching_signal == 0.0

    silent = code.observe(0, duration_ms=1000.0, target_level=0)
    assert silent.predicted_level is None
    assert silent.status == NO_OUTPUT

    between = code.observe(180, duration_ms=1000.0, target_level=1)
    assert between.predicted_level is None
    assert between.status == BETWEEN_LEVELS


def test_teaching_signal_has_direction_and_is_bounded():
    code = GradedPopulationCode(
        100,
        GradedRateCodeConfig(
            base_rate_hz=1.0,
            level_step_hz=1.5,
            tolerance_hz=0.5,
            max_level=7,
            max_teaching_signal=1.0,
        ),
    )

    too_low = code.observe(100, duration_ms=1000.0, target_level=1)
    assert too_low.signed_error_hz > 0.0
    assert too_low.teaching_signal == 1.0

    too_high = code.observe(400, duration_ms=1000.0, target_level=1)
    assert too_high.signed_error_hz < 0.0
    assert too_high.teaching_signal == -1.0

    very_low = code.observe(0, duration_ms=1000.0, target_level=7)
    assert very_low.teaching_signal == 1.0


def test_output_code_is_explicitly_non_trainable():
    code = GradedPopulationCode(512)
    summary = code.summary()
    assert summary["type"] == "fixed_population_rate_code"
    assert summary["trainable_decoder"] is False
    assert summary["population_size"] == 512
