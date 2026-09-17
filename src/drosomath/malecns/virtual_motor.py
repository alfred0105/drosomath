"""Motor interface from population firing rates to the four-arm world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import argparse
import json

from .virtual_body import ArmAction, FourArmWorld, VirtualStepResult


@dataclass(frozen=True, slots=True)
class MotorChannel:
    arm_index: int
    name: str


def four_arm_motor_channels() -> tuple[MotorChannel, ...]:
    """Return the fixed 20-channel antagonistic motor layout."""
    channels = []
    for arm_index in range(4):
        channels.extend(
            (
                MotorChannel(arm_index, "shoulder_positive"),
                MotorChannel(arm_index, "shoulder_negative"),
                MotorChannel(arm_index, "elbow_positive"),
                MotorChannel(arm_index, "elbow_negative"),
                MotorChannel(arm_index, "click"),
            )
        )
    return tuple(channels)


@dataclass(frozen=True, slots=True)
class FourArmMotorConfig:
    """Rate-to-action calibration for one virtual motor head."""

    baseline_rate_hz: float = 5.0
    velocity_gain: float = 0.20
    click_threshold_hz: float = 18.0
    click_refractory_steps: int = 3

    def __post_init__(self) -> None:
        if self.baseline_rate_hz < 0.0:
            raise ValueError("baseline_rate_hz must be >= 0")
        if self.velocity_gain <= 0.0:
            raise ValueError("velocity_gain must be > 0")
        if self.click_threshold_hz <= self.baseline_rate_hz:
            raise ValueError("click_threshold_hz must exceed baseline_rate_hz")
        if self.click_refractory_steps < 0:
            raise ValueError("click_refractory_steps must be >= 0")


class FourArmMotorAdapter:
    """Convert 20 channel firing rates into four signed joint actions.

    Each signed joint command uses antagonistic positive/negative channels.
    Click is an event channel with a small refractory period so a sustained
    high firing rate does not produce a click on every simulation step.
    """

    def __init__(self, config: FourArmMotorConfig | None = None) -> None:
        self.config = config or FourArmMotorConfig()
        self.channels = four_arm_motor_channels()
        self._click_cooldown = [0] * 4

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def reset(self) -> None:
        self._click_cooldown = [0] * 4

    def decode(self, rates_hz: Sequence[float]) -> tuple[ArmAction, ...]:
        if len(rates_hz) != self.channel_count:
            raise ValueError(f"expected {self.channel_count} motor rates")
        cfg = self.config
        actions = []
        for arm_index in range(4):
            offset = arm_index * 5
            shoulder_positive = self._excess(rates_hz[offset])
            shoulder_negative = self._excess(rates_hz[offset + 1])
            elbow_positive = self._excess(rates_hz[offset + 2])
            elbow_negative = self._excess(rates_hz[offset + 3])
            click_rate = float(rates_hz[offset + 4])
            shoulder = self._signed_velocity(shoulder_positive, shoulder_negative)
            elbow = self._signed_velocity(elbow_positive, elbow_negative)
            click = click_rate >= cfg.click_threshold_hz and self._click_cooldown[arm_index] == 0
            if click:
                self._click_cooldown[arm_index] = cfg.click_refractory_steps
            elif self._click_cooldown[arm_index] > 0:
                self._click_cooldown[arm_index] -= 1
            actions.append(ArmAction(shoulder_velocity=shoulder, elbow_velocity=elbow, click=click))
        return tuple(actions)

    def step(self, world: FourArmWorld, rates_hz: Sequence[float]) -> VirtualStepResult:
        """Decode rates and advance the virtual body by one step."""
        return world.step(self.decode(rates_hz))

    def _excess(self, rate_hz: float) -> float:
        return max(0.0, float(rate_hz) - self.config.baseline_rate_hz)

    def _signed_velocity(self, positive_excess: float, negative_excess: float) -> float:
        value = (positive_excess - negative_excess) * self.config.velocity_gain
        return max(-1.0, min(1.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the four-arm virtual motor channel layout.")
    parser.parse_args()
    print(json.dumps({
        "channel_count": len(four_arm_motor_channels()),
        "channels": [
            {"arm": channel.arm_index, "name": channel.name}
            for channel in four_arm_motor_channels()
        ],
    }, indent=2))
