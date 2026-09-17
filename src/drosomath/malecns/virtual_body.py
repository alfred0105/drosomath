"""Pure-Python virtual body for action-based MaleCNS experiments.

The environment is intentionally independent of the connectome.  A later
motor adapter can translate descending-neuron activity into the 12-dimensional
action vector: two joint velocities plus one click channel for each of four
arms.  A virtual keyboard can then reuse the same target/click interface.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class VirtualArmConfig:
    """Geometry and control limits for one planar two-joint arm."""

    shoulder_x: float
    shoulder_y: float
    upper_arm_length: float = 0.22
    forearm_length: float = 0.18
    joint_speed_rad_s: float = 3.0
    joint_min_rad: float = -math.pi
    joint_max_rad: float = math.pi

    def __post_init__(self) -> None:
        if self.upper_arm_length <= 0.0 or self.forearm_length <= 0.0:
            raise ValueError("arm segment lengths must be > 0")
        if self.joint_speed_rad_s <= 0.0:
            raise ValueError("joint_speed_rad_s must be > 0")
        if self.joint_min_rad >= self.joint_max_rad:
            raise ValueError("joint_min_rad must be < joint_max_rad")


@dataclass(slots=True)
class VirtualArmState:
    """Mutable joint state for one virtual arm."""

    shoulder_angle_rad: float = 0.0
    elbow_angle_rad: float = 0.0
    click_count: int = 0

    def endpoint(self, config: VirtualArmConfig) -> tuple[float, float]:
        elbow_angle = self.shoulder_angle_rad + self.elbow_angle_rad
        return (
            config.shoulder_x
            + config.upper_arm_length * math.cos(self.shoulder_angle_rad)
            + config.forearm_length * math.cos(elbow_angle),
            config.shoulder_y
            + config.upper_arm_length * math.sin(self.shoulder_angle_rad)
            + config.forearm_length * math.sin(elbow_angle),
        )


@dataclass(frozen=True, slots=True)
class VirtualTarget:
    """A clickable target in normalized world coordinates."""

    target_id: str
    x: float
    y: float
    radius: float = 0.055
    owner_arm: int | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("target radius must be > 0")
        if self.owner_arm is not None and self.owner_arm not in range(4):
            raise ValueError("owner_arm must be one of 0, 1, 2, 3")


@dataclass(frozen=True, slots=True)
class ArmAction:
    """Normalized motor command for one arm.

    Joint velocities are in ``[-1, 1]`` and are scaled by the arm's configured
    maximum speed. ``click`` is an event sampled on the current simulation step.
    """

    shoulder_velocity: float = 0.0
    elbow_velocity: float = 0.0
    click: bool = False


@dataclass(frozen=True, slots=True)
class VirtualStepResult:
    reward: float
    done: bool
    clicked_arm: int | None
    clicked_target: str | None
    remaining_targets: tuple[str, ...]
    endpoints: tuple[tuple[float, float], ...]


class FourArmWorld:
    """Four-arm virtual body with target clicking and future keyboard support."""

    arm_count = 4
    action_width_per_arm = 3

    def __init__(
        self,
        arm_configs: Sequence[VirtualArmConfig] | None = None,
        *,
        dt_s: float = 0.05,
        wrong_click_penalty: float = -0.25,
        time_penalty: float = -0.001,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be > 0")
        if wrong_click_penalty > 0.0:
            raise ValueError("wrong_click_penalty must be <= 0")
        if time_penalty > 0.0:
            raise ValueError("time_penalty must be <= 0")
        self.arm_configs = tuple(arm_configs or self.default_arm_configs())
        if len(self.arm_configs) != self.arm_count:
            raise ValueError("FourArmWorld requires exactly four arm configs")
        self.dt_s = float(dt_s)
        self.wrong_click_penalty = float(wrong_click_penalty)
        self.time_penalty = float(time_penalty)
        self.arms = [VirtualArmState() for _ in range(self.arm_count)]
        self.targets: list[VirtualTarget] = []
        self.elapsed_s = 0.0
        self.done = False
        self.reset()

    @staticmethod
    def default_arm_configs() -> tuple[VirtualArmConfig, ...]:
        return (
            VirtualArmConfig(0.18, 0.18, joint_speed_rad_s=3.0),
            VirtualArmConfig(0.82, 0.18, joint_speed_rad_s=3.0),
            VirtualArmConfig(0.18, 0.82, joint_speed_rad_s=3.0),
            VirtualArmConfig(0.82, 0.82, joint_speed_rad_s=3.0),
        )

    @property
    def action_size(self) -> int:
        return self.arm_count * self.action_width_per_arm

    def reset(self, targets: Iterable[VirtualTarget] | None = None) -> dict[str, object]:
        """Reset arm joints and install targets; returns the initial observation."""
        self.arms = [VirtualArmState() for _ in range(self.arm_count)]
        self.targets = list(targets) if targets is not None else [
            VirtualTarget("target_0", 0.50, 0.50)
        ]
        self.elapsed_s = 0.0
        self.done = False
        return self.observation()

    def observation(self) -> dict[str, object]:
        return {
            "elapsed_s": self.elapsed_s,
            "done": self.done,
            "endpoints": [
                list(state.endpoint(config))
                for state, config in zip(self.arms, self.arm_configs, strict=True)
            ],
            "targets": [
                {
                    "id": target.target_id,
                    "x": target.x,
                    "y": target.y,
                    "radius": target.radius,
                    "owner_arm": target.owner_arm,
                }
                for target in self.targets
            ],
        }

    def step(self, actions: Sequence[ArmAction]) -> VirtualStepResult:
        """Advance one virtual-body step and process at most one click per arm."""
        if len(actions) != self.arm_count:
            raise ValueError("FourArmWorld.step requires exactly four arm actions")
        if self.done:
            return VirtualStepResult(
                reward=0.0,
                done=True,
                clicked_arm=None,
                clicked_target=None,
                remaining_targets=tuple(target.target_id for target in self.targets),
                endpoints=self._endpoints(),
            )

        for state, config, action in zip(self.arms, self.arm_configs, actions, strict=True):
            state.shoulder_angle_rad = self._clip_angle(
                state.shoulder_angle_rad
                + self._clip_unit(action.shoulder_velocity)
                * config.joint_speed_rad_s
                * self.dt_s,
                config,
            )
            state.elbow_angle_rad = self._clip_angle(
                state.elbow_angle_rad
                + self._clip_unit(action.elbow_velocity)
                * config.joint_speed_rad_s
                * self.dt_s,
                config,
            )

        endpoints = self._endpoints()
        reward = self.time_penalty
        clicked_arm = None
        clicked_target = None
        for arm_index, (state, action, endpoint) in enumerate(zip(self.arms, actions, endpoints, strict=True)):
            if not action.click:
                continue
            state.click_count += 1
            match = self._target_at(endpoint, arm_index)
            if match is None:
                reward += self.wrong_click_penalty
                continue
            self.targets.remove(match)
            reward += 1.0
            clicked_arm = arm_index
            clicked_target = match.target_id

        self.elapsed_s += self.dt_s
        self.done = not self.targets
        return VirtualStepResult(
            reward=reward,
            done=self.done,
            clicked_arm=clicked_arm,
            clicked_target=clicked_target,
            remaining_targets=tuple(target.target_id for target in self.targets),
            endpoints=endpoints,
        )

    def _target_at(self, endpoint: tuple[float, float], arm_index: int) -> VirtualTarget | None:
        best = None
        best_distance = float("inf")
        for target in self.targets:
            if target.owner_arm is not None and target.owner_arm != arm_index:
                continue
            distance = math.hypot(endpoint[0] - target.x, endpoint[1] - target.y)
            if distance <= target.radius and distance < best_distance:
                best = target
                best_distance = distance
        return best

    def _endpoints(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            state.endpoint(config)
            for state, config in zip(self.arms, self.arm_configs, strict=True)
        )

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _clip_angle(value: float, config: VirtualArmConfig) -> float:
        return max(config.joint_min_rad, min(config.joint_max_rad, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a four-arm virtual-body smoke test.")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    world = FourArmWorld()
    rows = [world.observation()]
    zero = ArmAction()
    for _ in range(args.steps):
        result = world.step((zero, zero, zero, zero))
        rows.append({"reward": result.reward, "done": result.done, "endpoints": result.endpoints})
    print(json.dumps({"action_size": world.action_size, "steps": rows}, indent=2))


if __name__ == "__main__":
    main()
