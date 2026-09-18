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
    hitbox_width: float | None = None
    hitbox_height: float | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("target radius must be > 0")
        if self.hitbox_width is not None and self.hitbox_width <= 0.0:
            raise ValueError("hitbox_width must be > 0 when provided")
        if self.hitbox_height is not None and self.hitbox_height <= 0.0:
            raise ValueError("hitbox_height must be > 0 when provided")
        if (self.hitbox_width is None) != (self.hitbox_height is None):
            raise ValueError("hitbox_width and hitbox_height must be provided together")
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
            raise ValueError(
                f"{type(self).__name__} requires exactly {self.arm_count} arm configs"
            )
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
        arms = self._arm_geometry()
        return {
            "elapsed_s": self.elapsed_s,
            "done": self.done,
            "endpoints": [list(arm["endpoint"]) for arm in arms],
            "arms": arms,
            "targets": [
                {
                    "id": target.target_id,
                    "x": target.x,
                    "y": target.y,
                    "radius": target.radius,
                    "owner_arm": target.owner_arm,
                    "hitbox_width": target.hitbox_width,
                    "hitbox_height": target.hitbox_height,
                    "hitbox_shape": "box" if target.hitbox_width is not None else "circle",
                }
                for target in self.targets
            ],
        }

    def _arm_geometry(self) -> list[dict[str, list[float]]]:
        geometry = []
        for state, config in zip(self.arms, self.arm_configs, strict=True):
            elbow = (
                config.shoulder_x + config.upper_arm_length * math.cos(state.shoulder_angle_rad),
                config.shoulder_y + config.upper_arm_length * math.sin(state.shoulder_angle_rad),
            )
            endpoint = state.endpoint(config)
            geometry.append({
                "shoulder": [config.shoulder_x, config.shoulder_y],
                "elbow": [elbow[0], elbow[1]],
                "endpoint": [endpoint[0], endpoint[1]],
            })
        return geometry

    def position_arm_to(self, arm_index: int, x: float, y: float) -> dict[str, object]:
        """Place one arm endpoint at a reachable point using inverse kinematics."""
        if arm_index < 0 or arm_index >= self.arm_count:
            raise ValueError(f"arm_index must be in [0, {self.arm_count})")
        config = self.arm_configs[arm_index]
        dx = float(x) - config.shoulder_x
        dy = float(y) - config.shoulder_y
        # Circular layouts can produce signed-zero-sized residuals on the
        # horizontal/vertical axes.  Keep atan2 from selecting the opposite
        # IK branch because of a value such as -5e-17.
        if abs(dx) < 1e-12:
            dx = 0.0
        if abs(dy) < 1e-12:
            dy = 0.0
        distance = math.hypot(dx, dy)
        minimum = abs(config.upper_arm_length - config.forearm_length)
        maximum = config.upper_arm_length + config.forearm_length
        tolerance = 1e-6
        if distance < minimum - tolerance or distance > maximum + tolerance:
            raise ValueError(
                f"target ({x}, {y}) is outside arm workspace "
                f"[{minimum:.4f}, {maximum:.4f}]"
            )
        cosine = (
            distance * distance
            - config.upper_arm_length * config.upper_arm_length
            - config.forearm_length * config.forearm_length
        ) / (2.0 * config.upper_arm_length * config.forearm_length)
        elbow = math.acos(max(-1.0, min(1.0, cosine)))
        shoulder = math.atan2(dy, dx) - math.atan2(
            config.forearm_length * math.sin(elbow),
            config.upper_arm_length + config.forearm_length * math.cos(elbow),
        )
        state = self.arms[arm_index]
        state.shoulder_angle_rad = self._clip_angle(shoulder, config)
        state.elbow_angle_rad = self._clip_angle(elbow, config)
        return self.observation()

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
            dx = abs(endpoint[0] - target.x)
            dy = abs(endpoint[1] - target.y)
            if target.hitbox_width is not None:
                hit = (
                    dx <= target.hitbox_width / 2.0
                    and dy <= target.hitbox_height / 2.0
                )
                distance = math.hypot(dx, dy)
            else:
                distance = math.hypot(dx, dy)
                hit = distance <= target.radius
            if hit and distance < best_distance:
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


class OneArmWorld(FourArmWorld):
    """Single two-joint arm world for the circular-keyboard curriculum."""

    arm_count = 1

    @staticmethod
    def default_arm_configs() -> tuple[VirtualArmConfig, ...]:
        return (
            VirtualArmConfig(
                0.50,
                0.50,
                upper_arm_length=0.20,
                forearm_length=0.16,
                joint_speed_rad_s=3.0,
            ),
        )


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
