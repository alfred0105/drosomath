"""Virtual keyboards and token-to-click tasks for virtual motor bodies."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .virtual_body import (
    ArmAction,
    FourArmWorld,
    OneArmWorld,
    VirtualArmConfig,
    VirtualStepResult,
    VirtualTarget,
)
from .virtual_motor import FourArmMotorAdapter, OneArmMotorAdapter


KOREAN_JAMO: tuple[str, ...] = tuple("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ")
ENGLISH_KEYS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_KEYS: tuple[str, ...] = tuple(str(x) for x in range(10))

# O and X are real English letter keys, not mode labels. Hangul syllable
# composition (for example ㄱ + ㅏ -> 가) is intentionally a later layer.
KEY_LABELS: tuple[str, ...] = KOREAN_JAMO + ENGLISH_KEYS + DIGIT_KEYS


@dataclass(frozen=True, slots=True)
class VirtualKey:
    label: str
    x: float
    y: float
    width: float = 0.10
    height: float = 0.10
    owner_arm: int | None = None

    @property
    def click_radius(self) -> float:
        # Kept for compatibility with circular VirtualTarget users. Keyboard
        # targets additionally carry the exact visual key box below.
        return max(0.025, min(self.width, self.height) * 0.45)

    def as_target(self) -> VirtualTarget:
        return VirtualTarget(
            target_id=self.label,
            x=self.x,
            y=self.y,
            radius=self.click_radius,
            owner_arm=self.owner_arm,
            hitbox_width=self.width,
            hitbox_height=self.height,
        )


class VirtualKeyboard:
    """Compact keyboard containing the first language/control token set."""

    def __init__(self, keys: Iterable[VirtualKey] | None = None) -> None:
        rows = (KOREAN_JAMO, ENGLISH_KEYS, DIGIT_KEYS)
        default_keys = []
        for row_index, row in enumerate(rows):
            y = 0.70 - row_index * 0.16
            gap = 0.008
            width = min(0.10, (0.86 - gap * (len(row) - 1)) / len(row))
            total = len(row) * width + (len(row) - 1) * gap
            start = (1.0 - total) / 2.0 + width / 2.0
            for col_index, label in enumerate(row):
                default_keys.append(
                    VirtualKey(
                        label=label,
                        x=start + col_index * (width + gap),
                        y=y,
                        width=width,
                        height=0.10,
                        owner_arm=None,
                    )
                )
        supplied = tuple(keys) if keys is not None else tuple(default_keys)
        labels = tuple(key.label for key in supplied)
        if len(set(labels)) != len(labels):
            raise ValueError("keyboard key labels must be unique")
        missing = tuple(label for label in KEY_LABELS if label not in labels)
        if missing:
            raise ValueError(f"keyboard is missing required labels: {missing}")
        self.keys = {key.label: key for key in supplied}

    def key(self, label: str) -> VirtualKey:
        try:
            return self.keys[str(label)]
        except KeyError as exc:
            raise KeyError(f"unknown keyboard label: {label!r}") from exc

    def labels(self) -> tuple[str, ...]:
        return tuple(self.keys)

    def prompt_vector(self, label: str) -> tuple[int, ...]:
        """Return a deterministic one-hot prompt for future visual encoding."""
        label = self.key(label).label
        return tuple(int(candidate == label) for candidate in KEY_LABELS)

    def layout(self) -> list[dict[str, object]]:
        return [
            {
                "label": key.label,
                "x": key.x,
                "y": key.y,
                "width": key.width,
                "height": key.height,
                "owner_arm": key.owner_arm,
            }
            for key in self.keys.values()
        ]


class FanKeyboard(VirtualKeyboard):
    """Place keys in five non-overlapping reachable forward-fan layers."""

    def __init__(
        self,
        *,
        center_x: float = 0.50,
        center_y: float = 0.50,
        inner_radius: float = 0.135,
        outer_radius: float = 0.34,
        start_angle_deg: float = -78.0,
        end_angle_deg: float = 78.0,
        layers: int = 5,
        key_size: float = 0.026,
    ) -> None:
        if inner_radius <= 0.0 or outer_radius <= inner_radius or key_size <= 0.0:
            raise ValueError("fan radii and key_size must define a positive range")
        if layers < 1 or len(KEY_LABELS) % layers:
            raise ValueError("layers must evenly divide the key count")
        if end_angle_deg <= start_angle_deg:
            raise ValueError("end_angle_deg must exceed start_angle_deg")
        keys = []
        keys_per_layer = len(KEY_LABELS) // layers
        start_angle = math.radians(start_angle_deg)
        end_angle = math.radians(end_angle_deg)
        full_circle = math.isclose(
            end_angle - start_angle,
            2.0 * math.pi,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        for layer in range(layers):
            if layers == 1:
                radial = (inner_radius + outer_radius) / 2.0
            else:
                radial = inner_radius + (outer_radius - inner_radius) * layer / (layers - 1)
            for slot in range(keys_per_layer):
                denominator = keys_per_layer if full_circle else max(1, keys_per_layer - 1)
                fraction = slot / denominator
                angle = start_angle + (end_angle - start_angle) * fraction
                label = KEY_LABELS[layer * keys_per_layer + slot]
                keys.append(
                    VirtualKey(
                        label=label,
                        x=center_x + radial * math.cos(angle),
                        y=center_y + radial * math.sin(angle),
                        width=key_size,
                        height=key_size,
                        owner_arm=None,
                    )
                )
        super().__init__(keys)

    @classmethod
    def for_arm(cls, arm_config: VirtualArmConfig, *, margin: float = 0.02) -> "FanKeyboard":
        """Fit the fan just inside the arm's two-link reachable workspace."""
        if margin <= 0.0:
            raise ValueError("margin must be > 0")
        max_reach = arm_config.upper_arm_length + arm_config.forearm_length - margin
        # Leave room around the inner arc so neighboring 0.026-wide key boxes
        # never touch, while retaining five usable radial layers.
        min_reach = abs(arm_config.upper_arm_length - arm_config.forearm_length) + 0.095
        if max_reach <= min_reach:
            raise ValueError("arm workspace is too small for the fan")
        return cls(
            center_x=arm_config.shoulder_x,
            center_y=arm_config.shoulder_y,
            inner_radius=min_reach,
            outer_radius=max_reach,
        )


class CircularKeyboard(FanKeyboard):
    """Place the same layered keys around a full 360-degree annulus."""

    def __init__(
        self,
        *,
        center_x: float = 0.50,
        center_y: float = 0.50,
        inner_radius: float = 0.12,
        outer_radius: float = 0.34,
        layers: int = 5,
        key_size: float = 0.026,
    ) -> None:
        super().__init__(
            center_x=center_x,
            center_y=center_y,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            start_angle_deg=-180.0,
            end_angle_deg=180.0,
            layers=layers,
            key_size=key_size,
        )


@dataclass(frozen=True, slots=True)
class KeyboardStepResult:
    prompt: str
    reward: float
    done: bool
    clicked_label: str | None
    clicked_arm: int | None
    remaining_targets: tuple[str, ...]
    prompt_vector: tuple[int, ...]


class KeyboardMatchingTask:
    """Present one token and reward an arm that clicks its matching key."""

    def __init__(
        self,
        *,
        keyboard: VirtualKeyboard | None = None,
        world: FourArmWorld | OneArmWorld | None = None,
        motor: FourArmMotorAdapter | OneArmMotorAdapter | None = None,
        seed: int = 7,
    ) -> None:
        self.keyboard = keyboard or VirtualKeyboard()
        self.world = world or FourArmWorld()
        self.motor = motor or FourArmMotorAdapter()
        self.rng = random.Random(seed)
        self.prompt = KEY_LABELS[0]
        self.reset()

    def reset(self, prompt: str | None = None) -> dict[str, object]:
        if prompt is None:
            prompt = self.rng.choice(KEY_LABELS)
        prompt = self.keyboard.key(prompt).label
        self.prompt = prompt
        self.motor.reset()
        self.world.reset([self.keyboard.key(prompt).as_target()])
        return self.observation()

    def observation(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "prompt_vector": self.keyboard.prompt_vector(self.prompt),
            "keyboard": self.keyboard.layout(),
            "body": self.world.observation(),
        }

    def step(self, actions: Sequence[ArmAction]) -> KeyboardStepResult:
        result = self.world.step(actions)
        return self._wrap_step(result)

    def step_from_rates(self, rates_hz: Sequence[float]) -> KeyboardStepResult:
        """Use decoded neural motor rates to control the virtual keyboard task."""
        return self._wrap_step(self.motor.step(self.world, rates_hz))

    def _wrap_step(self, result: VirtualStepResult) -> KeyboardStepResult:
        return KeyboardStepResult(
            prompt=self.prompt,
            reward=result.reward,
            done=result.done,
            clicked_label=result.clicked_target,
            clicked_arm=result.clicked_arm,
            remaining_targets=result.remaining_targets,
            prompt_vector=self.keyboard.prompt_vector(self.prompt),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the virtual keyboard matching smoke test.")
    parser.add_argument("--prompt", default="O", choices=KEY_LABELS)
    args = parser.parse_args()
    task = KeyboardMatchingTask(seed=7)
    observation = task.reset(args.prompt)
    print(json.dumps({
        "prompt": observation["prompt"],
        "prompt_vector": observation["prompt_vector"],
        "key_count": len(observation["keyboard"]),
        "labels": list(task.keyboard.labels()),
        "action_size": task.world.action_size,
        "motor_channel_count": task.motor.channel_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
