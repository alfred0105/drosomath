"""Virtual keyboard and token-to-click matching task for the four-arm body."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .virtual_body import ArmAction, FourArmWorld, VirtualStepResult, VirtualTarget
from .virtual_motor import FourArmMotorAdapter


KEY_LABELS: tuple[str, ...] = ("한글", "영어", "O", "X") + tuple(str(x) for x in range(10))


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
        return min(self.width, self.height) * 0.45

    def as_target(self) -> VirtualTarget:
        return VirtualTarget(
            target_id=self.label,
            x=self.x,
            y=self.y,
            radius=self.click_radius,
            owner_arm=self.owner_arm,
        )


class VirtualKeyboard:
    """Compact keyboard containing the first language/control token set."""

    def __init__(self, keys: Iterable[VirtualKey] | None = None) -> None:
        rows = (
            ("한글", "영어", "O", "X"),
            tuple(str(x) for x in range(1, 8)),
            ("7", "8", "9", "0"),
        )
        default_keys = []
        for row_index, row in enumerate(rows):
            y = 0.62 - row_index * 0.14
            spacing = 0.13
            start = 0.24 if len(row) == 4 else 0.11
            for col_index, label in enumerate(row):
                # The third row intentionally repeats 7 only in the layout
                # source? Reject duplicates below and keep a single key map.
                if any(key.label == label for key in default_keys):
                    continue
                default_keys.append(
                    VirtualKey(
                        label=label,
                        x=start + col_index * spacing,
                        y=y,
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
    """Present one token and reward any arm that clicks its matching key."""

    def __init__(
        self,
        *,
        keyboard: VirtualKeyboard | None = None,
        world: FourArmWorld | None = None,
        motor: FourArmMotorAdapter | None = None,
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
