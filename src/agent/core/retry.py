from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from agent.events import ErrorEvent


RandomSource = Callable[[], float]


@dataclass(frozen=True, slots = True)
class RetryPolicy:

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter_fraction: float = 0.20
    random_source: RandomSource = field(
        default=random.random,
        repr=False,
        compare=False,
    )


    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")

        if self.base_delay <= 0:
            raise ValueError("base_delay must be greater than zero")

        if self.max_delay < self.base_delay:
            raise ValueError(
                "max_delay must be greater than or equal to base_delay"
            )

        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError(
                "jitter_fraction must be between 0 and 1"
            )


    def should_retry(self,error: ErrorEvent,retries_completed: int):
        return (
            error.retryable
            and retries_completed < self.max_retries
        )


    def delay_for(self, error : ErrorEvent, retry_number : int):
        if retry_number < 1:
            raise ValueError("retry_number must be at least 1")

        #sometimes the server replays , dont retry for 5s
        if error.retry_after is not None:
            delay = error.retry_after
        else:
            delay = min(
                self.max_delay,
                self.base_delay * (2 ** (retry_number - 1)),
            )

        random_value = self.random_source()
        if not 0 <= random_value < 1:
            raise ValueError(
                "random_source must return a value in [0, 1)"
            )

        return delay * (
            1 + self.jitter_fraction * random_value
        )