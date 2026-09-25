from __future__ import annotations

from dataclasses import dataclass

from agent.routing.classifier import classify_initial_prompt
from agent.routing.signals import RoutingSignals


@dataclass(frozen=True, slots=True)
class RoutingState:
    """
    Immutable routing state for one agent task.

    The initial prompt creates the first signal snapshot. Later tool failures
    or verification failures create new snapshots for later LLM turns.

    Raw prompts and raw tool output are intentionally not stored here.
    """

    signals: RoutingSignals

    @classmethod
    def from_initial_prompt(cls, prompt: str) -> RoutingState:
        """Create the first routing state for provider turn number one."""

        return cls(
            signals=classify_initial_prompt(
                prompt,
                turn_number=1,
            )
        )

    def observe_tool_result(
        self,
        *,
        is_error: bool,
    ) -> RoutingState:
        """
        Return updated state after one tool result.

        Successful tool results do not change routing facts. Failed tool
        results are evidence that the next model turn may need escalation.
        """

        if not is_error:
            return self

        return RoutingState(
            signals=self.signals.model_copy(
                update={
                    "tool_error_count": (
                        self.signals.tool_error_count + 1
                    ),
                }
            )
        )

    def observe_verification_failure(self) -> RoutingState:
        """
        Record that an external verifier found the task incomplete.

        Phase 5 defines this state transition now. A later verification
        integration will call it after a real test or evaluator failure.
        """

        return RoutingState(
            signals=self.signals.model_copy(
                update={
                    "verification_failure_count": (
                        self.signals.verification_failure_count + 1
                    ),
                }
            )
        )

    def begin_next_turn(self) -> RoutingState:
        """Return state for the next LLM turn without mutating this snapshot."""

        return RoutingState(
            signals=self.signals.model_copy(
                update={
                    "turn_number": self.signals.turn_number + 1,
                }
            )
        )