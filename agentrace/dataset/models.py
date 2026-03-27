"""``EvalTask`` dataclass: prompt, constraints, gold references, metadata."""

from dataclasses import dataclass


@dataclass
class EvalTask:
    """Single evaluation example for runner consumption."""

    pass  # TODO: fields (id, instruction, tools_allowed, gold, ...)
