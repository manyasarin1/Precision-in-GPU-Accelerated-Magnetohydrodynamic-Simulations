from dataclasses import dataclass
from typing import Optional

VALID_PRECISIONS = {"fp64", "fp32", "fp16", "bf16", "adaptive"}

@dataclass
class RunResult:
    states: list
    runtime_sec: float
    precision: str
    memory_bytes: Optional[int] = None
    extra: Optional[dict] = None

    def __post_init__(self):
        assert self.precision in VALID_PRECISIONS
        assert len(self.states) > 0
