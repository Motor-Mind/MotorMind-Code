"""What the summariser is allowed to hand back."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class MemoryNote(BaseModel):
    """What the summariser hands the planner: short, cited, and actionable."""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""                                     # <= ~120 words: where things stand
    failures: List[str] = Field(default_factory=list)     # what went wrong + evidence + avoid
    facts: List[str] = Field(default_factory=list)        # spatial / identity facts learned
    advice: List[str] = Field(default_factory=list)       # wording, ordering, budgets
