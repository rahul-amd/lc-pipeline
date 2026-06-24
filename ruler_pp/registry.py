"""Task registry. To add a task: implement a `Task` subclass and list it here."""

from __future__ import annotations

from .base import Task
from .tasks.common_words import CommonWords
from .tasks.count_distinct import CountDistinct
from .tasks.count_occurrences import CountOccurrences
from .tasks.count_predicate import CountPredicate
from .tasks.freq_words import FreqWords
from .tasks.graph import (
    GraphAncestors, GraphBFS, GraphChildren, GraphComponentSize, GraphCycle,
    GraphDescendants, GraphMaxDegree, GraphParents, GraphReachable,
    GraphShortestPath,
)
from .tasks.group_by import GroupBy
from .tasks.numeric_agg import NumericAgg
from .tasks.set_ops import SetOps
from .tasks.tally_by_category import TallyByCategory
from .tasks.top_k import TopK
from .tasks.variable_arithmetic import VariableArithmetic
from .tasks.variable_tracking import VariableTracking

_TASK_CLASSES = [
    # RULER-faithful
    CommonWords, FreqWords, VariableTracking,
    # ruler++ extras
    CountOccurrences, CountPredicate, CountDistinct, TallyByCategory,
    NumericAgg, GroupBy, TopK, SetOps, VariableArithmetic,
    # graph reasoning (GraphWalks-style)
    GraphBFS, GraphParents, GraphChildren, GraphDescendants, GraphAncestors,
    GraphShortestPath, GraphReachable, GraphCycle, GraphComponentSize,
    GraphMaxDegree,
]

TASKS: dict[str, Task] = {cls.name: cls() for cls in _TASK_CLASSES}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {', '.join(TASKS)}")
    return TASKS[name]
