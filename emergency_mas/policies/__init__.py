"""Decision policies: the two baselines live here; the agent system lives in emergency_mas.agents."""
from .baselines import NearestFirst, PriorityGreedy

__all__ = ["NearestFirst", "PriorityGreedy"]
