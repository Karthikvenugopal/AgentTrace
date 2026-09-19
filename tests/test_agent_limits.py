from agenttrace.agent.limits import ExecutionBudget
from agenttrace.config import AgentLimits


def test_budget_reports_distinct_limits() -> None:
    limits = AgentLimits(max_iterations=2, max_wall_time_seconds=10, max_total_tokens=20)
    budget = ExecutionBudget(limits, started_monotonic=100)
    assert budget.stop_reason(now_monotonic=101, next_iteration=0) is None
    budget.add_usage(15, 5)
    assert budget.stop_reason(now_monotonic=101, next_iteration=1) == "token budget reached"
    assert budget.stop_reason(now_monotonic=111, next_iteration=1) == "wall-time limit reached"
    assert budget.stop_reason(now_monotonic=101, next_iteration=2) == "iteration limit reached"
