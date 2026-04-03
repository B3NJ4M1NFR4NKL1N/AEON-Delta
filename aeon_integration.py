"""
AEON Integration Controller — Backward Compatibility Wrapper
=============================================================

All integration classes and functions have been merged into
``aeon_server.py`` (v4.0.0).  This module re-exports them for
backward compatibility so existing imports continue to work:

    from aeon_integration import UnifiedTrainingCycleController
    from aeon_integration import IntegrationState

No code changes required in downstream consumers.
"""

from aeon_server import (  # noqa: F401
    IntegrationState,
    UnifiedTrainingCycleController,
    DashboardMetricsCollector,
    VT_WEIGHTS_PATH,
    get_integration_state,
    load_vt_weights_into_model,
    connect_feedback_bus,
    trace_output_to_premise,
    get_metrics_collector,
)

__all__ = [
    "IntegrationState",
    "UnifiedTrainingCycleController",
    "DashboardMetricsCollector",
    "VT_WEIGHTS_PATH",
    "get_integration_state",
    "load_vt_weights_into_model",
    "connect_feedback_bus",
    "trace_output_to_premise",
    "get_metrics_collector",
]
