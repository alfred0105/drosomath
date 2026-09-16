"""Array-based plasticity primitives for whole-connectome simulations.

These modules are intentionally separate from ``drosomath.core``. The core
package uses one Python object per synapse, which is convenient for tiny test
networks but is not appropriate for millions of MaleCNS edges.
"""

from .brain_adapter import PlasticSparseFlyBrain
from .homeostasis import BudgetNormalizationStats, OutgoingBudgetNormalizer
from .memory_consolidation import (
    AdaptiveReplayConfig,
    AdaptiveReplayScheduler,
    ConsolidationConfig,
    MemoryConsolidator,
    ProtectedRewardRule,
    ReplayConfig,
    ReplayScheduler,
)
from .plastic_state import PlasticStateConfig, SparsePlasticityState
from .usage_learning import LearningUpdateStats, UsageRewardRule

__all__ = [
    "AdaptiveReplayConfig",
    "AdaptiveReplayScheduler",
    "BudgetNormalizationStats",
    "ConsolidationConfig",
    "LearningUpdateStats",
    "MemoryConsolidator",
    "OutgoingBudgetNormalizer",
    "PlasticSparseFlyBrain",
    "PlasticStateConfig",
    "ProtectedRewardRule",
    "ReplayConfig",
    "ReplayScheduler",
    "SparsePlasticityState",
    "UsageRewardRule",
]
