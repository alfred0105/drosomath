"""Array-based plasticity primitives for whole-connectome simulations.

These modules are intentionally separate from ``drosomath.core``. The core
package uses one Python object per synapse, which is convenient for tiny test
networks but is not appropriate for millions of MaleCNS edges.
"""

from .brain_adapter import PlasticSparseFlyBrain
from .homeostasis import BudgetNormalizationStats, ChannelHomeostasis, OutgoingBudgetNormalizer
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
from .plasticity_budget import PlasticityBudgetConfig, PlasticityBudgetManager
from .plasticity_budget_adaptation import (
    PlasticityBudgetAdaptation,
    PlasticityBudgetAdaptationConfig,
)
from .plasticity_need import PlasticityNeedConfig, PlasticityNeedRecord, PlasticityNeedTracker
from .structural_overlay import LearnedStructuralOverlay, StructuralOverlayConfig
from .usage_learning import LearningUpdateStats, RewardCredit, UsageRewardRule
from .directional_modulation import (
    DirectionalModulationConfig,
    DirectionalRouteHealth,
    DirectionalUpdate,
    OutputRouteIndex,
    PlasticityController,
)
from .performance import TimingProfiler

__all__ = [
    "AdaptiveReplayConfig",
    "AdaptiveReplayScheduler",
    "BudgetNormalizationStats",
    "ChannelHomeostasis",
    "ConsolidationConfig",
    "LearnedStructuralOverlay",
    "LearningUpdateStats",
    "RewardCredit",
    "MemoryConsolidator",
    "OutgoingBudgetNormalizer",
    "PlasticSparseFlyBrain",
    "PlasticStateConfig",
    "PlasticityBudgetConfig",
    "PlasticityBudgetManager",
    "PlasticityBudgetAdaptation",
    "PlasticityBudgetAdaptationConfig",
    "PlasticityNeedConfig",
    "PlasticityNeedRecord",
    "PlasticityNeedTracker",
    "ProtectedRewardRule",
    "ReplayConfig",
    "ReplayScheduler",
    "SparsePlasticityState",
    "StructuralOverlayConfig",
    "UsageRewardRule",
    "DirectionalModulationConfig",
    "DirectionalRouteHealth",
    "DirectionalUpdate",
    "OutputRouteIndex",
    "PlasticityController",
    "TimingProfiler",
]
