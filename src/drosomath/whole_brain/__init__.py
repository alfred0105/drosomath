"""Array-based plasticity primitives for whole-connectome simulations.

These modules are intentionally separate from ``drosomath.core``.  The core
package uses one Python object per synapse, which is convenient for tiny test
networks but is not appropriate for tens of millions of MaleCNS edges.
"""

from .homeostasis import BudgetNormalizationStats, OutgoingBudgetNormalizer
from .plastic_state import PlasticStateConfig, SparsePlasticityState
from .usage_learning import LearningUpdateStats, UsageRewardRule

__all__ = [
    "BudgetNormalizationStats",
    "LearningUpdateStats",
    "OutgoingBudgetNormalizer",
    "PlasticStateConfig",
    "SparsePlasticityState",
    "UsageRewardRule",
]
