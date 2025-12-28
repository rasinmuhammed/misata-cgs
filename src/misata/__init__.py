"""MISATA-CGS: Copula-Guided Causal Synthesis

A framework for causal simulation via Gaussian copulas and learned conditionals.
Enables do(X=x) interventions for policy simulation and decision support.
"""

__version__ = "0.1.0"

from misata.synthesizers.copula_guided import MISATASynthesizer
from misata.synthesizers.counterfactual import ConditionalInterventionSynthesizer

# Backward compatibility alias
CounterfactualSynthesizer = ConditionalInterventionSynthesizer

__all__ = ["MISATASynthesizer", "ConditionalInterventionSynthesizer", "CounterfactualSynthesizer"]
