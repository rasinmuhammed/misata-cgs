"""Synthesizers package."""

from misata.synthesizers.copula_guided import MISATASynthesizer
from misata.synthesizers.counterfactual import ConditionalInterventionSynthesizer

# Backward compatibility alias
CounterfactualSynthesizer = ConditionalInterventionSynthesizer

__all__ = ["MISATASynthesizer", "ConditionalInterventionSynthesizer", "CounterfactualSynthesizer"]
