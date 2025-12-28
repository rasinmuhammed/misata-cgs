"""Conditional Intervention Synthesizer for population-level what-if analysis.

NOTE: This implements conditional predictions under intervention (Rung 2),
not true individual counterfactuals (Rung 3). The distinction:
- Rung 2 (This): P(Y | do(X=x)) - What happens if we intervene?
- Rung 3 (Not this): P(Y_x | X=x', Y=y) - What would have happened to THIS individual?

For true counterfactuals, proper noise preservation through the SCM is required.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from typing import Optional, Dict, Tuple


class ConditionalInterventionSynthesizer:
    """
    MISATA Conditional Intervention Synthesizer.
    
    Extends copula synthesis with population-level intervention capability:
    - Learns conditional distributions P(Y | PA_Y)
    - Enables do(X=x) interventions for what-if analysis
    - Computes Average Treatment Effects (ATE)
    
    NOTE: Despite storing individual noise terms, the current implementation
    provides conditional predictions under intervention, not true counterfactuals.
    """
    
    def __init__(
        self,
        target_col: str,
        task: str = 'classification',
        random_state: int = 42
    ):
        self.target_col = target_col
        self.task = task
        self.random_state = random_state
        self._fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ConditionalInterventionSynthesizer':
        """
        Fit the synthesizer and store noise terms for counterfactuals.
        """
        self.columns = list(df.columns)
        self.n_samples = len(df)
        self.original_data = df.copy()
        
        # Store marginals
        self.marginals = {}
        for col in self.columns:
            values = df[col].values
            self.marginals[col] = {
                'sorted': np.sort(values),
                'min': values.min(),
                'max': values.max()
            }
        
        # Compute uniform representation (store for counterfactuals)
        self.uniform_data = pd.DataFrame()
        for col in self.columns:
            self.uniform_data[col] = stats.rankdata(df[col]) / (len(df) + 1)
        
        # Compute noise terms (latent representation)
        self.noise_terms = self.uniform_data.apply(
            lambda x: stats.norm.ppf(np.clip(x, 0.001, 0.999))
        )
        
        # Learn correlation
        corr_matrix = self.noise_terms.corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        self.corr_matrix = corr_matrix
        self.cholesky = np.linalg.cholesky(corr_matrix)
        self.cholesky_inv = np.linalg.inv(self.cholesky)
        
        # Fit target model
        feature_cols = [c for c in self.columns if c != self.target_col]
        if self.task == 'classification':
            self.target_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=self.random_state
            )
        else:
            self.target_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=self.random_state
            )
        
        self.target_model.fit(df[feature_cols], df[self.target_col])
        self.feature_cols = feature_cols
        self.target_rate = df[self.target_col].mean() if self.task == 'classification' else None
        
        self._fitted = True
        return self
    
    def conditional_intervention(
        self,
        individual_idx: int,
        intervention: Dict[str, float]
    ) -> pd.Series:
        """
        Compute conditional prediction under intervention.
        
        Given individual at index i, compute what Y would be
        under intervention do(X=x') using learned conditionals.
        
        NOTE: This is Rung 2 (intervention), not Rung 3 (counterfactual).
        
        Args:
            individual_idx: Index of individual in training data
            intervention: Dict of {variable: value} interventions
            
        Returns:
            Predicted outcome under intervention
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        
        # Get individual's noise terms
        individual_noise = self.noise_terms.iloc[individual_idx].values
        
        # Create counterfactual sample
        cf_values = {}
        for i, col in enumerate(self.columns):
            if col in intervention:
                # Intervened variable: set to intervention value
                cf_values[col] = intervention[col]
            elif col == self.target_col:
                continue  # Will compute below
            else:
                # Non-intervened: use original noise term to recover value
                u = stats.norm.cdf(individual_noise[i])
                sorted_vals = self.marginals[col]['sorted']
                positions = np.linspace(0, 1, len(sorted_vals))
                cf_values[col] = np.interp(u, positions, sorted_vals)
        
        # Compute counterfactual target
        X_cf = pd.DataFrame([{c: cf_values[c] for c in self.feature_cols}])
        
        if self.task == 'classification':
            prob = self.target_model.predict_proba(X_cf)[0, 1]
            # Use individual's noise to determine threshold
            original_prob = self.target_model.predict_proba(
                self.original_data[self.feature_cols].iloc[[individual_idx]]
            )[0, 1]
            original_outcome = self.original_data[self.target_col].iloc[individual_idx]
            
            # If original was positive and prob was high, counterfactual follows same logic
            if original_outcome == 1:
                cf_values[self.target_col] = 1 if prob >= original_prob * 0.8 else 0
            else:
                cf_values[self.target_col] = 1 if prob >= original_prob * 1.2 else 0
        else:
            cf_values[self.target_col] = self.target_model.predict(X_cf)[0]
        
        return pd.Series(cf_values)[self.columns]
    
    def intervention_batch(
        self,
        indices: list,
        intervention: Dict[str, float]
    ) -> pd.DataFrame:
        """Compute conditional interventions for multiple individuals."""
        results = []
        for idx in indices:
            cf = self.conditional_intervention(idx, intervention)
            results.append(cf)
        return pd.DataFrame(results)
    
    def average_treatment_effect(
        self,
        treatment_var: str,
        treatment_value: float,
        control_value: float,
        outcome_var: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Compute Average Treatment Effect (ATE).
        
        ATE = E[Y | do(X=treatment)] - E[Y | do(X=control)]
        
        Returns:
            (ATE, standard_error)
        """
        if outcome_var is None:
            outcome_var = self.target_col
        
        # Sample under treatment
        treatment_outcomes = []
        control_outcomes = []
        
        for idx in range(self.n_samples):
            cf_treat = self.conditional_intervention(idx, {treatment_var: treatment_value})
            cf_control = self.conditional_intervention(idx, {treatment_var: control_value})
            
            treatment_outcomes.append(cf_treat[outcome_var])
            control_outcomes.append(cf_control[outcome_var])
        
        treatment_outcomes = np.array(treatment_outcomes)
        control_outcomes = np.array(control_outcomes)
        
        ate = treatment_outcomes.mean() - control_outcomes.mean()
        se = np.sqrt(
            treatment_outcomes.var() / len(treatment_outcomes) +
            control_outcomes.var() / len(control_outcomes)
        )
        
        return ate, se
    
    def sample(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic samples (same as base synthesizer)."""
        if seed is None:
            seed = self.random_state
        rng = np.random.default_rng(seed)
        
        z = rng.standard_normal((n_samples, len(self.columns)))
        uniform = stats.norm.cdf(z @ self.cholesky.T)
        uniform = np.clip(uniform, 0.001, 0.999)
        
        synthetic_data = {}
        for i, col in enumerate(self.columns):
            if col == self.target_col:
                continue
            sorted_vals = self.marginals[col]['sorted']
            positions = np.linspace(0, 1, len(sorted_vals))
            synthetic_data[col] = np.interp(uniform[:, i], positions, sorted_vals)
        
        if self.target_col in self.columns:
            X_synth = pd.DataFrame({c: synthetic_data[c] for c in self.feature_cols})
            if self.task == 'classification':
                probs = self.target_model.predict_proba(X_synth)[:, 1]
                threshold = np.percentile(probs, (1 - self.target_rate) * 100)
                synthetic_data[self.target_col] = (probs >= threshold).astype(int)
            else:
                synthetic_data[self.target_col] = self.target_model.predict(X_synth)
        
        return pd.DataFrame(synthetic_data)[self.columns]
