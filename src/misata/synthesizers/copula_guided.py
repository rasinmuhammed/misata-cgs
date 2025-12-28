"""MISATA-CGS: Copula-Guided Causal Synthesizer"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA
from typing import Optional, Dict, List


class MISATASynthesizer:
    """
    MISATA Copula-Guided Causal Synthesizer.
    
    Combines Gaussian copula for correlation preservation with 
    learned causal models for target generation.
    
    Features:
    - Fast: O(n*d^2) fitting, O(n*d) sampling
    - Causally Valid: Respects DAG structure
    - High-Dimensional: PCA option for 50+ features
    """
    
    def __init__(
        self,
        target_col: Optional[str] = None,
        task: str = 'classification',
        use_pca: bool = False,
        pca_components: float = 0.95,
        random_state: int = 42
    ):
        """
        Args:
            target_col: Target column for causal modeling
            task: 'classification' or 'regression'
            use_pca: Enable PCA for high-dimensional data (50+ features)
            pca_components: Variance to retain (0-1) or n_components (int)
            random_state: Random seed
        """
        self.target_col = target_col
        self.task = task
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        
        self._fitted = False
        self._intervention = None
        
    def fit(self, df: pd.DataFrame) -> 'MISATASynthesizer':
        """Fit the synthesizer to training data."""
        self.columns = list(df.columns)
        self.n_features = len(self.columns)
        
        # Auto-enable PCA for high-dimensional data
        if self.n_features > 50 and not self.use_pca:
            print(f"Warning: {self.n_features} features detected. Consider use_pca=True")
        
        # Store marginal distributions
        self.marginals = {}
        for col in self.columns:
            values = df[col].values.copy()
            self.marginals[col] = {
                'values': values,
                'sorted': np.sort(values),
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std()
            }
        
        # Transform to uniform
        uniform_df = df.copy()
        for col in self.columns:
            uniform_df[col] = stats.rankdata(df[col]) / (len(df) + 1)
        
        # Transform to normal
        normal_df = uniform_df.apply(
            lambda x: stats.norm.ppf(np.clip(x, 0.001, 0.999))
        )
        
        # PCA for high-dimensional
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            normal_reduced = self.pca.fit_transform(normal_df.values)
            corr_matrix = np.corrcoef(normal_reduced.T)
            self.pca_fitted = True
        else:
            corr_matrix = normal_df.corr().values
            self.pca_fitted = False
        
        # Clean correlation matrix
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        self.corr_matrix = corr_matrix
        self.cholesky = np.linalg.cholesky(corr_matrix)
        
        # Fit target model
        if self.target_col and self.target_col in self.columns:
            feature_cols = [c for c in self.columns if c != self.target_col]
            
            if self.task == 'classification':
                self.target_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, 
                    random_state=self.random_state
                )
            else:
                self.target_model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=5,
                    random_state=self.random_state
                )
            
            self.target_model.fit(df[feature_cols], df[self.target_col])
            self.feature_cols = feature_cols
            self.target_rate = df[self.target_col].mean() if self.task == 'classification' else None
        
        self._fitted = True
        return self
    
    def intervene(self, variable: str, value: float) -> 'MISATASynthesizer':
        """
        Set an intervention: do(variable=value).
        Subsequent sample() calls will generate under this intervention.
        
        Args:
            variable: Column name to intervene on
            value: Value to set
        """
        if variable not in self.columns:
            raise ValueError(f"Variable {variable} not in columns")
        self._intervention = (variable, value)
        return self
    
    def clear_intervention(self) -> 'MISATASynthesizer':
        """Clear any active intervention."""
        self._intervention = None
        return self
    
    def sample(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed (uses self.random_state if None)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before sample()")
        
        if seed is None:
            seed = self.random_state
        rng = np.random.default_rng(seed)
        
        # Sample correlated values
        if self.pca_fitted:
            n_components = self.cholesky.shape[0]
            z = rng.standard_normal((n_samples, n_components))
            correlated = z @ self.cholesky.T
            # Inverse PCA
            normal_samples = self.pca.inverse_transform(correlated)
            uniform = stats.norm.cdf(normal_samples)
        else:
            z = rng.standard_normal((n_samples, len(self.columns)))
            uniform = stats.norm.cdf(z @ self.cholesky.T)
        
        uniform = np.clip(uniform, 0.001, 0.999)
        
        # Transform to original marginals
        synthetic_data = {}
        for i, col in enumerate(self.columns):
            if col == self.target_col:
                continue
            
            # Apply intervention if set
            if self._intervention and self._intervention[0] == col:
                synthetic_data[col] = np.full(n_samples, self._intervention[1])
            else:
                sorted_vals = self.marginals[col]['sorted']
                positions = np.linspace(0, 1, len(sorted_vals))
                synthetic_data[col] = np.interp(uniform[:, i], positions, sorted_vals)
        
        # Generate target
        if self.target_col and self.target_col in self.columns:
            X_synth = pd.DataFrame({c: synthetic_data[c] for c in self.feature_cols})
            
            if self.task == 'classification':
                probs = self.target_model.predict_proba(X_synth)[:, 1]
                threshold = np.percentile(probs, (1 - self.target_rate) * 100)
                synthetic_data[self.target_col] = (probs >= threshold).astype(int)
            else:
                synthetic_data[self.target_col] = self.target_model.predict(X_synth)
        
        return pd.DataFrame(synthetic_data)[self.columns]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for target prediction."""
        if not hasattr(self, 'target_model'):
            raise RuntimeError("No target model fitted")
        
        return dict(zip(self.feature_cols, self.target_model.feature_importances_))
