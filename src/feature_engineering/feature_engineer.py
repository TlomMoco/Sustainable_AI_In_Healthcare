"""
Feature Engineering Module

This module provides comprehensive feature engineering capabilities for healthcare data,
including feature extraction, selection, and transformation techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, 
    mutual_info_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
import joblib
from pathlib import Path


class FeatureExtractor:
    """
    Handles feature extraction from raw healthcare data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.extractors = {}
    
    def extract_features(self, data: pd.DataFrame, feature_types: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract features based on specified types.
        
        Args:
            data: Input dataset
            feature_types: List of feature types to extract
            
        Returns:
            Tuple of (feature_data, extraction_report)
        """
        feature_data = data.copy()
        extraction_report = {
            'original_features': len(data.columns),
            'extracted_features': {},
            'total_new_features': 0
        }
        
        for feature_type in feature_types:
            if feature_type == 'statistical':
                new_features, report = self._extract_statistical_features(data)
                feature_data = pd.concat([feature_data, new_features], axis=1)
                extraction_report['extracted_features']['statistical'] = report
                
            elif feature_type == 'temporal':
                new_features, report = self._extract_temporal_features(data)
                feature_data = pd.concat([feature_data, new_features], axis=1)
                extraction_report['extracted_features']['temporal'] = report
                
            elif feature_type == 'interaction':
                new_features, report = self._extract_interaction_features(data)
                feature_data = pd.concat([feature_data, new_features], axis=1)
                extraction_report['extracted_features']['interaction'] = report
                
            elif feature_type == 'polynomial':
                new_features, report = self._extract_polynomial_features(data)
                feature_data = pd.concat([feature_data, new_features], axis=1)
                extraction_report['extracted_features']['polynomial'] = report
        
        extraction_report['final_features'] = len(feature_data.columns)
        extraction_report['total_new_features'] = extraction_report['final_features'] - extraction_report['original_features']
        
        return feature_data, extraction_report
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract statistical features from numerical columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=data.index), {'features_created': 0, 'method': 'statistical'}
        
        statistical_features = pd.DataFrame(index=data.index)
        
        # Rolling statistics (if data has temporal ordering)
        if len(data) > 10:  # Only if sufficient data points
            for col in numeric_cols:
                # Rolling mean and std
                statistical_features[f'{col}_rolling_mean_5'] = data[col].rolling(window=5, min_periods=1).mean()
                statistical_features[f'{col}_rolling_std_5'] = data[col].rolling(window=5, min_periods=1).std()
                
                # Lag features
                statistical_features[f'{col}_lag_1'] = data[col].shift(1)
                statistical_features[f'{col}_lag_2'] = data[col].shift(2)
        
        # Cross-column statistics
        if len(numeric_cols) > 1:
            # Sum and mean across columns
            statistical_features['total_sum'] = data[numeric_cols].sum(axis=1)
            statistical_features['total_mean'] = data[numeric_cols].mean(axis=1)
            statistical_features['total_std'] = data[numeric_cols].std(axis=1)
            statistical_features['total_min'] = data[numeric_cols].min(axis=1)
            statistical_features['total_max'] = data[numeric_cols].max(axis=1)
        
        report = {
            'features_created': len(statistical_features.columns),
            'method': 'statistical',
            'base_columns': numeric_cols
        }
        
        return statistical_features, report
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract temporal features from datetime columns."""
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if len(datetime_cols) == 0:
            # Try to find datetime-like columns
            potential_datetime_cols = []
            for col in data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(data[col].dropna().head(100))
                        potential_datetime_cols.append(col)
                    except:
                        continue
            
            if not potential_datetime_cols:
                return pd.DataFrame(index=data.index), {'features_created': 0, 'method': 'temporal'}
            
            datetime_cols = potential_datetime_cols
        
        temporal_features = pd.DataFrame(index=data.index)
        
        for col in datetime_cols:
            try:
                datetime_series = pd.to_datetime(data[col])
                
                # Extract temporal components
                temporal_features[f'{col}_year'] = datetime_series.dt.year
                temporal_features[f'{col}_month'] = datetime_series.dt.month
                temporal_features[f'{col}_day'] = datetime_series.dt.day
                temporal_features[f'{col}_dayofweek'] = datetime_series.dt.dayofweek
                temporal_features[f'{col}_hour'] = datetime_series.dt.hour
                temporal_features[f'{col}_quarter'] = datetime_series.dt.quarter
                
                # Cyclical features
                temporal_features[f'{col}_month_sin'] = np.sin(2 * np.pi * datetime_series.dt.month / 12)
                temporal_features[f'{col}_month_cos'] = np.cos(2 * np.pi * datetime_series.dt.month / 12)
                temporal_features[f'{col}_day_sin'] = np.sin(2 * np.pi * datetime_series.dt.day / 31)
                temporal_features[f'{col}_day_cos'] = np.cos(2 * np.pi * datetime_series.dt.day / 31)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract temporal features from {col}: {e}")
        
        report = {
            'features_created': len(temporal_features.columns),
            'method': 'temporal',
            'base_columns': datetime_cols
        }
        
        return temporal_features, report
    
    def _extract_interaction_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract interaction features between numerical columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return pd.DataFrame(index=data.index), {'features_created': 0, 'method': 'interaction'}
        
        interaction_features = pd.DataFrame(index=data.index)
        
        # Limit interactions to prevent feature explosion
        max_interactions = min(10, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
        interaction_count = 0
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Multiplication interaction
                interaction_features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                
                # Division interaction (avoid division by zero)
                col2_safe = data[col2].replace(0, np.nan)
                interaction_features[f'{col1}_div_{col2}'] = data[col1] / col2_safe
                
                # Addition and subtraction
                interaction_features[f'{col1}_plus_{col2}'] = data[col1] + data[col2]
                interaction_features[f'{col1}_minus_{col2}'] = data[col1] - data[col2]
                
                interaction_count += 4
        
        report = {
            'features_created': len(interaction_features.columns),
            'method': 'interaction',
            'base_columns': numeric_cols,
            'interactions_created': interaction_count
        }
        
        return interaction_features, report
    
    def _extract_polynomial_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract polynomial features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return pd.DataFrame(index=data.index), {'features_created': 0, 'method': 'polynomial'}
        
        # Limit to prevent feature explosion
        max_features = min(5, len(numeric_cols))
        selected_cols = numeric_cols[:max_features]
        
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        polynomial_data = poly.fit_transform(data[selected_cols])
        
        # Create feature names
        feature_names = poly.get_feature_names_out(selected_cols)
        
        polynomial_features = pd.DataFrame(
            polynomial_data, 
            columns=[f'poly_{name}' for name in feature_names],
            index=data.index
        )
        
        # Remove original features (already in data)
        original_feature_names = [f'poly_{col}' for col in selected_cols]
        polynomial_features = polynomial_features.drop(columns=original_feature_names)
        
        self.extractors['polynomial'] = poly
        
        report = {
            'features_created': len(polynomial_features.columns),
            'method': 'polynomial',
            'base_columns': selected_cols,
            'degree': 2
        }
        
        return polynomial_features, report


class FeatureSelector:
    """
    Handles feature selection using various methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.selectors = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, methods: List[str], 
                       task_type: str = 'classification') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select features using specified methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of selection methods
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (selected_features, selection_report)
        """
        selection_results = {}
        selection_report = {
            'original_features': len(X.columns),
            'methods_applied': [],
            'selected_features': {}
        }
        
        for method in methods:
            if method == 'univariate':
                selected_features, report = self._univariate_selection(X, y, task_type)
                selection_results[method] = selected_features
                selection_report['selected_features'][method] = report
                
            elif method == 'recursive':
                selected_features, report = self._recursive_selection(X, y, task_type)
                selection_results[method] = selected_features
                selection_report['selected_features'][method] = report
                
            elif method == 'lasso':
                selected_features, report = self._lasso_selection(X, y, task_type)
                selection_results[method] = selected_features
                selection_report['selected_features'][method] = report
                
            elif method == 'tree_based':
                selected_features, report = self._tree_based_selection(X, y, task_type)
                selection_results[method] = selected_features
                selection_report['selected_features'][method] = report
        
        # Combine results from all methods
        final_features = self._combine_selection_results(selection_results, X.columns)
        
        selection_report['methods_applied'] = methods
        selection_report['final_features'] = len(final_features)
        selection_report['features_removed'] = selection_report['original_features'] - len(final_features)
        
        return X[final_features], selection_report
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[List[str], Dict[str, Any]]:
        """Univariate feature selection."""
        k_best = min(int(len(X.columns) * 0.7), 50)  # Select top 70% or 50 features max
        
        if task_type == 'classification':
            # Handle mixed data types
            numeric_X = X.select_dtypes(include=[np.number])
            categorical_X = X.select_dtypes(exclude=[np.number])
            
            selected_features = []
            
            # For numeric features, use f_classif
            if not numeric_X.empty:
                selector_numeric = SelectKBest(score_func=f_classif, k=min(k_best, len(numeric_X.columns)))
                selector_numeric.fit(numeric_X, y)
                selected_features.extend(numeric_X.columns[selector_numeric.get_support()].tolist())
                self.selectors['univariate_numeric'] = selector_numeric
            
            # For categorical features, use chi2 (after encoding)
            if not categorical_X.empty:
                # Simple label encoding for chi2
                categorical_encoded = categorical_X.copy()
                for col in categorical_encoded.columns:
                    categorical_encoded[col] = pd.Categorical(categorical_encoded[col]).codes
                
                selector_categorical = SelectKBest(score_func=chi2, k=min(k_best, len(categorical_encoded.columns)))
                selector_categorical.fit(categorical_encoded, y)
                selected_features.extend(categorical_encoded.columns[selector_categorical.get_support()].tolist())
                self.selectors['univariate_categorical'] = selector_categorical
        
        else:  # regression
            selector = SelectKBest(score_func=f_regression, k=k_best)
            selector.fit(X.select_dtypes(include=[np.number]), y)
            selected_features = X.select_dtypes(include=[np.number]).columns[selector.get_support()].tolist()
            self.selectors['univariate'] = selector
        
        report = {
            'method': 'univariate',
            'features_selected': len(selected_features),
            'selection_threshold': k_best
        }
        
        return selected_features, report
    
    def _recursive_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[List[str], Dict[str, Any]]:
        """Recursive feature elimination."""
        # Use only numeric features for RFE
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return [], {'method': 'recursive', 'features_selected': 0}
        
        n_features = max(1, min(int(len(numeric_X.columns) * 0.5), 20))
        
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(numeric_X, y)
        
        selected_features = numeric_X.columns[selector.get_support()].tolist()
        self.selectors['recursive'] = selector
        
        report = {
            'method': 'recursive',
            'features_selected': len(selected_features),
            'target_features': n_features
        }
        
        return selected_features, report
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[List[str], Dict[str, Any]]:
        """LASSO-based feature selection."""
        # Use only numeric features
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return [], {'method': 'lasso', 'features_selected': 0}
        
        if task_type == 'classification':
            # Use LogisticRegression with L1 penalty
            selector = SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000),
                threshold='median'
            )
        else:
            # Use LassoCV for regression
            selector = SelectFromModel(LassoCV(cv=5, random_state=42), threshold='median')
        
        selector.fit(numeric_X, y)
        selected_features = numeric_X.columns[selector.get_support()].tolist()
        
        self.selectors['lasso'] = selector
        
        report = {
            'method': 'lasso',
            'features_selected': len(selected_features),
            'threshold': 'median'
        }
        
        return selected_features, report
    
    def _tree_based_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Tuple[List[str], Dict[str, Any]]:
        """Tree-based feature selection using feature importance."""
        # Use only numeric features
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return [], {'method': 'tree_based', 'features_selected': 0}
        
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = SelectFromModel(estimator, threshold='mean')
        selector.fit(numeric_X, y)
        
        selected_features = numeric_X.columns[selector.get_support()].tolist()
        
        # Get feature importances
        feature_importances = dict(zip(
            numeric_X.columns,
            selector.estimator_.feature_importances_
        ))
        
        self.selectors['tree_based'] = selector
        
        report = {
            'method': 'tree_based',
            'features_selected': len(selected_features),
            'feature_importances': feature_importances,
            'threshold': 'mean'
        }
        
        return selected_features, report
    
    def _combine_selection_results(self, selection_results: Dict[str, List[str]], 
                                 all_features: pd.Index) -> List[str]:
        """Combine results from multiple selection methods."""
        if not selection_results:
            return all_features.tolist()
        
        # Count votes for each feature
        feature_votes = {}
        for method, features in selection_results.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features that got votes from at least half of the methods
        min_votes = max(1, len(selection_results) // 2)
        final_features = [feature for feature, votes in feature_votes.items() if votes >= min_votes]
        
        # Ensure we have at least some features
        if not final_features:
            # Fall back to features from the first method
            final_features = list(selection_results.values())[0] if selection_results else all_features.tolist()
        
        return final_features


class FeatureTransformer:
    """
    Handles feature transformations and dimensionality reduction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformers = {}
    
    def apply_transformations(self, X: pd.DataFrame, transformations: List[str], 
                            fit_transformers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply feature transformations.
        
        Args:
            X: Feature matrix
            transformations: List of transformations to apply
            fit_transformers: Whether to fit transformers
            
        Returns:
            Tuple of (transformed_features, transformation_report)
        """
        transformed_X = X.copy()
        transformation_report = {
            'original_features': len(X.columns),
            'transformations_applied': [],
            'transformation_details': {}
        }
        
        for transformation in transformations:
            if transformation == 'pca':
                transformed_X, report = self._apply_pca(transformed_X, fit_transformers)
                transformation_report['transformations_applied'].append('pca')
                transformation_report['transformation_details']['pca'] = report
                
            elif transformation == 'tsne':
                transformed_X, report = self._apply_tsne(transformed_X, fit_transformers)
                transformation_report['transformations_applied'].append('tsne')
                transformation_report['transformation_details']['tsne'] = report
        
        transformation_report['final_features'] = len(transformed_X.columns)
        
        return transformed_X, transformation_report
    
    def _apply_pca(self, X: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Principal Component Analysis."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            return X, {'components': 0, 'explained_variance_ratio': []}
        
        # Determine number of components (keep 95% variance or max 50 components)
        n_components = min(50, len(numeric_X.columns), len(X) - 1)
        
        if fit:
            pca = PCA(n_components=n_components, random_state=42)
            pca_features = pca.fit_transform(numeric_X)
            self.transformers['pca'] = pca
        else:
            if 'pca' in self.transformers:
                pca_features = self.transformers['pca'].transform(numeric_X)
                pca = self.transformers['pca']
            else:
                return X, {'components': 0, 'explained_variance_ratio': []}
        
        # Create PCA feature DataFrame
        pca_columns = [f'PCA_{i+1}' for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=X.index)
        
        # Keep non-numeric columns and add PCA features
        non_numeric_X = X.select_dtypes(exclude=[np.number])
        if not non_numeric_X.empty:
            result_X = pd.concat([non_numeric_X, pca_df], axis=1)
        else:
            result_X = pca_df
        
        # Find number of components needed for 95% variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        
        report = {
            'total_components': len(pca_columns),
            'components_for_95_variance': int(n_components_95),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': cumsum_variance.tolist()
        }
        
        return result_X, report
    
    def _apply_tsne(self, X: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply t-SNE for dimensionality reduction."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty or len(numeric_X) < 4:  # t-SNE needs at least 4 samples
            return X, {'components': 0}
        
        # t-SNE is computationally expensive, so limit data size
        if len(numeric_X) > 1000:
            sample_indices = np.random.choice(len(numeric_X), 1000, replace=False)
            numeric_sample = numeric_X.iloc[sample_indices]
        else:
            numeric_sample = numeric_X
            sample_indices = numeric_X.index
        
        # Apply t-SNE with 2 components for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(numeric_sample)//4))
        tsne_features = tsne.fit_transform(numeric_sample)
        
        # Create t-SNE feature DataFrame
        tsne_df = pd.DataFrame(
            tsne_features, 
            columns=['tSNE_1', 'tSNE_2'],
            index=sample_indices
        )
        
        # For full dataset, we'd need to use an out-of-sample extension
        # For now, just return the sample
        non_numeric_X = X.select_dtypes(exclude=[np.number])
        if not non_numeric_X.empty:
            result_X = pd.concat([non_numeric_X.loc[sample_indices], tsne_df], axis=1)
        else:
            result_X = tsne_df
        
        report = {
            'components': 2,
            'sample_size': len(numeric_sample),
            'original_features': len(numeric_X.columns)
        }
        
        return result_X, report
    
    def save_transformers(self, path: str):
        """Save fitted transformers to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformers, path)
        self.logger.info(f"Transformers saved to {path}")
    
    def load_transformers(self, path: str):
        """Load transformers from disk."""
        if Path(path).exists():
            self.transformers = joblib.load(path)
            self.logger.info(f"Transformers loaded from {path}")
        else:
            self.logger.warning(f"Transformer file not found: {path}")