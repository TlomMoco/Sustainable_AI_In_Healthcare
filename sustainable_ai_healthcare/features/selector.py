"""
Advanced feature selection algorithms for healthcare data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score
import warnings

# Advanced feature selection methods
try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except ImportError:
    HAS_BORUTA = False

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection toolkit for healthcare data with multiple selection strategies:
    - Statistical methods (univariate tests)
    - Model-based selection (importance, coefficients)
    - Wrapper methods (RFE, forward/backward selection)
    - Advanced methods (Boruta, stability selection)
    """
    
    def __init__(self, 
                 problem_type: str = 'auto',
                 selection_methods: List[str] = ['statistical', 'model_based', 'wrapper'],
                 scoring: Optional[str] = None):
        """
        Initialize feature selector.
        
        Args:
            problem_type: 'classification', 'regression', or 'auto'
            selection_methods: List of methods to use
            scoring: Scoring metric for evaluation
        """
        self.problem_type = problem_type
        self.selection_methods = selection_methods
        self.scoring = scoring
        self.selected_features = {}
        self.selection_scores = {}
        self.selectors = {}
        
    def _detect_problem_type(self, target: pd.Series) -> str:
        """Auto-detect problem type based on target variable."""
        if self.problem_type != 'auto':
            return self.problem_type
        
        if target.nunique() <= 10 and target.dtype in ['int64', 'bool', 'object', 'category']:
            return 'classification'
        else:
            return 'regression'
    
    def statistical_selection(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            method: str = 'f_test',
                            k: Union[int, float] = 10) -> Dict:
        """
        Statistical feature selection using univariate tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Statistical test method
            k: Number or percentage of features to select
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Performing statistical feature selection using {method}...")
        
        problem_type = self._detect_problem_type(y)
        
        # Choose appropriate statistical test
        if method == 'f_test':
            if problem_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        elif method == 'chi2':
            if problem_type == 'classification':
                # Ensure non-negative values for chi2
                X_chi2 = X.copy()
                X_chi2[X_chi2 < 0] = 0
                score_func = chi2
                X = X_chi2
            else:
                raise ValueError("Chi2 test is only suitable for classification problems")
        elif method == 'mutual_info':
            if problem_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        else:
            raise ValueError(f"Unsupported statistical method: {method}")
        
        # Select features
        if isinstance(k, float) and 0 < k < 1:
            # Select percentage
            selector = SelectPercentile(score_func=score_func, percentile=k*100)
        else:
            # Select k best
            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = dict(zip(X.columns, selector.scores_))
        
        results = {
            'method': f'statistical_{method}',
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': feature_scores,
            'selector': selector
        }
        
        self.selected_features[f'statistical_{method}'] = selected_features
        self.selection_scores[f'statistical_{method}'] = feature_scores
        self.selectors[f'statistical_{method}'] = selector
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return results
    
    def model_based_selection(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            model_type: str = 'random_forest',
                            threshold: Union[str, float] = 'median') -> Dict:
        """
        Model-based feature selection using feature importance or coefficients.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to use
            threshold: Importance threshold
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Performing model-based feature selection using {model_type}...")
        
        problem_type = self._detect_problem_type(y)
        
        # Choose appropriate model
        if model_type == 'random_forest':
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lasso':
            if problem_type == 'classification':
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(random_state=42)
        elif model_type == 'ridge':
            if problem_type == 'classification':
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(penalty='l2', random_state=42)
            else:
                model = RidgeCV()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit model and select features
        selector = SelectFromModel(model, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature importance/coefficients
        model.fit(X, y)
        if hasattr(model, 'feature_importances_'):
            feature_scores = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            feature_scores = dict(zip(X.columns, np.abs(coef)))
        else:
            feature_scores = {}
        
        results = {
            'method': f'model_based_{model_type}',
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': feature_scores,
            'selector': selector,
            'model': model
        }
        
        self.selected_features[f'model_based_{model_type}'] = selected_features
        self.selection_scores[f'model_based_{model_type}'] = feature_scores
        self.selectors[f'model_based_{model_type}'] = selector
        
        logger.info(f"Selected {len(selected_features)} features using {model_type}")
        return results
    
    def wrapper_selection(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         method: str = 'rfe',
                         n_features: Optional[int] = None,
                         cv: int = 5) -> Dict:
        """
        Wrapper-based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable  
            method: Wrapper method ('rfe', 'rfecv')
            n_features: Number of features to select (for RFE)
            cv: Cross-validation folds
            
        Returns:
            Dictionary with selection results
        """
        logger.info(f"Performing wrapper feature selection using {method}...")
        
        problem_type = self._detect_problem_type(y)
        
        # Choose estimator
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = self.scoring or 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = self.scoring or 'neg_mean_squared_error'
        
        # Apply wrapper method
        if method == 'rfe':
            n_features = n_features or max(1, X.shape[1] // 2)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        elif method == 'rfecv':
            selector = RFECV(estimator=estimator, cv=cv, scoring=scoring)
        else:
            raise ValueError(f"Unsupported wrapper method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature rankings
        feature_rankings = dict(zip(X.columns, selector.ranking_))
        feature_scores = {feat: 1.0 / rank for feat, rank in feature_rankings.items()}
        
        results = {
            'method': f'wrapper_{method}',
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'scores': feature_scores,
            'rankings': feature_rankings,
            'selector': selector
        }
        
        if hasattr(selector, 'grid_scores_'):
            results['cv_scores'] = selector.grid_scores_
        
        self.selected_features[f'wrapper_{method}'] = selected_features
        self.selection_scores[f'wrapper_{method}'] = feature_scores
        self.selectors[f'wrapper_{method}'] = selector
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return results
    
    def boruta_selection(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        max_iter: int = 100,
                        random_state: int = 42) -> Dict:
        """
        Boruta feature selection algorithm.
        
        Args:
            X: Feature matrix
            y: Target variable
            max_iter: Maximum iterations
            random_state: Random state
            
        Returns:
            Dictionary with selection results
        """
        if not HAS_BORUTA:
            raise ImportError("Boruta is not available. Install with: pip install boruta")
        
        logger.info("Performing Boruta feature selection...")
        
        problem_type = self._detect_problem_type(y)
        
        # Choose estimator
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # Run Boruta
        selector = BorutaPy(estimator, n_estimators='auto', max_iter=max_iter, random_state=random_state)
        selector.fit(X.values, y.values)
        
        # Get results
        selected_features = X.columns[selector.support_].tolist()
        tentative_features = X.columns[selector.support_weak_].tolist()
        
        feature_rankings = dict(zip(X.columns, selector.ranking_))
        feature_scores = {feat: 1.0 / rank for feat, rank in feature_rankings.items()}
        
        results = {
            'method': 'boruta',
            'selected_features': selected_features,
            'tentative_features': tentative_features,
            'n_selected': len(selected_features),
            'scores': feature_scores,
            'rankings': feature_rankings,
            'selector': selector
        }
        
        self.selected_features['boruta'] = selected_features
        self.selection_scores['boruta'] = feature_scores
        self.selectors['boruta'] = selector
        
        logger.info(f"Boruta selected {len(selected_features)} features ({len(tentative_features)} tentative)")
        return results
    
    def stability_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           method: str = 'lasso',
                           n_bootstrap: int = 100,
                           threshold: float = 0.6) -> Dict:
        """
        Stability selection using bootstrap sampling.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Base selection method
            n_bootstrap: Number of bootstrap samples
            threshold: Stability threshold
            
        Returns:
            Dictionary with selection results
        """
        logger.info("Performing stability selection...")
        
        problem_type = self._detect_problem_type(y)
        n_samples, n_features = X.shape
        
        # Track feature selection frequency
        selection_freq = np.zeros(n_features)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Apply base selection method
            if method == 'lasso':
                if problem_type == 'classification':
                    from sklearn.linear_model import LogisticRegressionCV
                    model = LogisticRegressionCV(penalty='l1', solver='liblinear', random_state=42)
                else:
                    model = LassoCV(random_state=42)
                
                selector = SelectFromModel(model, threshold='median')
                selector.fit(X_boot, y_boot)
                selected_mask = selector.get_support()
                
            elif method == 'random_forest':
                if problem_type == 'classification':
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                selector = SelectFromModel(model, threshold='median')
                selector.fit(X_boot, y_boot)
                selected_mask = selector.get_support()
            
            else:
                raise ValueError(f"Unsupported method for stability selection: {method}")
            
            # Update selection frequency
            selection_freq += selected_mask
        
        # Calculate stability scores
        stability_scores = selection_freq / n_bootstrap
        feature_scores = dict(zip(X.columns, stability_scores))
        
        # Select stable features
        stable_features = X.columns[stability_scores >= threshold].tolist()
        
        results = {
            'method': f'stability_{method}',
            'selected_features': stable_features,
            'n_selected': len(stable_features),
            'scores': feature_scores,
            'stability_threshold': threshold,
            'selection_frequencies': dict(zip(X.columns, selection_freq))
        }
        
        self.selected_features[f'stability_{method}'] = stable_features
        self.selection_scores[f'stability_{method}'] = feature_scores
        
        logger.info(f"Stability selection identified {len(stable_features)} stable features")
        return results
    
    def ensemble_selection(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          methods: Optional[List[str]] = None,
                          voting_threshold: float = 0.5) -> Dict:
        """
        Ensemble feature selection combining multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of methods to combine
            voting_threshold: Minimum fraction of methods that must select a feature
            
        Returns:
            Dictionary with ensemble selection results
        """
        logger.info("Performing ensemble feature selection...")
        
        if methods is None:
            methods = self.selection_methods
        
        # Run individual selection methods
        method_results = {}
        
        for method in methods:
            try:
                if method == 'statistical':
                    result = self.statistical_selection(X, y, method='f_test')
                    method_results['statistical'] = result
                    
                elif method == 'model_based':
                    result = self.model_based_selection(X, y, model_type='random_forest')
                    method_results['model_based'] = result
                    
                elif method == 'wrapper':
                    result = self.wrapper_selection(X, y, method='rfecv')
                    method_results['wrapper'] = result
                    
                elif method == 'boruta' and HAS_BORUTA:
                    result = self.boruta_selection(X, y)
                    method_results['boruta'] = result
                    
                elif method == 'stability':
                    result = self.stability_selection(X, y)
                    method_results['stability'] = result
                    
            except Exception as e:
                logger.warning(f"Error running {method}: {str(e)}")
                continue
        
        if not method_results:
            raise ValueError("No selection methods succeeded")
        
        # Combine results using voting
        feature_votes = {}
        for feature in X.columns:
            votes = 0
            for method_result in method_results.values():
                if feature in method_result['selected_features']:
                    votes += 1
            feature_votes[feature] = votes / len(method_results)
        
        # Select features based on voting threshold
        ensemble_features = [feat for feat, vote in feature_votes.items() 
                           if vote >= voting_threshold]
        
        # Calculate ensemble scores (average of normalized scores)
        ensemble_scores = {}
        for feature in X.columns:
            scores = []
            for method_result in method_results.values():
                if feature in method_result['scores']:
                    # Normalize score to [0, 1]
                    method_scores = list(method_result['scores'].values())
                    min_score, max_score = min(method_scores), max(method_scores)
                    if max_score > min_score:
                        normalized_score = (method_result['scores'][feature] - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0
                    scores.append(normalized_score)
            
            ensemble_scores[feature] = np.mean(scores) if scores else 0.0
        
        results = {
            'method': 'ensemble',
            'selected_features': ensemble_features,
            'n_selected': len(ensemble_features),
            'scores': ensemble_scores,
            'voting_scores': feature_votes,
            'voting_threshold': voting_threshold,
            'individual_methods': method_results
        }
        
        self.selected_features['ensemble'] = ensemble_features
        self.selection_scores['ensemble'] = ensemble_scores
        
        logger.info(f"Ensemble selection identified {len(ensemble_features)} features")
        return results
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       method: str = 'ensemble') -> pd.DataFrame:
        """
        Main feature selection interface.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method to use
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Starting feature selection using {method} method...")
        
        if method == 'ensemble':
            result = self.ensemble_selection(X, y)
        elif method == 'statistical':
            result = self.statistical_selection(X, y)
        elif method == 'model_based':
            result = self.model_based_selection(X, y)
        elif method == 'wrapper':
            result = self.wrapper_selection(X, y)
        elif method == 'boruta':
            result = self.boruta_selection(X, y)
        elif method == 'stability':
            result = self.stability_selection(X, y)
        else:
            raise ValueError(f"Unsupported selection method: {method}")
        
        selected_features = result['selected_features']
        return X[selected_features]
    
    def get_feature_rankings(self) -> pd.DataFrame:
        """
        Get comprehensive feature rankings from all applied methods.
        
        Returns:
            DataFrame with feature rankings
        """
        if not self.selection_scores:
            raise ValueError("No feature selection has been performed")
        
        # Create rankings DataFrame
        rankings_data = {}
        
        for method, scores in self.selection_scores.items():
            # Convert scores to ranks (higher score = better rank)
            score_series = pd.Series(scores)
            rankings_data[f'{method}_score'] = score_series
            rankings_data[f'{method}_rank'] = score_series.rank(ascending=False)
            rankings_data[f'{method}_selected'] = score_series.index.isin(
                self.selected_features.get(method, [])
            )
        
        rankings_df = pd.DataFrame(rankings_data)
        
        # Add average rank
        rank_columns = [col for col in rankings_df.columns if col.endswith('_rank')]
        rankings_df['avg_rank'] = rankings_df[rank_columns].mean(axis=1)
        
        # Add selection frequency
        selected_columns = [col for col in rankings_df.columns if col.endswith('_selected')]
        rankings_df['selection_frequency'] = rankings_df[selected_columns].sum(axis=1)
        
        return rankings_df.sort_values('avg_rank')
    
    def plot_feature_selection_results(self, 
                                     save_plots: bool = False,
                                     plot_dir: str = './plots'):
        """
        Create visualizations of feature selection results.
        
        Args:
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
        """
        if not self.selection_scores:
            logger.warning("No feature selection results to plot")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Feature rankings comparison
        rankings_df = self.get_feature_rankings()
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Feature selection frequency
        plt.subplot(2, 2, 1)
        freq_data = rankings_df['selection_frequency'].value_counts().sort_index()
        plt.bar(freq_data.index, freq_data.values)
        plt.xlabel('Number of Methods Selecting Feature')
        plt.ylabel('Number of Features')
        plt.title('Feature Selection Frequency')
        
        # Plot 2: Score distributions
        plt.subplot(2, 2, 2)
        score_columns = [col for col in rankings_df.columns if col.endswith('_score')]
        for col in score_columns[:3]:  # Limit to first 3 methods
            plt.hist(rankings_df[col].dropna(), alpha=0.5, label=col.replace('_score', ''))
        plt.xlabel('Feature Score')
        plt.ylabel('Count')
        plt.title('Feature Score Distributions')
        plt.legend()
        
        # Plot 3: Top features by average rank
        plt.subplot(2, 2, 3)
        top_features = rankings_df.head(20)
        plt.barh(range(len(top_features)), top_features['avg_rank'], 
                color=plt.cm.viridis(top_features['selection_frequency'] / top_features['selection_frequency'].max()))
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Average Rank')
        plt.title('Top 20 Features by Average Rank')
        
        # Plot 4: Method agreement heatmap
        plt.subplot(2, 2, 4)
        if len(self.selected_features) > 1:
            # Create binary matrix of feature selections
            all_features = set()
            for features in self.selected_features.values():
                all_features.update(features)
            
            agreement_matrix = []
            method_names = list(self.selected_features.keys())
            
            for method1 in method_names:
                row = []
                for method2 in method_names:
                    set1 = set(self.selected_features[method1])
                    set2 = set(self.selected_features[method2])
                    agreement = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
                    row.append(agreement)
                agreement_matrix.append(row)
            
            sns.heatmap(agreement_matrix, annot=True, xticklabels=method_names, 
                       yticklabels=method_names, cmap='Blues')
            plt.title('Method Agreement (Jaccard Index)')
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/feature_selection_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()