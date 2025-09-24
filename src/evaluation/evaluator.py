"""
Model Evaluation Module

This module provides comprehensive model evaluation capabilities including
performance metrics, visualization, and comparison tools for healthcare AI models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import learning_curve, validation_curve
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Main evaluation orchestrator that coordinates all evaluation components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator(config)
        self.performance_reporter = PerformanceReporter(config)
        
    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series,
                       task_type: str = 'classification', 
                       X_train: Optional[pd.DataFrame] = None,
                       y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            X_train: Training features (for learning curves)
            y_train: Training targets (for learning curves)
            
        Returns:
            Comprehensive evaluation report
        """
        
        self.logger.info(f"Starting comprehensive model evaluation for {len(models)} models")
        
        evaluation_report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'test_set_info': {
                'samples': len(X_test),
                'features': len(X_test.columns),
                'target_distribution': self._analyze_target_distribution(y_test, task_type)
            },
            'model_performances': {},
            'model_comparison': {},
            'recommendations': []
        }
        
        # Evaluate each model
        model_performances = {}
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}")
            
            try:
                performance = self._evaluate_single_model(
                    model, X_test, y_test, task_type, model_name,
                    X_train, y_train
                )
                model_performances[model_name] = performance
                evaluation_report['model_performances'][model_name] = performance
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                evaluation_report['model_performances'][model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Compare models
        if model_performances:
            evaluation_report['model_comparison'] = self._compare_models(
                model_performances, task_type
            )
        
        # Generate recommendations
        evaluation_report['recommendations'] = self._generate_recommendations(
            evaluation_report, task_type
        )
        
        # Generate visualizations if configured
        if self.config.get('generate_reports', True):
            evaluation_report['visualizations'] = self._generate_evaluation_visualizations(
                models, X_test, y_test, task_type
            )
        
        self.logger.info("Model evaluation completed")
        return evaluation_report
    
    def _evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                              task_type: str, model_name: str,
                              X_train: Optional[pd.DataFrame] = None,
                              y_train: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        
        performance = {
            'model_name': model_name,
            'status': 'success'
        }
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities for classification
        y_pred_proba = None
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                self.logger.warning(f"Could not get probabilities for {model_name}: {e}")
        
        # Calculate metrics
        performance['metrics'] = self.metrics_calculator.calculate_metrics(
            y_test, y_pred, y_pred_proba, task_type
        )
        
        # Additional analysis
        performance['prediction_analysis'] = self._analyze_predictions(
            y_test, y_pred, task_type
        )
        
        # Learning curves if training data is available
        if X_train is not None and y_train is not None:
            try:
                performance['learning_curves'] = self._calculate_learning_curves(
                    model, X_train, y_train, task_type
                )
            except Exception as e:
                self.logger.warning(f"Could not calculate learning curves for {model_name}: {e}")
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            performance['feature_importance'] = dict(zip(
                X_test.columns,
                model.feature_importances_
            ))
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) == 1:
                performance['feature_importance'] = dict(zip(
                    X_test.columns,
                    np.abs(model.coef_)
                ))
            else:
                # Multi-class case - use mean of absolute coefficients
                performance['feature_importance'] = dict(zip(
                    X_test.columns,
                    np.mean(np.abs(model.coef_), axis=0)
                ))
        
        return performance
    
    def _analyze_target_distribution(self, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """Analyze the distribution of target variable."""
        
        if task_type == 'classification':
            value_counts = y.value_counts()
            return {
                'type': 'classification',
                'classes': len(value_counts),
                'class_distribution': value_counts.to_dict(),
                'class_balance': float(value_counts.min() / value_counts.max()),
                'most_frequent_class': str(value_counts.index[0])
            }
        else:
            return {
                'type': 'regression',
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'median': float(y.median()),
                'quartiles': {
                    'q1': float(y.quantile(0.25)),
                    'q3': float(y.quantile(0.75))
                }
            }
    
    def _analyze_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                           task_type: str) -> Dict[str, Any]:
        """Analyze model predictions for insights."""
        
        analysis = {}
        
        if task_type == 'classification':
            # Prediction distribution
            pred_counts = pd.Series(y_pred).value_counts()
            analysis['prediction_distribution'] = pred_counts.to_dict()
            
            # Prediction accuracy by class
            class_accuracies = {}
            for class_label in np.unique(y_true):
                mask = y_true == class_label
                if mask.sum() > 0:
                    class_accuracy = (y_pred[mask] == class_label).mean()
                    class_accuracies[str(class_label)] = float(class_accuracy)
            
            analysis['class_accuracies'] = class_accuracies
            
        else:
            # Prediction vs actual analysis
            residuals = y_true - y_pred
            analysis['residuals'] = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'mean_absolute': float(np.abs(residuals).mean()),
                'median': float(np.median(residuals))
            }
            
            # Prediction range
            analysis['prediction_range'] = {
                'min': float(y_pred.min()),
                'max': float(y_pred.max()),
                'mean': float(y_pred.mean()),
                'std': float(y_pred.std())
            }
        
        return analysis
    
    def _calculate_learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                 task_type: str) -> Dict[str, Any]:
        """Calculate learning curves for the model."""
        
        # Define training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Calculate learning curves
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
    
    def _compare_models(self, model_performances: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Compare multiple models and rank them."""
        
        comparison = {
            'best_model': None,
            'ranking': [],
            'metric_comparison': {}
        }
        
        # Primary metric for ranking
        primary_metric = 'accuracy' if task_type == 'classification' else 'r2'
        
        # Extract primary metric values for each model
        model_scores = {}
        for model_name, performance in model_performances.items():
            if performance.get('status') == 'success':
                metrics = performance.get('metrics', {})
                score = metrics.get(primary_metric, 0)
                model_scores[model_name] = score
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['ranking'] = [{'model': model, 'score': score} for model, score in ranked_models]
        comparison['best_model'] = ranked_models[0][0] if ranked_models else None
        
        # Detailed metric comparison
        metrics_to_compare = self._get_comparison_metrics(task_type)
        
        for metric in metrics_to_compare:
            comparison['metric_comparison'][metric] = {}
            for model_name, performance in model_performances.items():
                if performance.get('status') == 'success':
                    metrics = performance.get('metrics', {})
                    comparison['metric_comparison'][metric][model_name] = metrics.get(metric, 0)
        
        return comparison
    
    def _get_comparison_metrics(self, task_type: str) -> List[str]:
        """Get metrics for model comparison."""
        if task_type == 'classification':
            return ['accuracy', 'precision', 'recall', 'f1', 'auc']
        else:
            return ['r2', 'mse', 'mae', 'rmse']
    
    def _generate_recommendations(self, evaluation_report: Dict[str, Any], 
                                task_type: str) -> List[str]:
        """Generate recommendations based on evaluation results."""
        
        recommendations = []
        
        # Best model recommendation
        best_model = evaluation_report['model_comparison'].get('best_model')
        if best_model:
            recommendations.append(f"Best performing model: {best_model}")
        
        # Performance analysis recommendations
        model_performances = evaluation_report['model_performances']
        
        if task_type == 'classification':
            # Check for class imbalance issues
            target_dist = evaluation_report['test_set_info']['target_distribution']
            if target_dist.get('class_balance', 1) < 0.5:
                recommendations.append(
                    "Class imbalance detected - consider using balanced sampling or class weights"
                )
            
            # Check for low recall models
            low_recall_models = []
            for model_name, perf in model_performances.items():
                if perf.get('status') == 'success':
                    recall = perf.get('metrics', {}).get('recall', 0)
                    if recall < 0.7:
                        low_recall_models.append(model_name)
            
            if low_recall_models:
                recommendations.append(
                    f"Low recall detected in models: {', '.join(low_recall_models)} - "
                    "consider adjusting decision threshold or using different algorithms"
                )
        
        else:  # regression
            # Check for high error models
            high_error_models = []
            for model_name, perf in model_performances.items():
                if perf.get('status') == 'success':
                    r2 = perf.get('metrics', {}).get('r2', 0)
                    if r2 < 0.5:
                        high_error_models.append(model_name)
            
            if high_error_models:
                recommendations.append(
                    f"Low R² score in models: {', '.join(high_error_models)} - "
                    "consider feature engineering or different algorithms"
                )
        
        # General recommendations
        if len(model_performances) > 1:
            recommendations.append(
                "Consider ensemble methods to combine strengths of different models"
            )
        
        return recommendations
    
    def _generate_evaluation_visualizations(self, models: Dict[str, Any], 
                                          X_test: pd.DataFrame, y_test: pd.Series,
                                          task_type: str) -> Dict[str, str]:
        """Generate evaluation visualizations."""
        
        viz_paths = {}
        
        # Model comparison chart
        viz_paths['model_comparison'] = self.performance_reporter.plot_model_comparison(
            models, X_test, y_test, task_type
        )
        
        # Individual model visualizations
        for model_name, model in models.items():
            if task_type == 'classification':
                # Confusion matrix
                viz_paths[f'{model_name}_confusion_matrix'] = \
                    self.performance_reporter.plot_confusion_matrix(
                        model, X_test, y_test, model_name
                    )
                
                # ROC curve
                viz_paths[f'{model_name}_roc_curve'] = \
                    self.performance_reporter.plot_roc_curve(
                        model, X_test, y_test, model_name
                    )
            
            else:
                # Prediction vs actual
                viz_paths[f'{model_name}_predictions'] = \
                    self.performance_reporter.plot_prediction_vs_actual(
                        model, X_test, y_test, model_name
                    )
                
                # Residuals plot
                viz_paths[f'{model_name}_residuals'] = \
                    self.performance_reporter.plot_residuals(
                        model, X_test, y_test, model_name
                    )
        
        return viz_paths


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for different task types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.configured_metrics = config.get('metrics', [])
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None,
                         task_type: str = 'classification') -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Prediction probabilities (for classification)
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary of calculated metrics
        """
        
        metrics = {}
        
        if task_type == 'classification':
            metrics.update(self._calculate_classification_metrics(
                y_true, y_pred, y_pred_proba
            ))
        else:
            metrics.update(self._calculate_regression_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        if 'accuracy' in self.configured_metrics:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        if 'precision' in self.configured_metrics:
            metrics['precision'] = float(precision_score(
                y_true, y_pred, average=average, zero_division=0
            ))
        
        if 'recall' in self.configured_metrics:
            metrics['recall'] = float(recall_score(
                y_true, y_pred, average=average, zero_division=0
            ))
        
        if 'f1' in self.configured_metrics:
            metrics['f1'] = float(f1_score(
                y_true, y_pred, average=average, zero_division=0
            ))
        
        # Specificity and Sensitivity for binary classification
        if n_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            if 'specificity' in self.configured_metrics:
                metrics['specificity'] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
            
            if 'sensitivity' in self.configured_metrics:
                metrics['sensitivity'] = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
        # AUC metrics
        if y_pred_proba is not None:
            try:
                if 'auc' in self.configured_metrics:
                    if n_classes == 2:
                        metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                    else:
                        metrics['auc'] = float(roc_auc_score(
                            y_true, y_pred_proba, multi_class='ovr'
                        ))
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        
        metrics = {}
        
        if 'r2' in self.configured_metrics or not self.configured_metrics:
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        if 'mse' in self.configured_metrics:
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        
        if 'mae' in self.configured_metrics:
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        if 'rmse' in self.configured_metrics:
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Additional metrics
        metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
        
        # Mean Absolute Percentage Error
        if not np.any(y_true == 0):  # Avoid division by zero
            metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        
        return metrics


class PerformanceReporter:
    """
    Generates performance reports and visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("reports/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(self, models: Dict[str, Any], X_test: pd.DataFrame,
                            y_test: pd.Series, task_type: str) -> str:
        """Plot comparison of multiple models."""
        
        if task_type == 'classification':
            primary_metric = 'accuracy'
        else:
            primary_metric = 'r2'
        
        # Calculate scores for all models
        model_scores = {}
        for name, model in models.items():
            try:
                if task_type == 'classification':
                    score = accuracy_score(y_test, model.predict(X_test))
                else:
                    score = r2_score(y_test, model.predict(X_test))
                model_scores[name] = score
            except Exception as e:
                self.logger.warning(f"Could not calculate score for {name}: {e}")
        
        if not model_scores:
            return ""
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        names = list(model_scores.keys())
        scores = list(model_scores.values())
        
        bars = plt.bar(names, scores)
        plt.title(f'Model Comparison - {primary_metric.title()}')
        plt.xlabel('Models')
        plt.ylabel(primary_metric.title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_confusion_matrix(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str) -> str:
        """Plot confusion matrix for classification model."""
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        output_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_roc_curve(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> str:
        """Plot ROC curve for binary classification."""
        
        if not hasattr(model, 'predict_proba'):
            return ""
        
        try:
            y_pred_proba = model.predict_proba(X_test)
            
            # Handle binary classification
            if len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                output_path = self.output_dir / f"{model_name}_roc_curve.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(output_path)
        
        except Exception as e:
            self.logger.warning(f"Could not plot ROC curve for {model_name}: {e}")
        
        return ""
    
    def plot_prediction_vs_actual(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                 model_name: str) -> str:
        """Plot predictions vs actual values for regression."""
        
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_path = self.output_dir / f"{model_name}_predictions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_residuals(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> str:
        """Plot residuals for regression model."""
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Residuals vs Predicted - {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residuals Distribution - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{model_name}_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)