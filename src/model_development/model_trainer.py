"""
Model Development Module

This module provides comprehensive model development capabilities for healthcare AI,
including various ML/DL algorithms, hyperparameter optimization, and training frameworks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.metrics import make_scorer
import logging
import joblib
from pathlib import Path
import json
import time
from datetime import datetime

# Neural Network imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, neural network models will be disabled")


class ModelTrainer:
    """
    Main model training orchestrator that handles multiple algorithms and training strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.training_history = {}
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.model_validator = ModelValidator(config)
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                    task_type: str = 'classification') -> Dict[str, Any]:
        """
        Train multiple models with the given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            task_type: 'classification' or 'regression'
            
        Returns:
            Training report with model performance and details
        """
        self.logger.info(f"Starting model training for {task_type} task")
        
        training_report = {
            'task_type': task_type,
            'training_start': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': len(X_train),
                'features': len(X_train.columns),
                'validation_samples': len(X_val) if X_val is not None else 0
            },
            'models_trained': {},
            'best_model': None,
            'training_errors': []
        }
        
        # Get algorithms from config
        algorithms = self.config.get('algorithms', ['random_forest', 'xgboost'])
        
        # Train each algorithm
        for algorithm in algorithms:
            self.logger.info(f"Training {algorithm}")
            
            try:
                start_time = time.time()
                
                # Train model
                model_result = self._train_single_model(
                    algorithm, X_train, y_train, X_val, y_val, task_type
                )
                
                training_time = time.time() - start_time
                model_result['training_time'] = training_time
                
                # Store model and results
                self.models[algorithm] = model_result['model']
                training_report['models_trained'][algorithm] = {
                    'status': 'success',
                    'training_time': training_time,
                    'cross_val_score': model_result.get('cv_score'),
                    'best_params': model_result.get('best_params'),
                    'validation_score': model_result.get('validation_score')
                }
                
                self.logger.info(f"{algorithm} trained successfully in {training_time:.2f} seconds")
                
            except Exception as e:
                self.logger.error(f"Failed to train {algorithm}: {str(e)}")
                training_report['training_errors'].append({
                    'algorithm': algorithm,
                    'error': str(e)
                })
                training_report['models_trained'][algorithm] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Identify best model based on cross-validation scores
        best_model_name = self._identify_best_model(training_report['models_trained'])
        training_report['best_model'] = best_model_name
        
        training_report['training_end'] = datetime.now().isoformat()
        
        self.logger.info(f"Model training completed. Best model: {best_model_name}")
        return training_report
    
    def _train_single_model(self, algorithm: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                           task_type: str) -> Dict[str, Any]:
        """Train a single model with the specified algorithm."""
        
        # Get base model
        base_model = self._get_base_model(algorithm, task_type)
        
        # Get hyperparameters for optimization
        param_grid = self._get_hyperparameter_grid(algorithm, task_type)
        
        # Optimize hyperparameters if grid is provided
        if param_grid:
            optimized_model, best_params = self.hyperparameter_optimizer.optimize(
                base_model, X_train, y_train, param_grid, task_type
            )
        else:
            optimized_model = base_model
            best_params = {}
        
        # Train final model
        if algorithm == 'neural_network' and TF_AVAILABLE:
            # Special handling for neural networks
            model_result = self._train_neural_network(
                X_train, y_train, X_val, y_val, task_type, best_params
            )
        else:
            # Train sklearn-compatible model
            optimized_model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_score = self.model_validator.cross_validate(
                optimized_model, X_train, y_train, task_type
            )
            
            # Validation score if validation data is provided
            validation_score = None
            if X_val is not None and y_val is not None:
                validation_score = self.model_validator.validate_model(
                    optimized_model, X_val, y_val, task_type
                )
            
            model_result = {
                'model': optimized_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'validation_score': validation_score
            }
        
        return model_result
    
    def _get_base_model(self, algorithm: str, task_type: str):
        """Get base model for the specified algorithm."""
        
        if task_type == 'classification':
            model_map = {
                'random_forest': RandomForestClassifier(random_state=42),
                'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'lightgbm': lgb.LGBMClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(random_state=42),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(random_state=42)
            }
        else:  # regression
            model_map = {
                'random_forest': RandomForestRegressor(random_state=42),
                'xgboost': xgb.XGBRegressor(random_state=42),
                'lightgbm': lgb.LGBMRegressor(random_state=42),
                'linear_regression': LinearRegression(),
                'svm': SVR(),
                'knn': KNeighborsRegressor(),
                'decision_tree': DecisionTreeRegressor(random_state=42)
            }
        
        if algorithm not in model_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model_map[algorithm]
    
    def _get_hyperparameter_grid(self, algorithm: str, task_type: str) -> Dict[str, List]:
        """Get hyperparameter grid for optimization."""
        
        # Get from config if available
        config_algorithms = self.config.get('algorithms', [])
        for algo_config in config_algorithms:
            if isinstance(algo_config, dict) and algo_config.get('name') == algorithm:
                return algo_config.get('hyperparameters', {})
        
        # Default hyperparameter grids
        if algorithm == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif algorithm == 'xgboost':
            return {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0]
            }
        elif algorithm == 'lightgbm':
            return {
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'n_estimators': [100, 200, 300],
                'min_child_samples': [20, 30, 50]
            }
        elif algorithm == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif algorithm == 'knn':
            return {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        
        return {}  # No hyperparameters to tune
    
    def _train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                             task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Train a neural network model."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network training")
        
        # Default parameters
        hidden_layers = params.get('hidden_layers', 128)
        dropout_rate = params.get('dropout_rate', 0.3)
        learning_rate = params.get('learning_rate', 0.001)
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        
        # Build model
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        model.add(layers.Dense(hidden_layers // 2, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if task_type == 'classification':
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(n_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val.values, y_val.values)
        
        # Train model
        history = model.fit(
            X_train.values, y_train.values,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5)
            ]
        )
        
        # Calculate scores
        train_score = model.evaluate(X_train.values, y_train.values, verbose=0)
        validation_score = None
        if validation_data:
            validation_score = model.evaluate(validation_data[0], validation_data[1], verbose=0)
        
        return {
            'model': model,
            'best_params': params,
            'cv_score': {'train_score': train_score},
            'validation_score': validation_score,
            'history': history.history
        }
    
    def _identify_best_model(self, models_trained: Dict[str, Any]) -> Optional[str]:
        """Identify the best model based on cross-validation scores."""
        
        best_model = None
        best_score = -np.inf
        
        for model_name, model_info in models_trained.items():
            if model_info['status'] != 'success':
                continue
            
            cv_score = model_info.get('cv_score')
            if cv_score is None:
                continue
            
            # Extract score (handle different formats)
            if isinstance(cv_score, dict):
                score = cv_score.get('mean', 0)
            elif isinstance(cv_score, (list, np.ndarray)):
                score = np.mean(cv_score)
            else:
                score = cv_score
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        models_path = Path(path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_path / f"{name}.pkl"
            
            # Special handling for neural networks
            if hasattr(model, 'save') and TF_AVAILABLE:
                # TensorFlow/Keras model
                model_path = models_path / f"{name}"
                model.save(model_path)
            else:
                # Scikit-learn compatible model
                joblib.dump(model, model_path)
            
            self.logger.info(f"Model {name} saved to {model_path}")
        
        # Save training history
        history_path = models_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)


class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization using various strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series, 
                param_grid: Dict[str, List], task_type: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters using grid search or random search.
        
        Args:
            model: Base model to optimize
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for optimization
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (optimized_model, best_parameters)
        """
        
        if not param_grid:
            return model, {}
        
        self.logger.info("Starting hyperparameter optimization")
        
        # Set up cross-validation
        cv_folds = self.config.get('cross_validation_folds', 5)
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Choose optimization strategy
        n_combinations = np.prod([len(values) for values in param_grid.values()])
        
        if n_combinations <= 50:
            # Use GridSearchCV for small parameter spaces
            optimizer = GridSearchCV(
                model, param_grid, cv=cv, n_jobs=-1, 
                scoring=self._get_scoring_metric(task_type)
            )
        else:
            # Use RandomizedSearchCV for large parameter spaces
            optimizer = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=cv, n_jobs=-1,
                scoring=self._get_scoring_metric(task_type), random_state=42
            )
        
        # Fit optimizer
        optimizer.fit(X, y)
        
        self.logger.info(f"Best parameters: {optimizer.best_params_}")
        self.logger.info(f"Best score: {optimizer.best_score_:.4f}")
        
        return optimizer.best_estimator_, optimizer.best_params_
    
    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric for the task type."""
        if task_type == 'classification':
            return 'accuracy'
        else:
            return 'r2'


class ModelValidator:
    """
    Handles model validation and performance assessment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            
        Returns:
            Cross-validation scores
        """
        
        cv_folds = self.config.get('cross_validation_folds', 5)
        stratified = self.config.get('stratified', True)
        
        # Set up cross-validation
        if task_type == 'classification' and stratified:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Get scoring metrics
        scoring_metrics = self._get_scoring_metrics(task_type)
        
        # Perform cross-validation
        cv_results = {}
        for metric_name, scorer in scoring_metrics.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            cv_results[f'{metric_name}_scores'] = scores.tolist()
            cv_results[f'{metric_name}_mean'] = float(scores.mean())
            cv_results[f'{metric_name}_std'] = float(scores.std())
        
        # Return primary metric for model comparison
        primary_metric = 'accuracy' if task_type == 'classification' else 'r2'
        cv_results['mean'] = cv_results.get(f'{primary_metric}_mean', 0)
        cv_results['std'] = cv_results.get(f'{primary_metric}_std', 0)
        
        return cv_results
    
    def validate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series, task_type: str) -> Dict[str, float]:
        """
        Validate model on holdout validation set.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            task_type: 'classification' or 'regression'
            
        Returns:
            Validation scores
        """
        
        # Get predictions
        if task_type == 'classification':
            y_pred = model.predict(X_val)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)
        else:
            y_pred = model.predict(X_val)
            y_pred_proba = None
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            r2_score, mean_squared_error, mean_absolute_error
        )
        
        validation_scores = {}
        
        if task_type == 'classification':
            validation_scores['accuracy'] = float(accuracy_score(y_val, y_pred))
            
            # Handle binary vs multiclass
            average = 'binary' if len(np.unique(y_val)) == 2 else 'weighted'
            
            validation_scores['precision'] = float(precision_score(y_val, y_pred, average=average, zero_division=0))
            validation_scores['recall'] = float(recall_score(y_val, y_pred, average=average, zero_division=0))
            validation_scores['f1'] = float(f1_score(y_val, y_pred, average=average, zero_division=0))
            
            # ROC AUC if probabilities are available
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_val)) == 2:
                        validation_scores['auc'] = float(roc_auc_score(y_val, y_pred_proba[:, 1]))
                    else:
                        validation_scores['auc'] = float(roc_auc_score(y_val, y_pred_proba, multi_class='ovr'))
                except Exception as e:
                    self.logger.warning(f"Could not calculate AUC: {e}")
        
        else:  # regression
            validation_scores['r2'] = float(r2_score(y_val, y_pred))
            validation_scores['mse'] = float(mean_squared_error(y_val, y_pred))
            validation_scores['mae'] = float(mean_absolute_error(y_val, y_pred))
            validation_scores['rmse'] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        
        return validation_scores
    
    def _get_scoring_metrics(self, task_type: str) -> Dict[str, str]:
        """Get scoring metrics for cross-validation."""
        if task_type == 'classification':
            return {
                'accuracy': 'accuracy',
                'precision': make_scorer(lambda y_true, y_pred: 
                    precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': make_scorer(lambda y_true, y_pred:
                    recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1': make_scorer(lambda y_true, y_pred:
                    f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        else:
            return {
                'r2': 'r2',
                'neg_mse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error'
            }