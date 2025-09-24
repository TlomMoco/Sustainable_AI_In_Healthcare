"""
Federated Learning Module

This module provides federated learning capabilities for healthcare AI,
including client-server architecture, privacy preservation, and secure aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import json
import time
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime

# Federated Learning imports
try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    logging.warning("Flower (flwr) not available, federated learning will use custom implementation")

# Privacy imports
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography not available, encryption features will be disabled")


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    model_type: str
    learning_rate: float = 0.01
    batch_size: int = 32
    local_epochs: int = 1
    privacy_budget: float = 1.0
    use_differential_privacy: bool = True


@dataclass
class FLRound:
    """Represents a federated learning round."""
    round_id: int
    timestamp: str
    participants: List[str]
    aggregated_metrics: Dict[str, float]
    convergence_metrics: Dict[str, float]


class FederatedClient(ABC):
    """
    Abstract base class for federated learning clients.
    """
    
    def __init__(self, client_config: ClientConfig, privacy_preserver: Optional['PrivacyPreserver'] = None):
        self.config = client_config
        self.client_id = client_config.client_id
        self.logger = logging.getLogger(f"FLClient-{self.client_id}")
        self.privacy_preserver = privacy_preserver
        self.model = None
        self.local_data = None
        
    @abstractmethod
    def load_data(self, data_path: str) -> None:
        """Load local training data."""
        pass
    
    @abstractmethod
    def train_local_model(self, global_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train model on local data."""
        pass
    
    @abstractmethod
    def evaluate_local_model(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate model on local data."""
        pass
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Extract model parameters."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Extract parameters based on model type
        if hasattr(self.model, 'coef_'):
            # Linear models
            params = {
                'coef_': self.model.coef_,
                'intercept_': getattr(self.model, 'intercept_', None)
            }
        elif hasattr(self.model, 'get_weights'):
            # Neural network models (Keras/TensorFlow)
            params = {'weights': self.model.get_weights()}
        else:
            # Tree-based models or others
            params = {'model_state': pickle.dumps(self.model)}
        
        return params
    
    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set model parameters."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            if 'coef_' in parameters:
                # Linear models
                self.model.coef_ = parameters['coef_']
                if 'intercept_' in parameters and parameters['intercept_'] is not None:
                    self.model.intercept_ = parameters['intercept_']
            elif 'weights' in parameters:
                # Neural network models
                self.model.set_weights(parameters['weights'])
            elif 'model_state' in parameters:
                # Full model state
                self.model = pickle.loads(parameters['model_state'])
        except Exception as e:
            self.logger.error(f"Failed to set model parameters: {e}")


class SKLearnFLClient(FederatedClient):
    """
    Federated learning client for scikit-learn models.
    """
    
    def __init__(self, client_config: ClientConfig, model_class, privacy_preserver: Optional['PrivacyPreserver'] = None):
        super().__init__(client_config, privacy_preserver)
        self.model_class = model_class
        self.model = model_class()
        
    def load_data(self, data_path: str) -> None:
        """Load local training data."""
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Assume last column is target
        self.X_local = data.iloc[:, :-1]
        self.y_local = data.iloc[:, -1]
        
        self.logger.info(f"Loaded {len(data)} samples with {len(self.X_local.columns)} features")
    
    def train_local_model(self, global_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train model on local data."""
        if self.X_local is None or self.y_local is None:
            raise ValueError("Local data not loaded")
        
        # Set global parameters if provided
        if global_parameters is not None:
            try:
                self.set_model_parameters(global_parameters)
            except Exception as e:
                self.logger.warning(f"Could not set global parameters: {e}")
        
        # Train model
        start_time = time.time()
        
        for epoch in range(self.config.local_epochs):
            self.model.fit(self.X_local, self.y_local)
        
        training_time = time.time() - start_time
        
        # Apply differential privacy if enabled
        if self.config.use_differential_privacy and self.privacy_preserver:
            model_params = self.get_model_parameters()
            model_params = self.privacy_preserver.add_noise(
                model_params, self.config.privacy_budget
            )
            self.set_model_parameters(model_params)
        
        return {
            'training_time': training_time,
            'samples_trained': len(self.X_local),
            'local_epochs': self.config.local_epochs
        }
    
    def evaluate_local_model(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate model on local data."""
        if self.X_local is None or self.y_local is None:
            raise ValueError("Local data not loaded")
        
        if parameters is not None:
            self.set_model_parameters(parameters)
        
        # Make predictions
        y_pred = self.model.predict(self.X_local)
        
        # Calculate metrics based on task type
        from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
        
        try:
            # Try classification metrics
            accuracy = accuracy_score(self.y_local, y_pred)
            return {'accuracy': float(accuracy), 'samples': len(self.y_local)}
        except:
            # Fall back to regression metrics
            r2 = r2_score(self.y_local, y_pred)
            mse = mean_squared_error(self.y_local, y_pred)
            return {'r2': float(r2), 'mse': float(mse), 'samples': len(self.y_local)}


class FederatedServer:
    """
    Federated learning server that coordinates training and aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("FederatedServer")
        self.clients = {}
        self.global_model_params = None
        self.round_history = []
        self.convergence_tracker = ConvergenceTracker()
        
    def register_client(self, client: FederatedClient) -> None:
        """Register a client with the server."""
        self.clients[client.client_id] = client
        self.logger.info(f"Registered client {client.client_id}")
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """
        Run the complete federated learning process.
        
        Returns:
            Training results and metrics
        """
        
        self.logger.info("Starting federated learning")
        
        num_rounds = self.config.get('rounds', 10)
        min_clients = max(2, len(self.clients) // 2)  # At least half of clients
        
        fl_results = {
            'start_time': datetime.now().isoformat(),
            'total_rounds': num_rounds,
            'participating_clients': list(self.clients.keys()),
            'round_results': [],
            'convergence_info': {},
            'final_metrics': {}
        }
        
        for round_num in range(num_rounds):
            self.logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients(min_clients)
            
            # Run federated round
            round_result = self._run_round(round_num, selected_clients)
            fl_results['round_results'].append(round_result)
            
            # Check convergence
            if self.convergence_tracker.check_convergence(round_result):
                self.logger.info(f"Convergence achieved at round {round_num + 1}")
                break
        
        # Final evaluation
        fl_results['final_metrics'] = self._evaluate_global_model()
        fl_results['convergence_info'] = self.convergence_tracker.get_convergence_info()
        fl_results['end_time'] = datetime.now().isoformat()
        
        self.logger.info("Federated learning completed")
        return fl_results
    
    def _select_clients(self, min_clients: int) -> List[str]:
        """Select clients for the current round."""
        strategy = self.config.get('client_selection', 'random')
        
        if strategy == 'all':
            return list(self.clients.keys())
        elif strategy == 'random':
            import random
            available_clients = list(self.clients.keys())
            num_selected = min(len(available_clients), max(min_clients, len(available_clients) // 2))
            return random.sample(available_clients, num_selected)
        else:
            return list(self.clients.keys())[:min_clients]
    
    def _run_round(self, round_num: int, selected_client_ids: List[str]) -> FLRound:
        """Run a single federated learning round."""
        
        round_start = time.time()
        
        # Train local models
        local_updates = {}
        local_metrics = {}
        
        for client_id in selected_client_ids:
            client = self.clients[client_id]
            
            try:
                # Send global parameters to client
                train_result = client.train_local_model(self.global_model_params)
                
                # Get updated parameters
                local_params = client.get_model_parameters()
                local_updates[client_id] = local_params
                
                # Evaluate local model
                eval_metrics = client.evaluate_local_model()
                local_metrics[client_id] = eval_metrics
                
                self.logger.info(f"Client {client_id} completed round {round_num}")
                
            except Exception as e:
                self.logger.error(f"Client {client_id} failed in round {round_num}: {e}")
        
        # Aggregate updates
        if local_updates:
            self.global_model_params = self._aggregate_parameters(local_updates)
            aggregated_metrics = self._aggregate_metrics(local_metrics)
        else:
            aggregated_metrics = {}
        
        round_time = time.time() - round_start
        
        # Create round record
        round_result = FLRound(
            round_id=round_num,
            timestamp=datetime.now().isoformat(),
            participants=selected_client_ids,
            aggregated_metrics=aggregated_metrics,
            convergence_metrics={'round_time': round_time}
        )
        
        self.round_history.append(round_result)
        return round_result
    
    def _aggregate_parameters(self, local_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate parameters from multiple clients using FedAvg.
        
        Args:
            local_updates: Dictionary mapping client_id to their parameter updates
            
        Returns:
            Aggregated global parameters
        """
        
        aggregation_strategy = self.config.get('aggregation_strategy', 'fedavg')
        
        if aggregation_strategy == 'fedavg':
            return self._fedavg_aggregation(local_updates)
        else:
            self.logger.warning(f"Unknown aggregation strategy: {aggregation_strategy}, using FedAvg")
            return self._fedavg_aggregation(local_updates)
    
    def _fedavg_aggregation(self, local_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Federated Averaging (FedAvg) aggregation."""
        
        if not local_updates:
            return self.global_model_params or {}
        
        # Get sample counts for weighted averaging
        sample_counts = {}
        for client_id, client in self.clients.items():
            if client_id in local_updates:
                sample_counts[client_id] = len(getattr(client, 'X_local', [1]))  # Default to 1 if not available
        
        total_samples = sum(sample_counts.values())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_client_params = next(iter(local_updates.values()))
        
        for param_name, param_value in first_client_params.items():
            if isinstance(param_value, np.ndarray):
                # Initialize with zeros
                aggregated_params[param_name] = np.zeros_like(param_value)
            elif isinstance(param_value, list):
                # For neural network weights
                aggregated_params[param_name] = [np.zeros_like(w) for w in param_value]
            else:
                # For scalar values or other types
                aggregated_params[param_name] = 0
        
        # Weighted average
        for client_id, params in local_updates.items():
            weight = sample_counts[client_id] / total_samples
            
            for param_name, param_value in params.items():
                if isinstance(param_value, np.ndarray):
                    aggregated_params[param_name] += weight * param_value
                elif isinstance(param_value, list):
                    # For neural network weights
                    for i, w in enumerate(param_value):
                        aggregated_params[param_name][i] += weight * w
                else:
                    aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def _aggregate_metrics(self, local_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate evaluation metrics from clients."""
        
        if not local_metrics:
            return {}
        
        aggregated = {}
        metric_names = set()
        
        # Collect all metric names
        for metrics in local_metrics.values():
            metric_names.update(metrics.keys())
        
        # Calculate weighted averages
        for metric_name in metric_names:
            if metric_name == 'samples':
                continue  # Skip sample count in averaging
            
            total_weighted_value = 0
            total_samples = 0
            
            for client_metrics in local_metrics.values():
                if metric_name in client_metrics:
                    samples = client_metrics.get('samples', 1)
                    value = client_metrics[metric_name]
                    total_weighted_value += value * samples
                    total_samples += samples
            
            if total_samples > 0:
                aggregated[metric_name] = total_weighted_value / total_samples
        
        aggregated['total_samples'] = sum(
            metrics.get('samples', 1) for metrics in local_metrics.values()
        )
        
        return aggregated
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model on all clients."""
        
        if not self.global_model_params:
            return {}
        
        all_metrics = {}
        
        for client_id, client in self.clients.items():
            try:
                metrics = client.evaluate_local_model(self.global_model_params)
                all_metrics[client_id] = metrics
            except Exception as e:
                self.logger.error(f"Failed to evaluate global model on client {client_id}: {e}")
        
        return self._aggregate_metrics(all_metrics)


class ConvergenceTracker:
    """
    Tracks convergence of federated learning process.
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.wait_count = 0
        self.convergence_history = []
        
    def check_convergence(self, round_result: FLRound) -> bool:
        """Check if the training has converged."""
        
        # Use primary metric for convergence (accuracy or r2)
        current_metric = round_result.aggregated_metrics.get('accuracy') or \
                        round_result.aggregated_metrics.get('r2') or 0
        
        self.convergence_history.append(current_metric)
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return False
        
        # Check for improvement
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Check convergence
        return self.wait_count >= self.patience
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        return {
            'converged': self.wait_count >= self.patience,
            'best_metric': self.best_metric,
            'patience': self.patience,
            'wait_count': self.wait_count,
            'convergence_history': self.convergence_history
        }


class PrivacyPreserver:
    """
    Handles privacy preservation techniques for federated learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("PrivacyPreserver")
        self.noise_multiplier = config.get('noise_multiplier', 1.1)
        
        # Initialize encryption if available
        self.encryption_key = None
        if CRYPTO_AVAILABLE and config.get('use_encryption', False):
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
    
    def add_differential_privacy_noise(self, parameters: Dict[str, Any], 
                                     privacy_budget: float) -> Dict[str, Any]:
        """Add differential privacy noise to model parameters."""
        
        noisy_params = {}
        
        for param_name, param_value in parameters.items():
            if isinstance(param_value, np.ndarray):
                # Add Gaussian noise
                noise_scale = self.noise_multiplier / privacy_budget
                noise = np.random.normal(0, noise_scale, param_value.shape)
                noisy_params[param_name] = param_value + noise
                
            elif isinstance(param_value, list):
                # For neural network weights
                noisy_params[param_name] = []
                for weight in param_value:
                    if isinstance(weight, np.ndarray):
                        noise_scale = self.noise_multiplier / privacy_budget
                        noise = np.random.normal(0, noise_scale, weight.shape)
                        noisy_params[param_name].append(weight + noise)
                    else:
                        noisy_params[param_name].append(weight)
            else:
                # Keep non-array parameters as is
                noisy_params[param_name] = param_value
        
        return noisy_params
    
    def add_noise(self, parameters: Dict[str, Any], privacy_budget: float) -> Dict[str, Any]:
        """Add privacy-preserving noise to parameters."""
        return self.add_differential_privacy_noise(parameters, privacy_budget)
    
    def encrypt_parameters(self, parameters: Dict[str, Any]) -> bytes:
        """Encrypt model parameters for secure transmission."""
        
        if not self.cipher:
            raise ValueError("Encryption not available or not enabled")
        
        # Serialize parameters
        serialized_params = pickle.dumps(parameters)
        
        # Encrypt
        encrypted_params = self.cipher.encrypt(serialized_params)
        
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_parameters: bytes) -> Dict[str, Any]:
        """Decrypt model parameters."""
        
        if not self.cipher:
            raise ValueError("Encryption not available or not enabled")
        
        # Decrypt
        serialized_params = self.cipher.decrypt(encrypted_parameters)
        
        # Deserialize
        parameters = pickle.loads(serialized_params)
        
        return parameters
    
    def calculate_privacy_spent(self, rounds_completed: int, 
                               privacy_budget_per_round: float) -> float:
        """Calculate total privacy budget spent."""
        return rounds_completed * privacy_budget_per_round