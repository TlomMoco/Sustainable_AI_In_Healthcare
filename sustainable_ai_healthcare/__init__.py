"""
Sustainable AI in Healthcare - A comprehensive ML pipeline for healthcare applications.

This package provides tools for building sustainable and efficient machine learning
models for healthcare applications, with focus on:
- Energy-efficient model training
- Carbon footprint tracking
- Responsible AI practices
- Healthcare-specific data processing
"""

__version__ = "1.0.0"
__author__ = "Daniel & Collaborator"
__email__ = "contact@sustainable-ai-healthcare.com"

from .data import DataLoader, HealthcareDataProcessor
from .features import FeatureAnalyzer, FeatureSelector
from .models import ModelTrainer, MLPipeline
from .evaluation import ModelEvaluator, PerformanceMetrics
from .sustainability import SustainabilityTracker, CarbonFootprintMonitor

__all__ = [
    "DataLoader",
    "HealthcareDataProcessor", 
    "FeatureAnalyzer",
    "FeatureSelector",
    "ModelTrainer",
    "MLPipeline",
    "ModelEvaluator",
    "PerformanceMetrics",
    "SustainabilityTracker",
    "CarbonFootprintMonitor",
]