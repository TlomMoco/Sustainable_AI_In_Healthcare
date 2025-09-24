"""
Data Preprocessing Module

This module provides comprehensive data preprocessing capabilities for healthcare data,
including cleaning, transformation, and normalization while preserving data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging
import joblib
from pathlib import Path


class DataPreprocessor:
    """
    Main preprocessing orchestrator that coordinates all preprocessing steps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cleaner = DataCleaner(config)
        self.transformer = DataTransformer(config)
        self.preprocessors = {}  # Store fitted preprocessors
        
    def preprocess_data(self, data: pd.DataFrame, fit_preprocessors: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw data to preprocess
            fit_preprocessors: Whether to fit preprocessors (True for training data)
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_report)
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        preprocessing_report = {
            'original_shape': data.shape,
            'steps_applied': [],
            'transformations': {}
        }
        
        # Step 1: Data Cleaning
        self.logger.info("Step 1: Data cleaning")
        cleaned_data, cleaning_report = self.cleaner.clean_data(data)
        preprocessing_report['steps_applied'].append('cleaning')
        preprocessing_report['transformations']['cleaning'] = cleaning_report
        
        # Step 2: Handle Missing Values
        self.logger.info("Step 2: Handling missing values")
        imputed_data, imputation_report = self._handle_missing_values(
            cleaned_data, fit_preprocessors
        )
        preprocessing_report['steps_applied'].append('imputation')
        preprocessing_report['transformations']['imputation'] = imputation_report
        
        # Step 3: Outlier Treatment
        self.logger.info("Step 3: Outlier treatment")
        outlier_treated_data, outlier_report = self._handle_outliers(
            imputed_data, fit_preprocessors
        )
        preprocessing_report['steps_applied'].append('outlier_treatment')
        preprocessing_report['transformations']['outlier_treatment'] = outlier_report
        
        # Step 4: Feature Scaling/Normalization
        self.logger.info("Step 4: Feature scaling")
        scaled_data, scaling_report = self._scale_features(
            outlier_treated_data, fit_preprocessors
        )
        preprocessing_report['steps_applied'].append('scaling')
        preprocessing_report['transformations']['scaling'] = scaling_report
        
        # Step 5: Categorical Encoding
        self.logger.info("Step 5: Categorical encoding")
        encoded_data, encoding_report = self._encode_categorical(
            scaled_data, fit_preprocessors
        )
        preprocessing_report['steps_applied'].append('encoding')
        preprocessing_report['transformations']['encoding'] = encoding_report
        
        preprocessing_report['final_shape'] = encoded_data.shape
        preprocessing_report['shape_change'] = {
            'rows_removed': data.shape[0] - encoded_data.shape[0],
            'columns_added': encoded_data.shape[1] - data.shape[1]
        }
        
        self.logger.info(f"Preprocessing completed. Shape: {data.shape} -> {encoded_data.shape}")
        return encoded_data, preprocessing_report
    
    def _handle_missing_values(self, data: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values using configured strategy."""
        strategy = self.config.get('missing_value_strategy', 'median')
        
        # Separate numerical and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        imputed_data = data.copy()
        report = {'strategy': strategy, 'columns_imputed': []}
        
        # Handle numerical columns
        if numeric_cols:
            if fit:
                if strategy in ['mean', 'median']:
                    self.preprocessors['numeric_imputer'] = SimpleImputer(strategy=strategy)
                elif strategy == 'knn':
                    self.preprocessors['numeric_imputer'] = KNNImputer(n_neighbors=5)
                else:
                    self.preprocessors['numeric_imputer'] = SimpleImputer(strategy='median')
                
                imputed_numeric = self.preprocessors['numeric_imputer'].fit_transform(data[numeric_cols])
            else:
                if 'numeric_imputer' in self.preprocessors:
                    imputed_numeric = self.preprocessors['numeric_imputer'].transform(data[numeric_cols])
                else:
                    imputed_numeric = data[numeric_cols].values
            
            imputed_data[numeric_cols] = imputed_numeric
            report['columns_imputed'].extend(numeric_cols)
        
        # Handle categorical columns
        if categorical_cols:
            if fit:
                self.preprocessors['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
                imputed_categorical = self.preprocessors['categorical_imputer'].fit_transform(data[categorical_cols])
            else:
                if 'categorical_imputer' in self.preprocessors:
                    imputed_categorical = self.preprocessors['categorical_imputer'].transform(data[categorical_cols])
                else:
                    imputed_categorical = data[categorical_cols].values
            
            imputed_data[categorical_cols] = imputed_categorical
            report['columns_imputed'].extend(categorical_cols)
        
        return imputed_data, report
    
    def _handle_outliers(self, data: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using configured method."""
        method = self.config.get('outlier_detection', 'iqr')
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return data, {'method': method, 'outliers_removed': 0}
        
        outlier_data = data.copy()
        total_outliers = 0
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                total_outliers += outliers.sum()
                
                # Cap outliers instead of removing them
                outlier_data.loc[data[col] < lower_bound, col] = lower_bound
                outlier_data.loc[data[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(data[col]))
                outliers = z_scores > 3
                total_outliers += outliers.sum()
                
                # Cap outliers
                median_val = data[col].median()
                outlier_data.loc[outliers, col] = median_val
        
        elif method == 'isolation_forest':
            if fit:
                self.preprocessors['outlier_detector'] = IsolationForest(
                    contamination=0.1, random_state=42
                )
                outlier_labels = self.preprocessors['outlier_detector'].fit_predict(data[numeric_cols])
            else:
                if 'outlier_detector' in self.preprocessors:
                    outlier_labels = self.preprocessors['outlier_detector'].predict(data[numeric_cols])
                else:
                    outlier_labels = np.ones(len(data))
            
            outliers = outlier_labels == -1
            total_outliers = outliers.sum()
            
            # Replace outliers with median values
            for col in numeric_cols:
                median_val = data[col].median()
                outlier_data.loc[outliers, col] = median_val
        
        report = {
            'method': method,
            'outliers_treated': int(total_outliers),
            'columns_processed': numeric_cols
        }
        
        return outlier_data, report
    
    def _scale_features(self, data: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features using configured method."""
        method = self.config.get('normalization', 'standard')
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return data, {'method': method, 'columns_scaled': []}
        
        scaled_data = data.copy()
        
        if fit:
            if method == 'standard':
                self.preprocessors['scaler'] = StandardScaler()
            elif method == 'minmax':
                self.preprocessors['scaler'] = MinMaxScaler()
            elif method == 'robust':
                self.preprocessors['scaler'] = RobustScaler()
            else:
                self.preprocessors['scaler'] = StandardScaler()
            
            scaled_values = self.preprocessors['scaler'].fit_transform(data[numeric_cols])
        else:
            if 'scaler' in self.preprocessors:
                scaled_values = self.preprocessors['scaler'].transform(data[numeric_cols])
            else:
                scaled_values = data[numeric_cols].values
        
        scaled_data[numeric_cols] = scaled_values
        
        report = {
            'method': method,
            'columns_scaled': numeric_cols
        }
        
        return scaled_data, report
    
    def _encode_categorical(self, data: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical variables using configured method."""
        method = self.config.get('categorical_encoding', 'onehot')
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not categorical_cols:
            return data, {'method': method, 'columns_encoded': []}
        
        encoded_data = data.copy()
        
        if method == 'onehot':
            if fit:
                self.preprocessors['encoder'] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_values = self.preprocessors['encoder'].fit_transform(data[categorical_cols])
                # Get feature names
                feature_names = self.preprocessors['encoder'].get_feature_names_out(categorical_cols)
            else:
                if 'encoder' in self.preprocessors:
                    encoded_values = self.preprocessors['encoder'].transform(data[categorical_cols])
                    feature_names = self.preprocessors['encoder'].get_feature_names_out(categorical_cols)
                else:
                    return data, {'method': method, 'columns_encoded': []}
            
            # Remove original categorical columns and add encoded ones
            encoded_data = encoded_data.drop(columns=categorical_cols)
            encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=encoded_data.index)
            encoded_data = pd.concat([encoded_data, encoded_df], axis=1)
        
        elif method == 'label':
            for col in categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                    self.preprocessors[f'encoder_{col}'] = encoder
                else:
                    if f'encoder_{col}' in self.preprocessors:
                        # Handle unknown categories
                        encoder = self.preprocessors[f'encoder_{col}']
                        known_categories = set(encoder.classes_)
                        encoded_data[col] = data[col].astype(str).apply(
                            lambda x: encoder.transform([x])[0] if x in known_categories else -1
                        )
        
        report = {
            'method': method,
            'columns_encoded': categorical_cols,
            'new_columns': encoded_data.shape[1] - data.shape[1] + len(categorical_cols)
        }
        
        return encoded_data, report
    
    def save_preprocessors(self, path: str):
        """Save fitted preprocessors to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessors, path)
        self.logger.info(f"Preprocessors saved to {path}")
    
    def load_preprocessors(self, path: str):
        """Load preprocessors from disk."""
        if Path(path).exists():
            self.preprocessors = joblib.load(path)
            self.logger.info(f"Preprocessors loaded from {path}")
        else:
            self.logger.warning(f"Preprocessor file not found: {path}")


class DataCleaner:
    """
    Handles data cleaning operations including duplicate removal and basic validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive data cleaning.
        
        Args:
            data: Raw data to clean
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        cleaned_data = data.copy()
        report = {
            'original_rows': len(data),
            'operations_performed': []
        }
        
        # Remove exact duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        duplicates_removed = initial_rows - len(cleaned_data)
        
        if duplicates_removed > 0:
            report['operations_performed'].append(f"Removed {duplicates_removed} duplicate rows")
        
        # Remove columns with all missing values
        null_cols = cleaned_data.columns[cleaned_data.isnull().all()].tolist()
        if null_cols:
            cleaned_data = cleaned_data.drop(columns=null_cols)
            report['operations_performed'].append(f"Removed columns with all null values: {null_cols}")
        
        # Remove rows with all missing values
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(how='all')
        null_rows_removed = initial_rows - len(cleaned_data)
        
        if null_rows_removed > 0:
            report['operations_performed'].append(f"Removed {null_rows_removed} rows with all null values")
        
        # Basic data type corrections
        cleaned_data = self._correct_data_types(cleaned_data)
        report['operations_performed'].append("Applied data type corrections")
        
        report['final_rows'] = len(cleaned_data)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        
        return cleaned_data, report
    
    def _correct_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic data type corrections."""
        corrected_data = data.copy()
        
        for col in corrected_data.columns:
            # Try to convert string numbers to numeric
            if corrected_data[col].dtype == 'object':
                # Check if column contains numeric strings
                sample_values = corrected_data[col].dropna().head(100)
                if len(sample_values) > 0:
                    try:
                        pd.to_numeric(sample_values)
                        corrected_data[col] = pd.to_numeric(corrected_data[col], errors='coerce')
                    except (ValueError, TypeError):
                        # Try datetime conversion
                        try:
                            pd.to_datetime(sample_values)
                            corrected_data[col] = pd.to_datetime(corrected_data[col], errors='coerce')
                        except (ValueError, TypeError):
                            pass
        
        return corrected_data


class DataTransformer:
    """
    Handles advanced data transformations and feature engineering preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def apply_transformations(self, data: pd.DataFrame, transformations: List[str]) -> pd.DataFrame:
        """
        Apply specified transformations to the data.
        
        Args:
            data: Data to transform
            transformations: List of transformation names to apply
            
        Returns:
            Transformed data
        """
        transformed_data = data.copy()
        
        for transformation in transformations:
            if transformation == 'log_transform':
                transformed_data = self._log_transform(transformed_data)
            elif transformation == 'sqrt_transform':
                transformed_data = self._sqrt_transform(transformed_data)
            elif transformation == 'box_cox':
                transformed_data = self._box_cox_transform(transformed_data)
            elif transformation == 'normalize_text':
                transformed_data = self._normalize_text(transformed_data)
        
        return transformed_data
    
    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to positive numerical columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        transformed_data = data.copy()
        
        for col in numeric_cols:
            if (data[col] > 0).all():
                transformed_data[col] = np.log1p(data[col])
        
        return transformed_data
    
    def _sqrt_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply square root transformation to non-negative numerical columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        transformed_data = data.copy()
        
        for col in numeric_cols:
            if (data[col] >= 0).all():
                transformed_data[col] = np.sqrt(data[col])
        
        return transformed_data
    
    def _box_cox_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Box-Cox transformation to positive numerical columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        transformed_data = data.copy()
        
        for col in numeric_cols:
            if (data[col] > 0).all():
                transformed_values, _ = stats.boxcox(data[col])
                transformed_data[col] = transformed_values
        
        return transformed_data
    
    def _normalize_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns by converting to lowercase and removing extra spaces."""
        text_cols = data.select_dtypes(include=['object']).columns
        transformed_data = data.copy()
        
        for col in text_cols:
            transformed_data[col] = (
                transformed_data[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
        
        return transformed_data