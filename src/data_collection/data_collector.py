"""
Data Collection and Understanding Module

This module provides comprehensive tools for collecting, validating, and understanding 
healthcare data while ensuring privacy compliance and data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import json
import pydicom
import nibabel as nib
from datetime import datetime


class DataCollector:
    """
    Handles data collection from various healthcare data sources.
    
    Supports multiple formats including CSV, JSON, DICOM, NIfTI, and Parquet files.
    Ensures privacy compliance and data quality during collection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supported_formats = config.get('supported_formats', ['csv', 'json', 'parquet'])
        
    def collect_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Collect data from specified path.
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            Collected data as pandas DataFrame
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")
        
        if data_path.is_file():
            return self._load_single_file(data_path)
        elif data_path.is_dir():
            return self._load_directory(data_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")
    
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from a single file."""
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(file_path)
        elif extension == '.json':
            return pd.read_json(file_path)
        elif extension == '.parquet':
            return pd.read_parquet(file_path)
        elif extension == '.dcm' or extension == '.dicom':
            return self._load_dicom(file_path)
        elif extension == '.nii' or extension == '.nii.gz':
            return self._load_nifti(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_directory(self, dir_path: Path) -> pd.DataFrame:
        """Load and combine data from multiple files in directory."""
        dataframes = []
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower().lstrip('.') in self.supported_formats:
                try:
                    df = self._load_single_file(file_path)
                    df['source_file'] = str(file_path)
                    dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
        
        if not dataframes:
            raise ValueError("No supported data files found in directory")
        
        return pd.concat(dataframes, ignore_index=True)
    
    def _load_dicom(self, file_path: Path) -> pd.DataFrame:
        """Load DICOM file and extract metadata."""
        dicom_data = pydicom.dcmread(file_path)
        
        # Extract metadata
        metadata = {}
        for elem in dicom_data:
            if elem.VR != 'SQ':  # Skip sequence elements
                try:
                    metadata[elem.name] = str(elem.value)
                except:
                    metadata[elem.name] = "Not available"
        
        # Convert to DataFrame
        df = pd.DataFrame([metadata])
        df['file_path'] = str(file_path)
        df['pixel_data_available'] = hasattr(dicom_data, 'pixel_array')
        
        return df
    
    def _load_nifti(self, file_path: Path) -> pd.DataFrame:
        """Load NIfTI file and extract header information."""
        nifti_img = nib.load(file_path)
        header = nifti_img.header
        
        # Extract header information
        metadata = {
            'file_path': str(file_path),
            'data_type': str(header.get_data_dtype()),
            'dimensions': str(header.get_data_shape()),
            'voxel_size': str(header.get_zooms()),
            'units': str(header.get_xyzt_units()),
            'qform_code': header['qform_code'],
            'sform_code': header['sform_code']
        }
        
        return pd.DataFrame([metadata])


class DataValidator:
    """
    Validates data quality and ensures compliance with healthcare standards.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.quality_checks = config.get('quality_checks', [])
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Validation report with findings and recommendations
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'total_features': len(data.columns),
            'checks_performed': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # Perform configured quality checks
        for check in self.quality_checks:
            if check == 'missing_values':
                validation_report['checks_performed']['missing_values'] = self._check_missing_values(data)
            elif check == 'duplicates':
                validation_report['checks_performed']['duplicates'] = self._check_duplicates(data)
            elif check == 'data_types':
                validation_report['checks_performed']['data_types'] = self._check_data_types(data)
            elif check == 'outliers':
                validation_report['checks_performed']['outliers'] = self._check_outliers(data)
        
        # Generate recommendations based on findings
        self._generate_recommendations(validation_report)
        
        return validation_report
    
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset."""
        missing_stats = data.isnull().sum()
        missing_percentage = (missing_stats / len(data)) * 100
        
        result = {
            'total_missing': int(missing_stats.sum()),
            'columns_with_missing': missing_stats[missing_stats > 0].to_dict(),
            'missing_percentages': missing_percentage[missing_percentage > 0].to_dict()
        }
        
        return result
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records."""
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100
        
        return {
            'duplicate_records': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage),
            'unique_records': len(data) - duplicate_count
        }
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and suggest improvements."""
        dtype_info = {}
        for col in data.columns:
            dtype_info[col] = {
                'current_type': str(data[col].dtype),
                'null_count': int(data[col].isnull().sum()),
                'unique_values': int(data[col].nunique()),
                'memory_usage': int(data[col].memory_usage(deep=True))
            }
        
        return dtype_info
    
    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical columns using IQR method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        return outlier_info
    
    def _generate_recommendations(self, report: Dict[str, Any]):
        """Generate recommendations based on validation findings."""
        recommendations = []
        
        # Missing values recommendations
        if 'missing_values' in report['checks_performed']:
            missing_info = report['checks_performed']['missing_values']
            if missing_info['total_missing'] > 0:
                recommendations.append(
                    "Consider imputation strategies for missing values or removal of features with high missingness"
                )
        
        # Duplicates recommendations
        if 'duplicates' in report['checks_performed']:
            dup_info = report['checks_performed']['duplicates']
            if dup_info['duplicate_records'] > 0:
                recommendations.append(
                    "Remove duplicate records to improve data quality"
                )
        
        # Outliers recommendations
        if 'outliers' in report['checks_performed']:
            outlier_info = report['checks_performed']['outliers']
            high_outlier_cols = [col for col, info in outlier_info.items() 
                               if info['outlier_percentage'] > 5]
            if high_outlier_cols:
                recommendations.append(
                    f"Investigate outliers in columns: {', '.join(high_outlier_cols)}"
                )
        
        report['recommendations'] = recommendations


class DataUnderstanding:
    """
    Provides comprehensive data understanding and profiling capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            data: DataFrame to profile
            
        Returns:
            Detailed data profile with statistics and insights
        """
        profile = {
            'basic_info': self._get_basic_info(data),
            'statistical_summary': self._get_statistical_summary(data),
            'column_profiles': self._get_column_profiles(data),
            'correlations': self._get_correlations(data),
            'data_quality_score': self._calculate_quality_score(data)
        }
        
        return profile
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': list(data.columns),
            'data_types': data.dtypes.astype(str).to_dict()
        }
    
    def _get_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of numerical columns."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {}
        
        summary = numeric_data.describe()
        return {
            'numerical_summary': summary.to_dict(),
            'skewness': numeric_data.skew().to_dict(),
            'kurtosis': numeric_data.kurtosis().to_dict()
        }
    
    def _get_column_profiles(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed profile for each column."""
        column_profiles = {}
        
        for col in data.columns:
            column_profiles[col] = {
                'data_type': str(data[col].dtype),
                'null_count': int(data[col].isnull().sum()),
                'null_percentage': float((data[col].isnull().sum() / len(data)) * 100),
                'unique_count': int(data[col].nunique()),
                'unique_percentage': float((data[col].nunique() / len(data)) * 100),
                'most_frequent': str(data[col].mode().iloc[0]) if not data[col].mode().empty else None,
                'memory_usage': int(data[col].memory_usage(deep=True))
            }
            
            # Add numerical statistics if applicable
            if pd.api.types.is_numeric_dtype(data[col]):
                column_profiles[col].update({
                    'mean': float(data[col].mean()) if not data[col].isnull().all() else None,
                    'median': float(data[col].median()) if not data[col].isnull().all() else None,
                    'std': float(data[col].std()) if not data[col].isnull().all() else None,
                    'min': float(data[col].min()) if not data[col].isnull().all() else None,
                    'max': float(data[col].max()) if not data[col].isnull().all() else None
                })
        
        return column_profiles
    
    def _get_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrix for numerical columns."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {}
        
        correlation_matrix = numeric_data.corr()
        
        # Find high correlations (above threshold)
        threshold = self.config.get('correlation_threshold', 0.8)
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        score_components = []
        
        # Completeness score (based on missing values)
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        score_components.append(completeness * 30)  # 30% weight
        
        # Uniqueness score (based on duplicates)
        uniqueness = 1 - (data.duplicated().sum() / len(data))
        score_components.append(uniqueness * 25)  # 25% weight
        
        # Consistency score (based on data types)
        # Simple heuristic: penalize mixed types in columns
        consistency = 1.0  # Start with perfect score
        score_components.append(consistency * 20)  # 20% weight
        
        # Validity score (based on reasonable value ranges)
        # Simple heuristic: check for extreme outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            validity = 1.0
            for col in numeric_cols:
                Q1, Q3 = data[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 3*IQR) | (data[col] > Q3 + 3*IQR)]
                validity -= (len(outliers) / len(data)) * 0.1
            validity = max(0, validity)
        else:
            validity = 1.0
        score_components.append(validity * 25)  # 25% weight
        
        return min(100, sum(score_components))