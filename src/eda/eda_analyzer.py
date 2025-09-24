"""
Exploratory Data Analysis Module

This module provides comprehensive EDA capabilities for healthcare data,
including statistical analysis, visualization, and insight generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
import logging
from pathlib import Path


class EDAAnalyzer:
    """
    Main EDA orchestrator that coordinates all analysis components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.visualizer = DataVisualizer(config)
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
    def perform_comprehensive_eda(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.
        
        Args:
            data: Dataset to analyze
            target_column: Name of target variable (if applicable)
            
        Returns:
            Comprehensive EDA report with insights and visualizations
        """
        self.logger.info("Starting comprehensive EDA")
        
        eda_report = {
            'dataset_overview': self._get_dataset_overview(data),
            'statistical_analysis': self.statistical_analyzer.analyze(data, target_column),
            'distribution_analysis': self._analyze_distributions(data),
            'correlation_analysis': self._analyze_correlations(data),
            'categorical_analysis': self._analyze_categorical_variables(data),
            'missing_data_analysis': self._analyze_missing_data(data),
            'outlier_analysis': self._analyze_outliers(data),
            'insights': []
        }
        
        # Generate visualizations if configured
        if self.config.get('generate_plots', True):
            eda_report['visualizations'] = self._generate_visualizations(data, target_column)
        
        # Generate insights based on analysis
        eda_report['insights'] = self._generate_insights(eda_report, data, target_column)
        
        self.logger.info("EDA completed successfully")
        return eda_report
    
    def _get_dataset_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset overview."""
        return {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': {
                'total': len(data.columns),
                'numerical': len(data.select_dtypes(include=[np.number]).columns),
                'categorical': len(data.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(data.select_dtypes(include=['datetime64']).columns)
            },
            'data_types': data.dtypes.value_counts().to_dict()
        }
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numerical variables."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        distribution_analysis = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Basic distribution statistics
            distribution_analysis[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
            }
            
            # Normality tests
            if len(col_data) >= 8:  # Minimum for Shapiro-Wilk
                shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000])  # Sample for large datasets
                distribution_analysis[col]['normality_test'] = {
                    'shapiro_wilk_statistic': float(shapiro_stat),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normal': bool(shapiro_p > 0.05)
                }
        
        return distribution_analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {}
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman correlation (for non-linear relationships)
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find high correlations
        threshold = self.config.get('correlation_threshold', 0.8)
        high_correlations = []
        
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]
                
                if abs(pearson_val) > threshold:
                    high_correlations.append({
                        'variable1': pearson_corr.columns[i],
                        'variable2': pearson_corr.columns[j],
                        'pearson_correlation': float(pearson_val),
                        'spearman_correlation': float(spearman_val)
                    })
        
        return {
            'pearson_matrix': pearson_corr.to_dict(),
            'spearman_matrix': spearman_corr.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _analyze_categorical_variables(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables."""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        categorical_analysis = {}
        
        for col in categorical_cols:
            col_data = data[col].dropna()
            value_counts = col_data.value_counts()
            
            categorical_analysis[col] = {
                'unique_values': len(value_counts),
                'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'distribution': value_counts.head(10).to_dict(),
                'cardinality_ratio': len(value_counts) / len(col_data) if len(col_data) > 0 else 0
            }
        
        return categorical_analysis
    
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        # Missing data by column
        missing_by_column = {}
        for col in data.columns:
            missing_count = missing_counts[col]
            if missing_count > 0:
                missing_by_column[col] = {
                    'count': int(missing_count),
                    'percentage': float((missing_count / len(data)) * 100)
                }
        
        # Missing data patterns
        missing_patterns = data.isnull().value_counts().head(10).to_dict()
        
        return {
            'total_missing_values': int(total_missing),
            'percentage_missing': float((total_missing / (len(data) * len(data.columns))) * 100),
            'missing_by_column': missing_by_column,
            'missing_patterns': {str(k): v for k, v in missing_patterns.items()}
        }
    
    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numerical variables."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers_zscore = col_data[z_scores > 3]
            
            outlier_analysis[col] = {
                'iqr_method': {
                    'count': len(outliers_iqr),
                    'percentage': float((len(outliers_iqr) / len(col_data)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'zscore_method': {
                    'count': len(outliers_zscore),
                    'percentage': float((len(outliers_zscore) / len(col_data)) * 100)
                }
            }
        
        return outlier_analysis
    
    def _generate_visualizations(self, data: pd.DataFrame, target_column: Optional[str]) -> Dict[str, str]:
        """Generate and save visualizations."""
        viz_paths = {}
        
        # Distribution plots for numerical variables
        viz_paths['distributions'] = self.visualizer.plot_distributions(data)
        
        # Correlation heatmap
        viz_paths['correlation_heatmap'] = self.visualizer.plot_correlation_heatmap(data)
        
        # Missing data visualization
        viz_paths['missing_data'] = self.visualizer.plot_missing_data(data)
        
        # Categorical variable plots
        viz_paths['categorical_plots'] = self.visualizer.plot_categorical_variables(data)
        
        # Target variable analysis (if applicable)
        if target_column and target_column in data.columns:
            viz_paths['target_analysis'] = self.visualizer.plot_target_analysis(data, target_column)
        
        return viz_paths
    
    def _generate_insights(self, eda_report: Dict[str, Any], data: pd.DataFrame, target_column: Optional[str]) -> List[str]:
        """Generate insights based on EDA findings."""
        insights = []
        
        # Dataset size insights
        shape = eda_report['dataset_overview']['shape']
        if shape[0] < 1000:
            insights.append("Small dataset detected - consider data augmentation techniques")
        elif shape[0] > 100000:
            insights.append("Large dataset detected - consider sampling for initial analysis")
        
        # Missing data insights
        missing_analysis = eda_report['missing_data_analysis']
        if missing_analysis['percentage_missing'] > 20:
            insights.append("High proportion of missing data detected - investigate data collection process")
        
        # Correlation insights
        if 'correlation_analysis' in eda_report and 'high_correlations' in eda_report['correlation_analysis']:
            high_corrs = eda_report['correlation_analysis']['high_correlations']
            if len(high_corrs) > 0:
                insights.append(f"Found {len(high_corrs)} highly correlated variable pairs - consider feature selection")
        
        # Distribution insights
        if 'distribution_analysis' in eda_report:
            skewed_vars = []
            for var, stats in eda_report['distribution_analysis'].items():
                if abs(stats.get('skewness', 0)) > 2:
                    skewed_vars.append(var)
            
            if skewed_vars:
                insights.append(f"Highly skewed variables detected: {', '.join(skewed_vars)} - consider transformations")
        
        # Outlier insights
        if 'outlier_analysis' in eda_report:
            high_outlier_vars = []
            for var, analysis in eda_report['outlier_analysis'].items():
                if analysis['iqr_method']['percentage'] > 10:
                    high_outlier_vars.append(var)
            
            if high_outlier_vars:
                insights.append(f"High outlier percentages in: {', '.join(high_outlier_vars)} - investigate data quality")
        
        # Categorical variable insights
        if 'categorical_analysis' in eda_report:
            high_cardinality_vars = []
            for var, analysis in eda_report['categorical_analysis'].items():
                if analysis['cardinality_ratio'] > 0.9:
                    high_cardinality_vars.append(var)
            
            if high_cardinality_vars:
                insights.append(f"High cardinality categorical variables: {', '.join(high_cardinality_vars)} - consider grouping strategies")
        
        return insights


class DataVisualizer:
    """
    Handles all data visualization tasks for EDA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        warnings.filterwarnings('ignore')
    
    def plot_distributions(self, data: pd.DataFrame) -> str:
        """Create distribution plots for numerical variables."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ""
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram with KDE
                data[col].hist(bins=30, alpha=0.7, ax=axes[i], density=True)
                data[col].plot.kde(ax=axes[i], color='red')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_correlation_heatmap(self, data: pd.DataFrame) -> str:
        """Create correlation heatmap."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return ""
        
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        output_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_missing_data(self, data: pd.DataFrame) -> str:
        """Visualize missing data patterns."""
        missing_data = data.isnull()
        
        if not missing_data.any().any():
            return ""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Missing data heatmap
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis', ax=ax1)
        ax1.set_title('Missing Data Pattern')
        
        # Missing data bar chart
        missing_counts = missing_data.sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if not missing_counts.empty:
            missing_counts.plot(kind='bar', ax=ax2)
            ax2.set_title('Missing Data Count by Column')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Missing Count')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "missing_data.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_categorical_variables(self, data: pd.DataFrame) -> str:
        """Plot distributions of categorical variables."""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return ""
        
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                value_counts = data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "categorical_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_target_analysis(self, data: pd.DataFrame, target_column: str) -> str:
        """Analyze relationship between features and target variable."""
        if target_column not in data.columns:
            return ""
        
        # Determine if target is categorical or numerical
        target_is_numeric = pd.api.types.is_numeric_dtype(data[target_column])
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) == 0:
            return ""
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:len(axes)]):
            if target_is_numeric:
                # Scatter plot for numeric target
                axes[i].scatter(data[col], data[target_column], alpha=0.5)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(target_column)
                axes[i].set_title(f'{col} vs {target_column}')
            else:
                # Box plot for categorical target
                unique_targets = data[target_column].unique()
                if len(unique_targets) <= 10:  # Only if reasonable number of categories
                    data.boxplot(column=col, by=target_column, ax=axes[i])
                    axes[i].set_title(f'{col} by {target_column}')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "target_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


class StatisticalAnalyzer:
    """
    Performs statistical analysis and hypothesis testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Dataset to analyze
            target_column: Target variable for supervised analysis
            
        Returns:
            Statistical analysis results
        """
        analysis_results = {
            'descriptive_statistics': self._descriptive_statistics(data),
            'normality_tests': self._normality_tests(data),
            'independence_tests': self._independence_tests(data)
        }
        
        if target_column and target_column in data.columns:
            analysis_results['target_analysis'] = self._target_analysis(data, target_column)
        
        return analysis_results
    
    def _descriptive_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {}
        
        # Basic descriptive statistics
        desc_stats = numeric_data.describe()
        
        # Additional statistics
        additional_stats = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                additional_stats[col] = {
                    'variance': float(col_data.var()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis()),
                    'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else None
                }
        
        return {
            'basic_statistics': desc_stats.to_dict(),
            'additional_statistics': additional_stats
        }
    
    def _normality_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform normality tests on numerical variables."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        normality_results = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) < 8:  # Minimum sample size
                continue
            
            # Use sample for large datasets
            sample_data = col_data.sample(min(5000, len(col_data)))
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            
            # Anderson-Darling test
            anderson_result = stats.anderson(sample_data, dist='norm')
            
            normality_results[col] = {
                'shapiro_wilk': {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': bool(shapiro_p > 0.05)
                },
                'anderson_darling': {
                    'statistic': float(anderson_result.statistic),
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_levels': anderson_result.significance_level.tolist()
                }
            }
        
        return normality_results
    
    def _independence_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test independence between categorical variables."""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) < 2:
            return {}
        
        independence_results = {}
        
        # Test pairs of categorical variables
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                # Create contingency table
                contingency_table = pd.crosstab(data[col1], data[col2])
                
                # Chi-square test
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                independence_results[f"{col1}_vs_{col2}"] = {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'is_independent': bool(p_value > 0.05)
                }
        
        return independence_results
    
    def _target_analysis(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze relationships with target variable."""
        target_analysis = {}
        
        # Determine if target is categorical or numerical
        target_is_numeric = pd.api.types.is_numeric_dtype(data[target_column])
        
        if target_is_numeric:
            # Correlation with numerical features
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            correlations = {}
            for col in numeric_cols:
                if not data[col].isnull().all():
                    corr_coeff, p_value = stats.pearsonr(
                        data[col].dropna(), 
                        data[target_column].loc[data[col].dropna().index]
                    )
                    correlations[col] = {
                        'correlation': float(corr_coeff),
                        'p_value': float(p_value),
                        'is_significant': bool(p_value < 0.05)
                    }
            
            target_analysis['correlations'] = correlations
        
        else:
            # ANOVA for categorical target with numerical features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            anova_results = {}
            
            for col in numeric_cols:
                groups = [group[col].dropna() for name, group in data.groupby(target_column)]
                groups = [group for group in groups if len(group) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[col] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'is_significant': bool(p_value < 0.05)
                    }
            
            target_analysis['anova_results'] = anova_results
        
        return target_analysis