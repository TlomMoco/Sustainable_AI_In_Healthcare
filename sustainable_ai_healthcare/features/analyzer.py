"""
Feature analysis tools for healthcare data with comprehensive statistical analysis,
visualization, and feature importance assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings

# Statistical tests
from scipy.stats import normaltest, shapiro, kstest, levene

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for healthcare datasets including:
    - Descriptive statistics
    - Distribution analysis
    - Correlation analysis
    - Feature importance
    - Statistical tests
    - Visualization
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the feature analyzer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.analysis_results = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_distributions(self, 
                            df: pd.DataFrame, 
                            target: Optional[pd.Series] = None,
                            save_plots: bool = False,
                            plot_dir: str = './plots') -> Dict:
        """
        Analyze feature distributions with statistical tests.
        
        Args:
            df: Input DataFrame
            target: Target variable
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary with distribution analysis results
        """
        logger.info("Analyzing feature distributions...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        distribution_results = {
            'numeric_distributions': {},
            'categorical_distributions': {},
            'normality_tests': {},
            'summary_stats': {}
        }
        
        # Analyze numeric features
        for col in numeric_cols:
            series = df[col].dropna()
            
            # Basic statistics
            stats_dict = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'percentiles': {
                    '25th': series.quantile(0.25),
                    '50th': series.quantile(0.50),
                    '75th': series.quantile(0.75),
                    '90th': series.quantile(0.90),
                    '95th': series.quantile(0.95),
                    '99th': series.quantile(0.99)
                }
            }
            distribution_results['numeric_distributions'][col] = stats_dict
            
            # Normality tests
            normality_results = {}
            try:
                # Shapiro-Wilk test (for n < 5000)
                if len(series) < 5000:
                    shapiro_stat, shapiro_p = shapiro(series)
                    normality_results['shapiro'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                
                # D'Agostino's normality test
                dagostino_stat, dagostino_p = normaltest(series)
                normality_results['dagostino'] = {
                    'statistic': dagostino_stat,
                    'p_value': dagostino_p,
                    'is_normal': dagostino_p > 0.05
                }
                
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_p = kstest(series, 'norm', args=(series.mean(), series.std()))
                normality_results['kolmogorov_smirnov'] = {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'is_normal': ks_p > 0.05
                }
                
            except Exception as e:
                logger.warning(f"Error in normality tests for {col}: {str(e)}")
            
            distribution_results['normality_tests'][col] = normality_results
            
            # Create distribution plot
            if save_plots:
                fig, axes = plt.subplots(2, 2, figsize=self.figsize)
                fig.suptitle(f'Distribution Analysis: {col}')
                
                # Histogram
                axes[0, 0].hist(series, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Histogram')
                axes[0, 0].set_xlabel(col)
                axes[0, 0].set_ylabel('Frequency')
                
                # Box plot
                axes[0, 1].boxplot(series)
                axes[0, 1].set_title('Box Plot')
                axes[0, 1].set_ylabel(col)
                
                # Q-Q plot
                stats.probplot(series, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot (Normal)')
                
                # Density plot
                series.plot.density(ax=axes[1, 1])
                axes[1, 1].set_title('Density Plot')
                axes[1, 1].set_xlabel(col)
                
                plt.tight_layout()
                if save_plots:
                    import os
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(f"{plot_dir}/distribution_{col}.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Analyze categorical features
        for col in categorical_cols:
            series = df[col].dropna()
            value_counts = series.value_counts()
            
            categorical_stats = {
                'unique_values': series.nunique(),
                'mode': series.mode()[0] if len(series.mode()) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.to_dict(),
                'entropy': -np.sum((value_counts / len(series)) * np.log2(value_counts / len(series) + 1e-10))
            }
            distribution_results['categorical_distributions'][col] = categorical_stats
            
            # Create categorical plot
            if save_plots and len(value_counts) <= 20:  # Only plot if reasonable number of categories
                plt.figure(figsize=self.figsize)
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(f"{plot_dir}/categorical_{col}.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        self.analysis_results['distributions'] = distribution_results
        logger.info(f"Distribution analysis complete for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        
        return distribution_results
    
    def analyze_correlations(self, 
                           df: pd.DataFrame, 
                           target: Optional[pd.Series] = None,
                           method: str = 'pearson',
                           save_plots: bool = False,
                           plot_dir: str = './plots') -> Dict:
        """
        Comprehensive correlation analysis.
        
        Args:
            df: Input DataFrame
            target: Target variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info(f"Analyzing correlations using {method} method...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        correlation_results = {
            'correlation_matrix': {},
            'target_correlations': {},
            'high_correlations': [],
            'correlation_clusters': {}
        }
        
        # Calculate correlation matrix
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr(method=method)
            correlation_results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            correlation_results['high_correlations'] = high_corr_pairs
            
            # Create correlation heatmap
            if save_plots:
                plt.figure(figsize=(max(10, len(corr_matrix.columns)), max(8, len(corr_matrix.columns))))
                mask = np.triu(np.ones_like(corr_matrix))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           mask=mask, square=True, fmt='.2f')
                plt.title(f'Feature Correlation Matrix ({method.title()})')
                plt.tight_layout()
                if save_plots:
                    import os
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(f"{plot_dir}/correlation_matrix_{method}.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        # Analyze correlations with target
        if target is not None:
            target_corr = {}
            for col in numeric_df.columns:
                try:
                    if method == 'pearson':
                        corr_val, p_val = pearsonr(numeric_df[col].dropna(), 
                                                  target.loc[numeric_df[col].dropna().index])
                    else:  # spearman
                        corr_val, p_val = spearmanr(numeric_df[col].dropna(), 
                                                   target.loc[numeric_df[col].dropna().index])
                    
                    target_corr[col] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Error calculating correlation for {col}: {str(e)}")
            
            correlation_results['target_correlations'] = target_corr
            
            # Plot target correlations
            if save_plots and target_corr:
                correlations = [v['correlation'] for v in target_corr.values()]
                features = list(target_corr.keys())
                
                plt.figure(figsize=self.figsize)
                colors = ['red' if abs(c) > 0.5 else 'blue' for c in correlations]
                plt.barh(features, correlations, color=colors, alpha=0.7)
                plt.xlabel('Correlation with Target')
                plt.title('Feature-Target Correlations')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                plt.tight_layout()
                if save_plots:
                    plt.savefig(f"{plot_dir}/target_correlations.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        self.analysis_results['correlations'] = correlation_results
        return correlation_results
    
    def analyze_feature_importance(self, 
                                 df: pd.DataFrame, 
                                 target: pd.Series,
                                 problem_type: str = 'auto',
                                 save_plots: bool = False,
                                 plot_dir: str = './plots') -> Dict:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            df: Input DataFrame
            target: Target variable
            problem_type: 'classification', 'regression', or 'auto'
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info("Analyzing feature importance...")
        
        from sklearn.feature_selection import (
            mutual_info_classif, mutual_info_regression,
            f_classif, f_regression, chi2
        )
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Auto-detect problem type
        if problem_type == 'auto':
            if target.nunique() <= 10 and target.dtype in ['int64', 'bool', 'object', 'category']:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        
        importance_results = {
            'problem_type': problem_type,
            'mutual_information': {},
            'statistical_tests': {},
            'tree_importance': {},
            'ranking_summary': {}
        }
        
        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Mutual Information
        try:
            if problem_type == 'classification':
                mi_scores = mutual_info_classif(numeric_df, target, random_state=42)
            else:
                mi_scores = mutual_info_regression(numeric_df, target, random_state=42)
            
            mi_dict = dict(zip(numeric_df.columns, mi_scores))
            importance_results['mutual_information'] = mi_dict
            
        except Exception as e:
            logger.warning(f"Error calculating mutual information: {str(e)}")
        
        # Statistical Tests
        try:
            if problem_type == 'classification':
                # F-test
                f_scores, f_pvals = f_classif(numeric_df, target)
                stat_dict = {
                    col: {'f_score': f_scores[i], 'p_value': f_pvals[i]}
                    for i, col in enumerate(numeric_df.columns)
                }
            else:
                # F-test for regression
                f_scores, f_pvals = f_regression(numeric_df, target)
                stat_dict = {
                    col: {'f_score': f_scores[i], 'p_value': f_pvals[i]}
                    for i, col in enumerate(numeric_df.columns)
                }
            
            importance_results['statistical_tests'] = stat_dict
            
        except Exception as e:
            logger.warning(f"Error calculating statistical tests: {str(e)}")
        
        # Tree-based feature importance
        try:
            if problem_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rf.fit(numeric_df, target)
            tree_importance = dict(zip(numeric_df.columns, rf.feature_importances_))
            importance_results['tree_importance'] = tree_importance
            
        except Exception as e:
            logger.warning(f"Error calculating tree-based importance: {str(e)}")
        
        # Create ranking summary
        rankings = {}
        for col in numeric_df.columns:
            rankings[col] = {}
            
            # Mutual information ranking
            if importance_results['mutual_information']:
                mi_values = list(importance_results['mutual_information'].values())
                mi_ranks = pd.Series(mi_values).rank(ascending=False)
                rankings[col]['mi_rank'] = mi_ranks[list(importance_results['mutual_information'].keys()).index(col)]
            
            # Statistical test ranking
            if importance_results['statistical_tests']:
                f_values = [v['f_score'] for v in importance_results['statistical_tests'].values()]
                f_ranks = pd.Series(f_values).rank(ascending=False)
                rankings[col]['stat_rank'] = f_ranks[list(importance_results['statistical_tests'].keys()).index(col)]
            
            # Tree importance ranking
            if importance_results['tree_importance']:
                tree_values = list(importance_results['tree_importance'].values())
                tree_ranks = pd.Series(tree_values).rank(ascending=False)
                rankings[col]['tree_rank'] = tree_ranks[list(importance_results['tree_importance'].keys()).index(col)]
            
            # Average ranking
            ranks = [v for v in rankings[col].values() if not pd.isna(v)]
            rankings[col]['avg_rank'] = np.mean(ranks) if ranks else np.nan
        
        importance_results['ranking_summary'] = rankings
        
        # Visualization
        if save_plots and importance_results['tree_importance']:
            # Feature importance plot
            importance_df = pd.DataFrame({
                'Feature': list(importance_results['tree_importance'].keys()),
                'Importance': list(importance_results['tree_importance'].values())
            }).sort_values('Importance', ascending=True)
            
            plt.figure(figsize=self.figsize)
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Feature Importance')
            plt.title('Tree-based Feature Importance')
            plt.tight_layout()
            if save_plots:
                import os
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(f"{plot_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        self.analysis_results['feature_importance'] = importance_results
        return importance_results
    
    def generate_feature_report(self, 
                              df: pd.DataFrame, 
                              target: Optional[pd.Series] = None,
                              save_report: bool = True,
                              report_path: str = './feature_analysis_report.html') -> str:
        """
        Generate comprehensive feature analysis report.
        
        Args:
            df: Input DataFrame
            target: Target variable
            save_report: Whether to save HTML report
            report_path: Path to save the report
            
        Returns:
            HTML report string
        """
        logger.info("Generating comprehensive feature analysis report...")
        
        # Run all analyses
        dist_results = self.analyze_distributions(df, target)
        corr_results = self.analyze_correlations(df, target)
        
        if target is not None:
            importance_results = self.analyze_feature_importance(df, target)
        else:
            importance_results = None
        
        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Healthcare Feature Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
                .warning {{ color: #ff6600; }}
                .good {{ color: #009900; }}
            </style>
        </head>
        <body>
            <h1>Healthcare Feature Analysis Report</h1>
            
            <div class="summary">
                <h2>Dataset Summary</h2>
                <p><strong>Dataset Shape:</strong> {df.shape[0]} samples, {df.shape[1]} features</p>
                <p><strong>Numeric Features:</strong> {len(df.select_dtypes(include=[np.number]).columns)}</p>
                <p><strong>Categorical Features:</strong> {len(df.select_dtypes(include=['object', 'category']).columns)}</p>
                <p><strong>Missing Values:</strong> {df.isnull().sum().sum()}</p>
                <p><strong>Target Variable:</strong> {'Provided' if target is not None else 'Not provided'}</p>
            </div>
            
            <h2>Data Quality Issues</h2>
            <ul>
        """
        
        # Add data quality issues
        quality_report = dist_results.get('summary_stats', {})
        missing_features = [col for col, count in df.isnull().sum().items() if count > 0]
        if missing_features:
            html_report += f"<li class='warning'>Features with missing values: {', '.join(missing_features[:10])}{'...' if len(missing_features) > 10 else ''}</li>"
        
        high_corr = corr_results.get('high_correlations', [])
        if high_corr:
            html_report += f"<li class='warning'>{len(high_corr)} pairs of highly correlated features found</li>"
        
        html_report += """
            </ul>
            
            <h2>Feature Distributions</h2>
            <table>
                <tr><th>Feature</th><th>Type</th><th>Missing %</th><th>Unique Values</th><th>Skewness</th></tr>
        """
        
        # Add distribution table
        for col in df.columns:
            col_type = 'Numeric' if df[col].dtype in [np.number, 'int64', 'float64'] else 'Categorical'
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            skewness = df[col].skew() if col_type == 'Numeric' else 'N/A'
            
            html_report += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{missing_pct:.1f}%</td>
                    <td>{unique_count}</td>
                    <td>{skewness:.2f if isinstance(skewness, float) else skewness}</td>
                </tr>
            """
        
        html_report += """
            </table>
        """
        
        # Add feature importance section if available
        if importance_results:
            html_report += """
                <h2>Feature Importance</h2>
                <table>
                    <tr><th>Feature</th><th>Tree Importance</th><th>Mutual Information</th><th>Average Rank</th></tr>
            """
            
            rankings = importance_results.get('ranking_summary', {})
            for col, ranking in rankings.items():
                tree_imp = importance_results.get('tree_importance', {}).get(col, 'N/A')
                mi_score = importance_results.get('mutual_information', {}).get(col, 'N/A')
                avg_rank = ranking.get('avg_rank', 'N/A')
                
                html_report += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{tree_imp:.4f if isinstance(tree_imp, float) else tree_imp}</td>
                        <td>{mi_score:.4f if isinstance(mi_score, float) else mi_score}</td>
                        <td>{avg_rank:.1f if isinstance(avg_rank, float) else avg_rank}</td>
                    </tr>
                """
            
            html_report += "</table>"
        
        html_report += """
            <h2>Recommendations</h2>
            <ul>
        """
        
        # Add recommendations
        if missing_features:
            html_report += "<li>Consider imputation strategies for features with missing values</li>"
        
        if high_corr:
            html_report += "<li>Remove or combine highly correlated features to reduce multicollinearity</li>"
        
        skewed_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if abs(df[col].skew()) > 2:
                skewed_features.append(col)
        
        if skewed_features:
            html_report += f"<li>Apply transformations to highly skewed features: {', '.join(skewed_features[:5])}{'...' if len(skewed_features) > 5 else ''}</li>"
        
        html_report += """
                <li>Validate data quality and domain-specific constraints</li>
                <li>Consider feature engineering opportunities</li>
            </ul>
            
        </body>
        </html>
        """
        
        if save_report:
            with open(report_path, 'w') as f:
                f.write(html_report)
            logger.info(f"Feature analysis report saved to {report_path}")
        
        return html_report