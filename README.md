# Sustainable AI in Healthcare

A comprehensive AI pipeline for developing sustainable and ethical AI solutions in healthcare, featuring federated learning capabilities, comprehensive model evaluation, and advanced interpretability tools.

## 🚀 Features

### Complete AI Pipeline
- **Data Collection & Understanding**: Multi-format support (CSV, JSON, DICOM, NIfTI, Parquet) with comprehensive data validation
- **Data Preprocessing**: Advanced cleaning, imputation, scaling, and encoding with privacy-preserving techniques
- **Exploratory Data Analysis**: Statistical analysis, visualizations, correlation analysis, and automated insights
- **Feature Engineering**: Extraction, selection, transformation with PCA, t-SNE, and advanced techniques
- **Model Development**: Multiple ML/DL algorithms with hyperparameter optimization and automated model selection
- **Comprehensive Evaluation**: Performance metrics, bias analysis, fairness assessment, and detailed reporting
- **Federated Learning**: Privacy-preserving distributed learning with differential privacy and secure aggregation
- **Model Interpretation**: SHAP, LIME, permutation importance, bias analysis, and actionable insights

### Sustainability & Ethics Focus
- **Energy Consumption Tracking**: Monitor and optimize computational resources
- **Carbon Footprint Analysis**: Estimate and minimize environmental impact
- **Bias Detection & Mitigation**: Comprehensive fairness analysis across sensitive attributes
- **Privacy Preservation**: Differential privacy, encryption, and secure computation
- **Green Computing Practices**: Efficient algorithms and resource optimization

### Healthcare-Specific Features
- **Medical Data Support**: DICOM, NIfTI, and healthcare-specific formats
- **Privacy Compliance**: HIPAA-conscious design and privacy-by-design principles
- **Interpretable AI**: Explainable models crucial for healthcare decision-making
- **Federated Learning**: Enable collaborative learning without sharing sensitive patient data

## 📋 Requirements

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies for Advanced Features
```bash
# For neural networks
pip install tensorflow torch torchvision

# For explainability
pip install shap lime

# For medical imaging
pip install pydicom nibabel SimpleITK

# For federated learning
pip install flwr pysyft

# For privacy preservation
pip install cryptography differential-privacy
```

## 🏗️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/TlomMoco/Sustainable_AI_In_Healthcare.git
cd Sustainable_AI_In_Healthcare
```

2. **Install in development mode**:
```bash
pip install -e .
```

3. **Install additional development tools**:
```bash
pip install -e .[dev]
```

## 🚀 Quick Start

### Basic Usage

```python
from src.pipeline import SustainableAIPipeline

# Initialize pipeline with configuration
pipeline = SustainableAIPipeline(config_path="configs/default_config.yaml")

# Run complete pipeline
results = pipeline.run_full_pipeline(data_path="path/to/your/data.csv")

# Get pipeline status
status = pipeline.get_pipeline_status()
print(status)
```

### Step-by-Step Usage

```python
from src.pipeline import SustainableAIPipeline

pipeline = SustainableAIPipeline()

# Step 1: Data Collection and Understanding
data_results = pipeline.collect_and_understand_data("data/healthcare_data.csv")

# Step 2: Data Preprocessing
preprocessing_results = pipeline.preprocess_data()

# Step 3: Exploratory Data Analysis
eda_results = pipeline.perform_eda()

# Step 4: Feature Engineering
feature_results = pipeline.engineer_features()

# Step 5: Model Development
model_results = pipeline.develop_models()

# Step 6: Model Evaluation
evaluation_results = pipeline.evaluate_models()

# Step 7: Federated Learning (Optional)
federated_results = pipeline.implement_federated_learning()

# Step 8: Interpretation and Insights
interpretation_results = pipeline.interpret_and_discuss()
```

### Federated Learning Example

```python
from src.federated_learning.federated_learner import FederatedServer, SKLearnFLClient
from src.federated_learning.federated_learner import ClientConfig, PrivacyPreserver
from sklearn.ensemble import RandomForestClassifier

# Create federated learning configuration
fl_config = {
    'rounds': 10,
    'client_selection': 'random',
    'aggregation_strategy': 'fedavg',
    'privacy': {
        'differential_privacy': True,
        'privacy_budget': 1.0
    }
}

# Initialize server
server = FederatedServer(fl_config)

# Create clients
privacy_preserver = PrivacyPreserver(fl_config['privacy'])

for i in range(3):  # 3 clients
    client_config = ClientConfig(
        client_id=f"client_{i}",
        model_type="random_forest",
        privacy_budget=1.0
    )
    
    client = SKLearnFLClient(
        client_config, 
        RandomForestClassifier,
        privacy_preserver
    )
    
    client.load_data(f"data/client_{i}_data.csv")
    server.register_client(client)

# Run federated learning
fl_results = server.run_federated_learning()
```

## 📊 Pipeline Components

### 1. Data Collection (`src/data_collection/`)
- Multi-format data loading (CSV, JSON, Parquet, DICOM, NIfTI)
- Data validation and quality checks
- Privacy-compliant data handling
- Comprehensive data profiling

### 2. Data Preprocessing (`src/preprocessing/`)
- Advanced data cleaning and transformation
- Missing value imputation strategies
- Outlier detection and handling
- Feature scaling and normalization
- Categorical encoding

### 3. Exploratory Data Analysis (`src/eda/`)
- Statistical analysis and hypothesis testing
- Data visualization and reporting
- Correlation analysis
- Distribution analysis
- Automated insight generation

### 4. Feature Engineering (`src/feature_engineering/`)
- Feature extraction (statistical, temporal, interaction)
- Feature selection (univariate, recursive, LASSO, tree-based)
- Dimensionality reduction (PCA, t-SNE)
- Polynomial and interaction features

### 5. Model Development (`src/model_development/`)
- Multiple ML algorithms (Random Forest, XGBoost, LightGBM, Neural Networks)
- Automated hyperparameter optimization
- Cross-validation and model selection
- Performance tracking and comparison

### 6. Model Evaluation (`src/evaluation/`)
- Comprehensive performance metrics
- Bias and fairness analysis
- Model comparison and ranking
- Visualization and reporting
- Statistical significance testing

### 7. Federated Learning (`src/federated_learning/`)
- Client-server architecture
- Privacy-preserving aggregation
- Differential privacy implementation
- Secure communication protocols
- Convergence monitoring

### 8. Model Interpretation (`src/interpretation/`)
- SHAP values and explanations
- LIME local explanations
- Permutation importance
- Bias and fairness analysis
- Sustainability metrics
- Actionable insights generation

## ⚙️ Configuration

The pipeline is highly configurable through YAML files. See `configs/default_config.yaml` for all available options:

```yaml
# Data Configuration
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  validation_split: 0.2
  test_split: 0.15

# Model Development
models:
  algorithms:
    - name: "random_forest"
      hyperparameters:
        n_estimators: [100, 200, 300]
        max_depth: [10, 20, 30]

# Evaluation
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc"]
  cross_validation_folds: 5

# Federated Learning
federated_learning:
  num_clients: 3
  rounds: 10
  privacy:
    differential_privacy: true
    privacy_budget: 1.0

# Sustainability
sustainability:
  track_energy_consumption: true
  carbon_footprint_estimation: true
```

## 📈 Outputs and Reports

The pipeline generates comprehensive reports and visualizations:

- **Data Quality Reports**: Missing data analysis, outlier detection, statistical summaries
- **EDA Reports**: Distribution plots, correlation heatmaps, statistical insights
- **Model Performance Reports**: Accuracy metrics, confusion matrices, ROC curves
- **Interpretation Reports**: Feature importance, SHAP explanations, bias analysis
- **Sustainability Reports**: Energy consumption, carbon footprint, efficiency metrics
- **Federated Learning Reports**: Training progress, convergence metrics, privacy analysis

## 🔒 Privacy and Security

- **Differential Privacy**: Add calibrated noise to preserve privacy
- **Secure Aggregation**: Federated learning with encrypted communication
- **Data Anonymization**: Remove or anonymize sensitive identifiers
- **Access Control**: Role-based access to sensitive components
- **HIPAA Compliance**: Healthcare data privacy considerations

## 🌱 Sustainability Features

- **Energy Monitoring**: Track computational resources and energy consumption
- **Carbon Footprint**: Estimate and minimize environmental impact
- **Efficient Algorithms**: Optimize for computational efficiency
- **Green Computing**: Best practices for sustainable AI development
- **Resource Optimization**: Memory and CPU usage optimization

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_federated_learning.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

Detailed documentation for each component:

- **API Reference**: Complete function and class documentation
- **Tutorials**: Step-by-step guides for common use cases  
- **Examples**: Jupyter notebooks with real-world examples
- **Configuration Guide**: Complete configuration options
- **Deployment Guide**: Production deployment considerations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Healthcare AI research community
- Open source machine learning libraries
- Privacy-preserving ML research
- Sustainable AI initiatives
- Federated learning frameworks

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example notebooks

## 🔮 Roadmap

- [ ] Advanced privacy-preserving techniques
- [ ] Integration with more healthcare data formats
- [ ] Real-time model monitoring and drift detection
- [ ] Enhanced sustainability metrics
- [ ] Cloud deployment templates
- [ ] Integration with popular ML platforms

---

**Built with ❤️ for sustainable and ethical healthcare AI**
