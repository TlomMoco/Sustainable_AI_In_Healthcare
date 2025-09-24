#!/usr/bin/env python3
"""
Simple test script to verify the Sustainable AI Healthcare Pipeline works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import SustainableAIPipeline


def create_test_data():
    """Create a simple test dataset."""
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 200
    n_features = 8
    
    data = np.random.randn(n_samples, n_features)
    
    # Create meaningful column names
    columns = [
        'age', 'bmi', 'blood_pressure', 'cholesterol',
        'glucose', 'heart_rate', 'exercise_hours', 'sleep_hours'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Make data more realistic
    df['age'] = np.clip(df['age'] * 15 + 50, 20, 80).astype(int)
    df['bmi'] = np.clip(df['bmi'] * 5 + 25, 18, 40)
    df['blood_pressure'] = np.clip(df['blood_pressure'] * 20 + 120, 90, 180)
    
    # Add target variable (binary classification)
    target = (df['age'] > 60) & (df['bmi'] > 30) | (df['blood_pressure'] > 140)
    df['risk'] = target.astype(int)
    
    # Add some missing values
    missing_mask = np.random.choice([True, False], size=(n_samples, 2), p=[0.1, 0.9])
    df.loc[missing_mask[:, 0], 'cholesterol'] = np.nan
    df.loc[missing_mask[:, 1], 'glucose'] = np.nan
    
    return df


def test_pipeline():
    """Test the complete pipeline."""
    print("🧪 Testing Sustainable AI Healthcare Pipeline...")
    
    # Create test data
    print("📊 Creating test data...")
    test_data = create_test_data()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Initialize pipeline
        print("🔧 Initializing pipeline...")
        pipeline = SustainableAIPipeline()
        
        # Test step by step
        print("\n📋 Testing individual steps:")
        
        # Step 1: Data Collection
        print("  1. Data Collection and Understanding...")
        try:
            data_results = pipeline.collect_and_understand_data(temp_file)
            print(f"     ✅ Success: Data shape {data_results.get('data_shape', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 2: Preprocessing
        print("  2. Data Preprocessing...")
        try:
            prep_results = pipeline.preprocess_data()
            print(f"     ✅ Success: {prep_results.get('original_shape', 'N/A')} → {prep_results.get('processed_shape', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 3: EDA
        print("  3. Exploratory Data Analysis...")
        try:
            eda_results = pipeline.perform_eda()
            print(f"     ✅ Success: {eda_results.get('status', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 4: Feature Engineering
        print("  4. Feature Engineering...")
        try:
            fe_results = pipeline.engineer_features()
            print(f"     ✅ Success: {fe_results.get('original_features', 'N/A')} → {fe_results.get('final_features', 'N/A')} features")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 5: Model Development
        print("  5. Model Development...")
        try:
            model_results = pipeline.develop_models()
            models_trained = model_results.get('models_trained', [])
            print(f"     ✅ Success: {len(models_trained)} models trained")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 6: Evaluation
        print("  6. Model Evaluation...")
        try:
            eval_results = pipeline.evaluate_models()
            print(f"     ✅ Success: {eval_results.get('status', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 7: Federated Learning
        print("  7. Federated Learning...")
        try:
            fl_results = pipeline.implement_federated_learning()
            print(f"     ✅ Success: {fl_results.get('status', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        # Step 8: Interpretation
        print("  8. Model Interpretation...")
        try:
            interp_results = pipeline.interpret_and_discuss()
            print(f"     ✅ Success: {interp_results.get('status', 'N/A')}")
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return False
        
        print("\n🎉 All tests passed successfully!")
        
        # Show pipeline status
        status = pipeline.get_pipeline_status()
        print("\n📊 Final Pipeline Status:")
        for step, completed in status.items():
            status_symbol = "✅" if completed else "❌"
            print(f"  {status_symbol} {step.replace('_', ' ').title()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


def test_full_pipeline():
    """Test the complete pipeline in one go."""
    print("\n🚀 Testing full pipeline execution...")
    
    # Create test data
    test_data = create_test_data()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Initialize and run full pipeline
        pipeline = SustainableAIPipeline()
        results = pipeline.run_full_pipeline(temp_file)
        
        print("✅ Full pipeline execution successful!")
        print(f"   Steps completed: {len([r for r in results.values() if isinstance(r, dict) and r.get('status') == 'completed'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False
        
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass


if __name__ == "__main__":
    print("🏥 Sustainable AI in Healthcare - Pipeline Test Suite")
    print("=" * 60)
    
    # Test individual steps
    step_test_passed = test_pipeline()
    
    # Test full pipeline
    full_test_passed = test_full_pipeline()
    
    print("\n" + "=" * 60)
    if step_test_passed and full_test_passed:
        print("🎉 ALL TESTS PASSED! The pipeline is working correctly.")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        exit(1)