import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys

# Add the directory containing Zero_Trust_Final.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ZeroTrustNetworkSecurityDetector from the first script
from Zero_Trust_Final import ZeroTrustNetworkSecurityDetector

# Cached data loading function
@st.cache_data
def load_data(file):
    """Load data from uploaded file"""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def main():
    st.title("Zero Trust Network Security Anomaly Detector")
    
    # Sidebar for model operation selection
    st.sidebar.header("Model Options")
    model_option = st.sidebar.radio(
        "Choose Operation", 
        ["Train New Model", "Load Existing Model"]
    )

    if model_option == "Train New Model":
        # Dataset upload for training
        st.sidebar.subheader("Train New Model")
        train_file = st.sidebar.file_uploader(
            "Upload Training Dataset", 
            type=['csv']
        )
        
        if train_file is not None:
            # Load training data
            train_data = load_data(train_file)
            
            if train_data is not None:
                st.write("Training Data Preview:")
                st.dataframe(train_data.head())
                
                # Contamination rate slider
                contamination = st.sidebar.slider(
                    "Anomaly Contamination Rate", 
                    min_value=0.01, 
                    max_value=0.5, 
                    value=0.1, 
                    step=0.01
                )
                
                # Train button
                if st.sidebar.button("Train Model"):
                    try:
                        # Prepare data (assuming binary classification)
                        features = train_data.drop(['label', 'attack_cat'], axis=1)
                        
                        # Initialize and train detector
                        zt_detector = ZeroTrustNetworkSecurityDetector(
                            contamination=contamination,
                            output_dir='zero_trust_results'
                        )
                        
                        # Train on data
                        zt_detector.train(features)
                        
                        # Save the model (now using .pkl)
                        zt_detector.save_model('zero_trust_model.pkl')
                        
                        st.success("Model trained and saved successfully!")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

    else:  # Load Existing Model
        st.sidebar.subheader("Load Existing Model")
        # Model file upload
        model_file = st.sidebar.file_uploader(
            "Upload Trained Model", 
            type=['pkl']  # Updated to .pkl
        )
        
        # Detection dataset upload
        detect_file = st.sidebar.file_uploader(
            "Upload Detection Dataset", 
            type=['csv']
        )
        
        if model_file is not None and detect_file is not None:
            try:
                # Temporarily save uploaded model
                with open(os.path.join('zero_trust_results', model_file.name), 'wb') as f:
                    f.write(model_file.getbuffer())
                
                # Load the model (now using .pkl)
                model_path = os.path.join('zero_trust_results', model_file.name)
                zt_detector = ZeroTrustNetworkSecurityDetector.load_model(model_path)
                
                # Load detection data
                detect_data = load_data(detect_file)
                
                if detect_data is not None:
                    # Prepare features (excluding labels for detection)
                    features = detect_data.drop(['label', 'attack_cat'], axis=1)
                    
                    # Detect anomalies
                    predictions = zt_detector.detect_anomalies(features)
                    
                    # Display results
                    st.subheader("Anomaly Detection Results")
                    
                    # Count and display anomalies
                    anomaly_count = list(predictions).count(-1)
                    st.write(f"Detected {anomaly_count} anomalies out of {len(predictions)} samples")
                    
                    # If ground truth labels exist
                    if 'label' in detect_data.columns:
                        # Evaluate performance
                        performance = zt_detector.evaluate_model(features, detect_data['label'])
                        
                        # Display confusion matrix
                        st.subheader("Confusion Matrix")
                        cm_file = os.path.join('zero_trust_results', 'confusion_matrix.png')
                        if os.path.exists(cm_file):
                            st.image(cm_file)
                        
                        # Display metrics
                        st.subheader("Performance Metrics")
                        metrics_file = os.path.join('zero_trust_results', 'performance_metrics.csv')
                        if os.path.exists(metrics_file):
                            metrics_df = pd.read_csv(metrics_file, index_col=0)
                            st.dataframe(metrics_df)
            except Exception as e:
                st.error(f"Error in model loading or detection: {e}")

if __name__ == "__main__":
    main()