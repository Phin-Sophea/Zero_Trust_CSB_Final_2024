import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Keep joblib import

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

class ZeroTrustNetworkSecurityDetector:
    def __init__(self, contamination=0.1, random_state=42, output_dir='zero_trust_results'):
        """
        Initialize Zero Trust Network Security Anomaly Detector
        
        Args:
            contamination (float): Proportion of outliers in the dataset
            random_state (int): Seed for reproducibility
            output_dir (str): Directory to save experiment results
        """
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Anomaly Detection Model
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False
        )
        
        # Experiment tracking
        self.feature_names = None
        self.contamination = contamination
        self.random_state = random_state

    def preprocess_data(self, data):
        """
        Preprocess UNSW-NB15 network data for anomaly detection
        
        Args:
            data (pd.DataFrame): Raw network traffic data
        
        Returns:
            np.ndarray: Processed feature matrix
        """
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Define feature types for UNSW-NB15
        numeric_features = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'ct_state_ttl',
            'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
            'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
        ]
        
        categorical_features = [
            'proto', 'service', 'state'
        ]
        
        # Handle missing values in numeric features
        for feature in numeric_features:
            if feature in processed_data.columns:
                processed_data[feature] = processed_data[feature].fillna(
                    processed_data[feature].median()
                )
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in processed_data.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                processed_data[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
                    processed_data[feature].astype(str)
                )
        
        # Combine features
        encoded_categorical_features = [f'{f}_encoded' for f in categorical_features]
        features_to_use = numeric_features + encoded_categorical_features
        features_to_use = [f for f in features_to_use if f in processed_data.columns]
        self.feature_names = features_to_use
        
        # Scale numeric features
        X = processed_data[features_to_use].values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled

    def train(self, training_data):
        """
        Train the anomaly detection model
        
        Args:
            training_data (pd.DataFrame): Training network traffic data
        """
        # Preprocess the training data
        X_train = self.preprocess_data(training_data)
        
        # Fit the Isolation Forest detector
        self.detector.fit(X_train)
        
        print("Model training completed successfully.")

    def detect_anomalies(self, data):
        """
        Detect network anomalies
        
        Args:
            data (pd.DataFrame): Network traffic data to analyze
        
        Returns:
            np.ndarray: Anomaly predictions (-1 for anomalies, 1 for normal)
        """
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Predict anomalies
        return self.detector.predict(X)

    def evaluate_model(self, test_data, true_labels):
        """
        Evaluate anomaly detection performance
        
        Args:
            test_data (pd.DataFrame): Test network data
            true_labels (pd.Series): Ground truth labels
        """
        # Detect anomalies
        predictions = self.detect_anomalies(test_data)
        
        # Convert predictions to binary labels (UNSW-NB15 uses 0 for normal, 1 for attack)
        binary_predictions = [1 if p == -1 else 0 for p in predictions]
        binary_true = true_labels.astype(int).values
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(binary_true, binary_predictions),
            'Precision': precision_score(binary_true, binary_predictions),
            'Recall': recall_score(binary_true, binary_predictions),
            'F1 Score': f1_score(binary_true, binary_predictions)
        }
        
        # Print and save classification report
        report = classification_report(binary_true, binary_predictions)
        print("Anomaly Detection Performance:")
        print(report)
        
        # Save classification report
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion Matrix Visualization
        cm = confusion_matrix(binary_true, binary_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Zero Trust Anomaly Detection Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save detailed results
        results_df = test_data.copy()
        results_df['true_label'] = true_labels
        results_df['predicted_label'] = ['attack' if p == -1 else 'normal' for p in predictions]
        results_df.to_csv(os.path.join(self.output_dir, 'anomaly_detection_results.csv'), index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df.to_csv(os.path.join(self.output_dir, 'performance_metrics.csv'))
        
        return metrics

    def save_model(self, filename='zero_trust_model.pkl'):
        """
        Save the trained model and related preprocessing components
        
        Args:
            filename (str): Name of the file to save the model
        """
        # Prepare a dictionary with all necessary components
        model_data = {
            'detector': self.detector,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        # Create full file path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the model
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a previously trained model
        
        Args:
            filepath (str): Path to the saved model file
        
        Returns:
            ZeroTrustNetworkSecurityDetector: Loaded model instance
        """
        # Load the model data
        model_data = joblib.load(filepath)
        
        # Create a new instance of the detector
        detector = cls()
        
        # Restore model components
        detector.detector = model_data['detector']
        detector.scaler = model_data['scaler']
        detector.label_encoders = model_data['label_encoders']
        detector.feature_names = model_data['feature_names']
        
        return detector

def main():
    # Load UNSW-NB15 dataset
    data_path = 'UNSW_NB15_training-set.csv'
    network_data = pd.read_csv(data_path)
    
    # Separate features and labels
    features = network_data.drop(['label', 'attack_cat'], axis=1)
    labels = network_data['label']
    
    # Split data into training (normal traffic only) and testing sets
    normal_mask = labels == 0
    normal_data = features[normal_mask]
    
    # Initialize and train Zero Trust detector
    zt_detector = ZeroTrustNetworkSecurityDetector(
        contamination=0.1, 
        output_dir='zero_trust_results'
    )
    
    # Train on normal data only
    zt_detector.train(normal_data)
    
    # Save the trained model
    zt_detector.save_model()
    
    # Detect and evaluate anomalies on the full dataset
    performance_metrics = zt_detector.evaluate_model(features, labels)
    
    print("Experiment completed. Results saved in 'zero_trust_results' directory.")

if __name__ == "__main__":
    main()