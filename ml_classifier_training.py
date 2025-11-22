#!/usr/bin/env python3
"""
Machine Learning Classifier for YSO Variability

Trains multiple ML models on synthetic YSO lightcurve data to classify variability types.
Supports Random Forest, XGBoost, and Neural Networks with cross-validation and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns

from COMPLETE_ANALYSIS_FRAMEWORK import ZTFLightcurve, NEOWISELightcurve, VariabilityAnalyzer
from mock_data_generator import MockVariabilityGenerator, RealisticYSODataset


class YSOFeatureExtractor:
    """Extract features from YSO lightcurves for ML classification"""
    
    @staticmethod
    def extract_features(ztf: ZTFLightcurve, 
                        neowise: Optional[NEOWISELightcurve] = None,
                        include_advanced: bool = True) -> np.ndarray:
        """
        Extract comprehensive feature set from lightcurves
        
        Basic features (6):
        1. Optical variability amplitude (std magnitude)
        2. Optical variability timescale (time baseline)
        3. Optical mean magnitude
        4. Signal-to-noise ratio
        5. Observation cadence
        6. Number of observations normalized
        
        Advanced features (8+):
        7. Skewness of magnitude distribution
        8. Kurtosis of magnitude distribution
        9. Autocorrelation at lag 1
        10. Slope of linear trend
        11. IR variability amplitude (W1)
        12. IR variability amplitude (W2)
        13. IR/Optical amplitude ratio
        14. Median observation error
        15. Range of magnitudes
        
        Returns:
            Feature array (n_features,)
        """
        
        features = []
        
        # Basic optical statistics
        opt_stats = VariabilityAnalyzer.calculate_statistics(ztf.mag, ztf.magerr)
        
        features.append(opt_stats['std_magnitude'])  # 1. Optical amplitude
        features.append(ztf.time_baseline)           # 2. Time baseline
        features.append(opt_stats['mean_magnitude']) # 3. Mean magnitude
        features.append(opt_stats['snr'])            # 4. SNR
        features.append(ztf.mean_cadence)            # 5. Cadence
        features.append(np.log10(ztf.n_observations))  # 6. Log(N_obs)
        
        if include_advanced:
            from scipy.stats import skew, kurtosis
            
            # Magnitude distribution properties
            features.append(skew(ztf.mag))          # 7. Skewness
            features.append(kurtosis(ztf.mag))      # 8. Kurtosis
            
            # Autocorrelation at lag 1
            if len(ztf.mag) > 2:
                mag_normalized = (ztf.mag - np.mean(ztf.mag)) / np.std(ztf.mag)
                autocorr = np.corrcoef(mag_normalized[:-1], mag_normalized[1:])[0, 1]
                features.append(autocorr)           # 9. Autocorr(lag=1)
            else:
                features.append(0)
            
            # Trend analysis
            trend = VariabilityAnalyzer.detect_trend(ztf.mjd, ztf.mag, ztf.magerr)
            if trend:
                features.append(trend[0])           # 10. Trend slope
            else:
                features.append(0)
            
            # IR properties if available
            if neowise:
                ir_stats_w1 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w1, neowise.magerr_w1)
                ir_stats_w2 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w2, neowise.magerr_w2)
                
                features.append(ir_stats_w1['std_magnitude'])  # 11. W1 amplitude
                features.append(ir_stats_w2['std_magnitude'])  # 12. W2 amplitude
                
                # IR/Optical ratio
                if opt_stats['std_magnitude'] > 0:
                    ratio = (ir_stats_w1['std_magnitude'] + ir_stats_w2['std_magnitude']) / (2 * opt_stats['std_magnitude'])
                else:
                    ratio = 0
                features.append(ratio)              # 13. IR/Opt ratio
            else:
                features.extend([0, 0, 0])
            
            features.append(opt_stats['median_error'])  # 14. Median error
            features.append(opt_stats['range'])         # 15. Magnitude range
        
        return np.array(features)


class YSOMLClassifier:
    """Train and evaluate ML models for YSO classification"""
    
    def __init__(self, output_dir: str = './ml_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.results = {}
    
    def prepare_data(self, ztf_lcs: List[ZTFLightcurve],
                    neowise_lcs: Optional[List[NEOWISELightcurve]] = None,
                    labels: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
        """
        Prepare training and test data
        
        Args:
            ztf_lcs: List of ZTF lightcurves
            neowise_lcs: Optional list of NEOWISE lightcurves
            labels: Variability type labels for each lightcurve
            test_size: Fraction for test set
            random_state: Random seed
        
        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        
        print(f"\nPreparing data from {len(ztf_lcs)} lightcurves...")
        
        # Extract features
        X = []
        for i, ztf in enumerate(ztf_lcs):
            neowise = neowise_lcs[i] if neowise_lcs else None
            features = YSOFeatureExtractor.extract_features(ztf, neowise)
            X.append(features)
        
        X = np.array(X)
        print(f"Feature matrix shape: {X.shape}")
        
        # Generate labels if not provided
        if labels is None:
            var_types = ['stable', 'periodic', 'irregular', 'burst', 'linear_trend']
            labels = [var_types[i % len(var_types)] for i in range(len(ztf_lcs))]
        
        y = self.label_encoder.fit_transform(labels)
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Feature names for interpretation
        self.feature_names = [
            'opt_amplitude', 'time_baseline', 'mean_mag', 'snr', 'cadence', 'log_n_obs',
            'skewness', 'kurtosis', 'autocorr_lag1', 'trend_slope',
            'ir_w1_amplitude', 'ir_w2_amplitude', 'ir_opt_ratio', 'median_error', 'mag_range'
        ]
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100) -> Dict:
        """Train Random Forest classifier"""
        
        print("\n" + "=" * 70)
        print("TRAINING: Random Forest Classifier")
        print("=" * 70)
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.results['RandomForest'] = {'feature_importance': feature_importance}
        
        return rf
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train Gradient Boosting classifier"""
        
        print("\n" + "=" * 70)
        print("TRAINING: Gradient Boosting Classifier")
        print("=" * 70)
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=1
        )
        
        gb.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.results['GradientBoosting'] = {'feature_importance': feature_importance}
        
        return gb
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional[Dict]:
        """Train XGBoost classifier (if available)"""
        
        try:
            import xgboost as xgb
            
            print("\n" + "=" * 70)
            print("TRAINING: XGBoost Classifier")
            print("=" * 70)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            xgb_model.fit(X_train, y_train)
            self.models['XGBoost'] = xgb_model
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
            
            self.results['XGBoost'] = {'feature_importance': feature_importance}
            
            return xgb_model
        
        except ImportError:
            print("\n⚠ XGBoost not installed. Install with: pip install xgboost")
            return None
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional:
        """Train Neural Network classifier (if TensorFlow available)"""
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            print("\n" + "=" * 70)
            print("TRAINING: Neural Network Classifier")
            print("=" * 70)
            
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(n_features,)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(n_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            self.models['NeuralNetwork'] = model
            self.results['NeuralNetwork'] = {'history': history}
            
            print("✓ Neural Network training completed")
            
            return model
        
        except ImportError:
            print("\n⚠ TensorFlow not installed. Install with: pip install tensorflow")
            return None
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all trained models"""
        
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        evaluation_results = []
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            evaluation_results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Save detailed report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            self.results[model_name]['classification_report'] = report
            self.results[model_name]['confusion_matrix'] = cm
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm, model_name)
        
        eval_df = pd.DataFrame(evaluation_results).sort_values('F1-Score', ascending=False)
        print("\n" + eval_df.to_string(index=False))
        
        return eval_df
    
    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot confusion matrix"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name}.png', dpi=150)
        plt.close()
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """Perform cross-validation on all models"""
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION ANALYSIS")
        print("=" * 70)
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = []
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            print(f"  CV Scores: {scores}")
            print(f"  Mean:      {scores.mean():.4f} (+/- {scores.std():.4f})")
            
            cv_results.append({
                'Model': model_name,
                'Mean_CV_Score': scores.mean(),
                'Std_CV_Score': scores.std(),
                'Min_CV_Score': scores.min(),
                'Max_CV_Score': scores.max()
            })
        
        return pd.DataFrame(cv_results)
    
    def save_models(self):
        """Save trained models to disk"""
        
        print("\n" + "=" * 70)
        print("SAVING MODELS")
        print("=" * 70)
        
        for model_name, model in self.models.items():
            if model_name != 'NeuralNetwork':
                filepath = self.output_dir / f'{model_name.lower()}_model.pkl'
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ Saved {model_name} to {filepath}")
            else:
                filepath = self.output_dir / f'neural_network_model.h5'
                model.save(filepath)
                print(f"✓ Saved Neural Network to {filepath}")
        
        # Save scalers
        scaler_path = self.output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        print(f"✓ Saved StandardScaler to {scaler_path}")
        
        # Save label encoder
        le_path = self.output_dir / 'label_encoder.pkl'
        with open(le_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved LabelEncoder to {le_path}")
    
    def generate_report(self, eval_df: pd.DataFrame):
        """Generate comprehensive report"""
        
        report_path = self.output_dir / 'classification_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("YSO ML CLASSIFIER TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(eval_df.to_string(index=False) + "\n\n")
            
            f.write("FEATURE NAMES\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(self.feature_names, 1):
                f.write(f"{i:2d}. {name}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n✓ Report saved to {report_path}")


def main():
    """Complete ML training pipeline"""
    
    print("\n" + "=" * 80)
    print("YSO MACHINE LEARNING CLASSIFIER TRAINING")
    print("=" * 80)
    
    # Step 1: Generate training data
    print("\nSTEP 1: Generating synthetic training data...")
    ztf_lcs, neowise_lcs = MockVariabilityGenerator.generate_dataset(
        n_sources=200,
        variability_distribution={
            'stable': 0.20,
            'periodic': 0.20,
            'irregular': 0.35,
            'burst': 0.15,
            'linear_trend': 0.10
        }
    )
    
    print(f"✓ Generated {len(ztf_lcs)} lightcurves")
    
    # Step 2: Initialize classifier
    print("\nSTEP 2: Initializing ML classifier...")
    classifier = YSOMLClassifier(output_dir='./ml_results')
    
    # Step 3: Prepare data
    print("\nSTEP 3: Preparing training/test data...")
    X_train, X_test, y_train, y_test, feature_names = classifier.prepare_data(
        ztf_lcs, neowise_lcs
    )
    
    # Step 4: Train models
    print("\nSTEP 4: Training models...")
    classifier.train_random_forest(X_train, y_train)
    classifier.train_gradient_boosting(X_train, y_train)
    classifier.train_xgboost(X_train, y_train)
    classifier.train_neural_network(X_train, y_train)
    
    # Step 5: Evaluate models
    print("\nSTEP 5: Evaluating models...")
    eval_df = classifier.evaluate_models(X_test, y_test)
    
    # Step 6: Cross-validation
    print("\nSTEP 6: Performing cross-validation...")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_df = classifier.cross_validate(X_all, y_all)
    
    # Step 7: Save results
    print("\nSTEP 7: Saving models and results...")
    classifier.save_models()
    classifier.generate_report(eval_df)
    
    # Save evaluation and CV results
    eval_df.to_csv(classifier.output_dir / 'evaluation_results.csv', index=False)
    cv_df.to_csv(classifier.output_dir / 'cross_validation_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✓ ML CLASSIFIER TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {classifier.output_dir}")
    print(f"  - Trained models (PKL/H5)")
    print(f"  - Feature importance plots")
    print(f"  - Confusion matrices")
    print(f"  - Evaluation and CV results (CSV)")
    print(f"  - Comprehensive report (TXT)")


if __name__ == '__main__':
    main()
