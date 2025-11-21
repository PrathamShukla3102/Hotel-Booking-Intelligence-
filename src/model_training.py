import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

import lightgbm as lgb
import joblib
import warnings
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class HotelCancellationModel:
    """
    Complete ML pipeline for hotel booking cancellation prediction
    """

    def __init__(self, data_path='/Users/apple/Documents/hotel-booking-intelligence/data/processed/hotels_bookings_processed.csv'):
        """Initialize with cleaned data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self):
        """Load preprocessed data"""
        print("Loading cleaned data...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f" Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            print(f" Error loading data: {str(e)}")
            return None

    def feature_engineering(self):
        """Create advanced features for modeling"""
        print("\n Engineering features...")

        # Make a copy
        df = self.df.copy()

        # 1. Temporal Features
        print("   Creating temporal features...")

        # Month encoding
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)

        # Season
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Fall

        df['season'] = df['arrival_month_num'].apply(get_season)

        # Peak season (June-August)
        df['is_peak_season'] = df['arrival_month_num'].isin([6, 7, 8]).astype(int)

        # Quarter
        df['quarter'] = df['arrival_month_num'].apply(lambda x: (x - 1) // 3 + 1)

        # Ensure lead_time is non-negative as it represents days
        df['lead_time'] = df['lead_time'].clip(lower=0)

        # Lead time categories
        df['lead_time_bucket'] = pd.cut(
            df['lead_time'],
            bins=[0, 7, 30, 90, 180, 365, 1000],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True # To include lead_time = 0 in the first bin
        ).astype(float).fillna(-1).astype(int) # Convert to float before fillna to handle Categorical type issue

        # 2. Guest Features
        print("   Creating guest features...")

        # Total guests
        df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)

        # Has children
        df['has_children'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

        # Guest type
        def categorize_guests(row):
            if row['total_guests'] == 1:
                return 0  # Solo
            elif row['total_guests'] == 2 and row['children'] == 0 and row['babies'] == 0:
                return 1  # Couple
            elif row['children'] > 0 or row['babies'] > 0:
                return 2  # Family
            else:
                return 3  # Group

        df['guest_type'] = df.apply(categorize_guests, axis=1)

        # 3. Stay Features
        print("   Creating stay features...")

        # Total nights
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

        # Weekend ratio
        df['weekend_ratio'] = df['stays_in_weekend_nights'] / (df['total_nights'] + 1)

        # Stay duration category
        df['stay_duration_cat'] = pd.cut(
            df['total_nights'],
            bins=[0, 1, 3, 7, 30],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(-1).astype(int) # Convert to float before fillna to handle Categorical type issue

        # 4. Financial Features
        print(" Creating financial features...")

        # Total revenue
        df['total_revenue'] = df['adr'] * df['total_nights']

        # Revenue per guest
        df['revenue_per_guest'] = df['total_revenue'] / (df['total_guests'] + 1)

        # ADR per guest
        df['adr_per_guest'] = df['adr'] / (df['total_guests'] + 1)

        # Price category (0-3)
        df['price_category'] = pd.qcut(
            df['adr'],
            q=4,
            labels=[0, 1, 2, 3],
            duplicates='drop'
        )

        # 5. Booking Behavior Features
        print(" Creating booking behavior features...")

        # Cancellation history ratio
        df['cancellation_ratio'] = df['previous_cancellations'] / (
            df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1
        )

        # Has booking changes
        df['has_changes'] = (df['booking_changes'] > 0).astype(int)

        # Has special requests
        df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)

        # Room mismatch
        df['room_mismatch'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

        # 6. Risk Indicators
        print(" Creating risk indicators...")

        # Long lead time
        df['long_lead_time'] = (df['lead_time'] > 180).astype(int)

        # No deposit flag
        df['no_deposit'] = (df['deposit_type'] == 'No Deposit').astype(int)

        # High modification count
        df['high_modifications'] = (df['booking_changes'] > 2).astype(int)

        # Previous cancellation flag
        df['has_prev_cancellation'] = (df['previous_cancellations'] > 0).astype(int)

        # Waitlist flag
        df['was_on_waitlist'] = (df['days_in_waiting_list'] > 0).astype(int)

        # 7. Interaction Features
        print("Creating interaction features...")

        # Lead time × deposit type interaction
        df['lead_deposit_interaction'] = df['lead_time'] * df['no_deposit']

        # ADR × lead time
        df['adr_lead_interaction'] = df['adr'] * df['lead_time']

        # Special requests × repeat guest
        df['requests_repeat_interaction'] = (
            df['total_of_special_requests'] * df['is_repeated_guest']
        )

        print(f"Feature engineering complete. Total features: {len(df.columns)}")

        self.df = df
        return df

    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n Preparing features for modeling...")

        # Define feature columns
        # Numerical features
        numerical_features = [
            'lead_time', 'arrival_date_year', 'arrival_date_week_number',
            'arrival_date_day_of_month', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'babies',
            'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes',
            'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
            'total_of_special_requests',
            # Engineered features
            'arrival_month_num', 'season', 'is_peak_season', 'quarter',
            'lead_time_bucket', 'total_guests', 'has_children', 'guest_type',
            'total_nights', 'weekend_ratio', 'stay_duration_cat',
            'total_revenue', 'revenue_per_guest', 'adr_per_guest',
            'cancellation_ratio', 'has_changes', 'has_special_requests',
            'room_mismatch', 'long_lead_time', 'no_deposit',
            'high_modifications', 'has_prev_cancellation', 'was_on_waitlist',
            'lead_deposit_interaction', 'adr_lead_interaction',
            'requests_repeat_interaction'
        ]

        # Categorical features to encode
        categorical_features = [
            'hotel', 'meal', 'country', 'market_segment',
            'distribution_channel', 'reserved_room_type',
            'assigned_room_type', 'deposit_type', 'customer_type'
        ]

        # Create a copy for processing
        df_model = self.df.copy()

        # Handle missing values in numerical features
        for col in numerical_features:
            if col in df_model.columns:
                df_model[col] = df_model[col].fillna(0)

        # Encode categorical features
        print(" Encoding categorical features...")
        for col in categorical_features:
            if col in df_model.columns:
                le = LabelEncoder()
                df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
                self.label_encoders[col] = le

        # Select features for modeling
        encoded_cats = [col + '_encoded' for col in categorical_features if col in df_model.columns]
        available_numerical = [col for col in numerical_features if col in df_model.columns]

        feature_cols = available_numerical + encoded_cats

        # Remove features if they exist in feature_cols but cause issues
        if 'price_category' in df_model.columns:
            df_model['price_category'] = df_model['price_category'].fillna(0).astype(int)
            feature_cols.append('price_category')

        # Prepare X and y
        X = df_model[feature_cols].copy()
        y = df_model['is_canceled'].copy()

        # Store feature names
        self.feature_names = feature_cols

        print(f" Prepared {len(feature_cols)} features for modeling")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        print(f"   Cancellation rate: {y.mean()*100:.2f}%")

        return X, y

    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Split data and scale features"""
        print(f"\n Splitting data (train: {(1-test_size)*100:.0f}%, test: {test_size*100:.0f}%)...")

        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f" Train set: {len(self.X_train):,} samples")
        print(f" Test set: {len(self.X_test):,} samples")

        # Scale features
        print("\n Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(" Features scaled using StandardScaler")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def handle_class_imbalance(self, method='smote'):
        """Handle class imbalance using SMOTE"""
        print(f"\n⚖️ Handling class imbalance using {method.upper()}...")

        print(f"   Before SMOTE: {dict(pd.Series(self.y_train).value_counts())}")

        if method == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )

        print(f"   After SMOTE: {dict(pd.Series(self.y_train_balanced).value_counts())}")
        print(" Class imbalance handled")

        return self.X_train_balanced, self.y_train_balanced

    def train_logistic_regression(self):
        """Train Logistic Regression (Baseline)"""
        print("\n" + "="*80)
        print(" Training Logistic Regression (Baseline Model)")
        print("="*80)

        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )

        model.fit(self.X_train_balanced, self.y_train_balanced)

        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # Store model and results
        self.models['Logistic_Regression'] = model
        self.results['Logistic_Regression'] = self._evaluate_model(
            y_pred, y_pred_proba, 'Logistic Regression'
        )

        print(" Logistic Regression trained successfully")

        return model

    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*80)
        print(" Training Random Forest Classifier")
        print("="*80)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(self.X_train_balanced, self.y_train_balanced)

        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # Store model and results
        self.models['Random_Forest'] = model
        self.results['Random_Forest'] = self._evaluate_model(
            y_pred, y_pred_proba, 'Random Forest'
        )

        # Feature importance
        self._plot_feature_importance(model, 'Random_Forest')

        print(" Random Forest trained successfully")

        return model

    def train_xgboost(self):
        """Train XGBoost Classifier"""
        print("\n" + "="*80)
        print(" Training XGBoost Classifier")
        print("="*80)

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train_balanced == 0).sum() / (self.y_train_balanced == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        model.fit(
            self.X_train_balanced,
            self.y_train_balanced,
            eval_set=[(self.X_test_scaled, self.y_test)],
            verbose=False
        )

        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # Store model and results
        self.models['XGBoost'] = model
        self.results['XGBoost'] = self._evaluate_model(
            y_pred, y_pred_proba, 'XGBoost'
        )

        # Feature importance
        self._plot_feature_importance(model, 'XGBoost')

        print(" XGBoost trained successfully")

        return model

    def train_lightgbm(self):
        """Train LightGBM Classifier"""
        print("\n" + "="*80)
        print("⚡ Training LightGBM Classifier")
        print("="*80)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(
            self.X_train_balanced,
            self.y_train_balanced,
            eval_set=[(self.X_test_scaled, self.y_test)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # Store model and results
        self.models['LightGBM'] = model
        self.results['LightGBM'] = self._evaluate_model(
            y_pred, y_pred_proba, 'LightGBM'
        )

        # Feature importance
        self._plot_feature_importance(model, 'LightGBM')

        print(" LightGBM trained successfully")

        return model

    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        print(f"\n Evaluating {model_name}...")

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        # Print metrics
        print(f"\n{'Metric':<20} {'Score':<10}")
        print("-" * 30)
        print(f"{'Accuracy':<20} {accuracy:.4f}")
        print(f"{'Precision':<20} {precision:.4f}")
        print(f"{'Recall':<20} {recall:.4f}")
        print(f"{'F1-Score':<20} {f1:.4f}")
        print(f"{'ROC-AUC':<20} {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n Confusion Matrix:")
        print(cm)

        # Classification report
        print(f"\n Classification Report:")
        print(classification_report(self.y_test, y_pred,
                                   target_names=['Not Canceled', 'Canceled']))

        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }

        return results

    def _plot_feature_importance(self, model, model_name, top_n=20):
        """Plot feature importance"""
        print(f"\n Plotting feature importance for {model_name}...")

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print(" Model doesn't have feature_importances_ attribute")
            return

        # Create dataframe
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = feature_imp.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}',
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        Path('models/plots').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'models/plots/{model_name}_feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Feature importance plot saved: models/plots/{model_name}_feature_importance.png")

        # Save top features to CSV
        feature_imp.to_csv(f'models/plots/{model_name}_feature_importance.csv', index=False)
        print(f" Feature importance saved: models/plots/{model_name}_feature_importance.csv")

        return feature_imp

    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print(" MODEL COMPARISON")
        print("="*80)

        # Create comparison dataframe
        comparison = pd.DataFrame({
            model: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            }
            for model, results in self.results.items()
        }).T

        print("\n" + comparison.to_string())

        # Identify best model based on F1-Score
        best_model_name = comparison['F1-Score'].idxmax()
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {comparison.loc[best_model_name, 'F1-Score']:.4f}")
        print(f"   ROC-AUC: {comparison.loc[best_model_name, 'ROC-AUC']:.4f}")

        # Plot comparison
        self._plot_model_comparison(comparison)

        # Save comparison
        comparison.to_csv('models/model_comparison.csv')
        print("\n Model comparison saved: models/model_comparison.csv")

        return comparison

    def _plot_model_comparison(self, comparison):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = comparison[metric].sort_values(ascending=False)
            bars = ax.barh(range(len(values)), values.values, color=colors[idx], alpha=0.7)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(values.index)
            ax.set_xlabel('Score', fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])

            # Add value labels
            for i, v in enumerate(values.values):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        # Remove extra subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('models/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Model comparison plot saved: models/plots/model_comparison.png")

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("\n Plotting ROC curves...")

        plt.figure(figsize=(10, 8))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (model_name, results) in enumerate(self.results.items()):
            y_pred_proba = results['predictions_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = results['roc_auc']

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})',
                    linewidth=2, color=colors[idx % len(colors)])

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig('models/plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" ROC curves saved: models/plots/roc_curves.png")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\n📊 Plotting confusion matrices...")

        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')

        axes = axes.ravel()

        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_matrix'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Not Canceled', 'Canceled'],
                       yticklabels=['Not Canceled', 'Canceled'],
                       cbar=False)
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11)
            axes[idx].set_xlabel('Predicted Label', fontsize=11)

        plt.tight_layout()
        plt.savefig('models/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(" Confusion matrices saved: models/plots/confusion_matrices.png")

    def save_models(self):
        """Save all trained models"""
        print("\n Saving models...")

        Path('models/saved').mkdir(parents=True, exist_ok=True)

        # Save all models
        for model_name, model in self.models.items():
            joblib.dump(model, f'models/saved/{model_name}.pkl')
            print(f"    Saved: {model_name}.pkl")

        # Save best model separately
        joblib.dump(self.best_model, 'models/saved/best_model.pkl')
        print(f"    Saved: best_model.pkl ({self.best_model_name})")

        # Save scaler
        joblib.dump(self.scaler, 'models/saved/scaler.pkl')
        print(f"    Saved: scaler.pkl")

        # Save label encoders
        joblib.dump(self.label_encoders, 'models/saved/label_encoders.pkl')
        print(f"    Saved: label_encoders.pkl")

        # Save feature names
        joblib.dump(self.feature_names, 'models/saved/feature_names.pkl')
        print(f"    Saved: feature_names.pkl")

        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_features': len(self.feature_names),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'models_trained': list(self.models.keys()),
            'results': {
                model: {
                    'accuracy': float(results['accuracy']),
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1_score': float(results['f1_score']),
                    'roc_auc': float(results['roc_auc'])
                }
                for model, results in self.results.items()
            }
        }
        with open('models/saved/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        print("    Saved: metadata.json")

        print(" All models and metadata saved successfully")


    def run_full_pipeline(self, test_size=0.2, random_state=42, smote_method='smote'):
        """Run the complete ML pipeline"""
        print("\n" + "-"*40)
        print("STARTING COMPLETE ML PIPELINE")
        print("-"*40 + "\n")

        # Step 1: Load data
        if self.df is None:
            self.load_data()

        # Step 2: Feature Engineering
        self.feature_engineering()

        # Step 3: Prepare features
        X, y = self.prepare_features()

        # Step 4: Split and Scale Data
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = \
            self.split_and_scale_data(X, y, test_size, random_state)

        # Step 5: Handle Class Imbalance
        self.X_train_balanced, self.y_train_balanced = self.handle_class_imbalance(smote_method)

        # Step 6: Train Models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()

        # Step 7: Compare Models
        self.compare_models()

        # Step 8: Plot ROC Curves and Confusion Matrices
        self.plot_roc_curves()
        self.plot_confusion_matrices()

        # Step 9: Save Models
        self.save_models()

        print("\n" + "__"*40)
        print("ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("__"*40 + "\n")


# Usage Example
if __name__ == "__main__":
    # Make sure your processed data path is correct
    processed_data_path = '/Users/apple/Documents/hotel-booking-intelligence/data/processed/hotels_bookings_processed.csv'

    # Initialize the model pipeline
    pipeline = HotelCancellationModel(data_path=processed_data_path)

    # Run the full pipeline
    pipeline.run_full_pipeline()