
# Hotel Booking Intelligence  
End-to-End Machine Learning Pipeline for Predicting Booking Cancellations

This project builds a fully automated data intelligence workflow that predicts whether a hotel booking will be canceled. It covers everything from raw data ingestion to preprocessing, EDA, feature engineering, model training, model comparison, and generating deployable artifacts.

## **1. Business Problem**
Hotel cancellations cause major operational and financial losses.  
Accurate cancellation prediction helps hotels:
- Improve overbooking strategies  
- Optimize pricing and inventory  
- Identify high-risk customers  
- Maximize occupancy & revenue  

This project uses machine learning to estimate the **probability of cancellation** for each booking.

## **2. Dataset Overview**
The dataset used is the **Hotel Booking Demand Dataset**, containing City + Resort hotel bookings.

**Target Variable:**  
- `is_canceled`:  
  - `0` → Not Canceled  
  - `1` → Canceled

**Raw File Path:**  
data/raw/hotel_bookings.csv

### Key Data Attributes

- **Booking Details:** lead_time, arrival_date_year/month/week_number  
- **Guest Details:** adults, children, babies  
- **Behavioural:** previous cancellations, booking_changes, special requests  
- **Financials:** adr (Average Daily Rate), deposit_type  
- **Categorical:** market_segment, distribution_channel, customer_type  

## **3. Data Preprocessing Pipeline**

All preprocessing is done through `HotelDataPreprocessor`.

### Major Steps

1. **Load Data**  
   - Reads raw CSV  
   - Summaries: shape, memory, missing values, numeric stats

2. **Handle Missing Values**  
   - `children` → 0  
   - `country` → "Unknown"  
   - `agent` → 0  
   - `company` → 0  

3. **Outlier Removal**  
   - Applied IQR method to:  
     - `lead_time`  
     - `adr`  
   - Removes extreme and negative values

4. **Basic Feature Engineering**  
   - `total_guests = adults + children + babies`  
   - `total_nights = stays_in_weekend_nights + stays_in_week_nights`  

5. **Save Cleaned Data**  
data/processed/hotel_bookings_clean.csv


## **4. Exploratory Data Analysis (EDA)**

Exploratory Data Analysis is performed after preprocessing to understand booking trends, pricing patterns, and cancellation behavior.  
All EDA visualizations are automatically generated and saved in:
notebooks/eda_plots/

### **Major Plots Included**
- **Cancellation Distribution**  
- **Lead Time Analysis**  
- **ADR (Average Daily Rate) Analysis**  
- **Market Segment Trends**  
- **Correlation Heatmap**  
- **Guest Composition**  
- **Deposit Type vs Cancellation**  

### **Insights from EDA**
- **Longer lead times** strongly correlate with **higher cancellation probability**.  
- **No-deposit bookings** show the **highest cancellation rate** across all segments.  
- **Online Travel Agencies (OTA)** contribute the most cancellations.  
- **Seasonality and ADR pricing** influence both booking volume and cancellation likelihood.  

EDA gives a complete understanding of cancellation drivers before modeling.



## 5. Machine Learning Pipeline

All modeling logic is implemented in the `HotelCancellationModel` class.

### **End-to-End ML Workflow**
1. **Feature Engineering**  
   - Creation of 40+ advanced features: temporal, financial, behavioral, interaction-based.

2. **Feature Preparation**  
   - Standard scaling for numerical features.  
   - Label Encoding for all categorical features.  

3. **Train/Test Split**  
   - 80% training, 20% testing with stratification.  

4. **Handling Class Imbalance**  
   - SMOTE applied to oversample minority class (cancellations).  

5. **Model Training**  
   Trains four ML models:
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - LightGBM  

6. **Model Comparison**  
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC  
   - Confusion Matrices  
   - ROC Curves  
   - Consolidated comparison table  

7. **Model Selection & Saving**  
   - Best model identified (XGBoost / LightGBM).  
   - Saves:  
     - Best model  
     - All trained models  
     - Scaler  
     - Encoders  
     - Feature names  
     - Metadata  

The ML pipeline ensures reproducible training, consistent data preprocessing, and deployment-ready artifacts.
## 6. Feature Engineering (Advanced)

The project applies extensive feature engineering to enhance predictive performance.  
These engineered features capture temporal patterns, customer behavior, pricing dynamics, and risk signals.

### **Temporal Features**
- `arrival_month_num`
- `season`
- `quarter`
- `is_peak_season`
- `lead_time_bucket`

### **Guest & Stay Features**
- `total_guests`
- `has_children`
- `guest_type` (solo / couple / family / group)
- `total_nights`
- `weekend_ratio`
- `stay_duration_cat`

### **Financial Features**
- `total_revenue`
- `revenue_per_guest`
- `adr_per_guest`
- `price_category` (ADR quartiles)

### **Behaviour Features**
- `cancellation_ratio`
- `has_changes`
- `has_special_requests`
- `room_mismatch`

### **Risk Flags**
- `no_deposit`
- `long_lead_time`
- `high_modifications`
- `has_prev_cancellation`
- `was_on_waitlist`

### **Interaction Features**
- `lead_deposit_interaction`
- `adr_lead_interaction`
- `requests_repeat_interaction`


## 7. Feature Preparation

### **Numerical Features**
- Missing values filled with **0**
- Standardized using **StandardScaler**

### **Categorical Encoding**
Categorical variables encoded with **LabelEncoder**:
hotel, meal, country, market_segment,
distribution_channel, reserved_room_type,
assigned_room_type, deposit_type, customer_type

---

## 8. Train/Test Split & Class Imbalance Handling

- **80% Train / 20% Test**
- **Stratified split** to maintain class proportions
- **Standard scaling** applied after splitting
- **SMOTE** oversampling applied on training data to balance classes

---

## 9. Models Trained

The pipeline trains four supervised learning models:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**

### **Each model outputs:**
- Predictions  
- Probabilities  
- Confusion Matrix  
- Classification Report  
- ROC–AUC Score  



## 10. Model Performance Summary

Performance results are saved at:
models/model_comparison.csv
models/plots/model_comparison.png

### **Key Metrics (Approx):**

| Model                | Accuracy | F1-Score | ROC-AUC |
|----------------------|----------|----------|---------|
| Logistic Regression  | ~0.79    | ~0.73    | ~0.88   |
| Random Forest        | ~0.86    | ~0.81    | ~0.93   |
| LightGBM             | ~0.87    | ~0.81    | ~0.94   |
| **XGBoost**          | **~0.87**| **~0.82**| **~0.94** |

### **🏆 Best Model: XGBoost**



## 11. Visual Evaluation

All evaluation visualizations are stored under:
models/plots/

### **Confusion Matrices**
File: `confusion_matrices.png`  
Shows model performance on predicting cancellations vs non-cancellations.

### **ROC Curves**
File: `roc_curves.png`  
XGBoost and LightGBM show the strongest separation curve.

---

## 12. Feature Importances

Top-20 most important features generated for:
Random_Forest_feature_importance.png
XGBoost_feature_importance.png
LightGBM_feature_importance.png

### **Most Influential Predictors**
- `no_deposit`
- `deposit_type_encoded`
- `lead_time`
- `adr_lead_interaction`
- `market_segment_encoded`
- `total_of_special_requests`
- `room_mismatch`
- `revenue_per_guest`

---

## 13. Saving Trained Models

All trained models and artifacts are saved to:
models/saved/
Logistic_Regression.pkl
Random_Forest.pkl
XGBoost.pkl
LightGBM.pkl
best_model.pkl
scaler.pkl
label_encoders.pkl
feature_names.pkl
metadata.json

### **metadata.json contains:**
- Best model  
- Number of features  
- Train/Test sample size  
- Evaluation metrics for all models  



## 14. Run the Full Pipeline


from src.hotel_cancellation_model import HotelCancellationModel

pipeline = HotelCancellationModel(
    data_path="data/processed/hotels_bookings_processed.csv"
)

pipeline.run_full_pipeline()

This executes:

--Data loading

--Feature engineering

--Feature preparation

--Model training

--Model comparison

--Saving results & best model



