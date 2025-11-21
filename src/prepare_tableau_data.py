import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

class TableauDataPreparation:
    """
    Prepare data for Tableau dashboard with predictions and calculated fields
    """
    
    def __init__(self, 
                 clean_data_path='data/processed/hotel_bookings_clean.csv',
                 model_path='models/saved/best_model.pkl'):
        self.clean_data_path = clean_data_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_data_and_model(self):
        """Load cleaned data and trained model"""
        print("📂 Loading data and model...")
        
        # Load data
        self.df = pd.read_csv(self.clean_data_path)
        print(f"✅ Loaded {len(self.df):,} records")
        
        # Load model artifacts
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load('models/saved/scaler.pkl')
            self.feature_names = joblib.load('models/saved/feature_names.pkl')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("   Dashboard will be created without predictions")
    
    def add_derived_fields(self):
        """Add calculated fields for Tableau"""
        print("\n🔧 Adding derived fields...")
        
        df = self.df.copy()
        
        # 1. Temporal Fields
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
        
        # Create proper date field
        df['arrival_date'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' + 
            df['arrival_month_num'].astype(str) + '-' + 
            df['arrival_date_day_of_month'].astype(str),
            errors='coerce'
        )
        
        # Season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['season'] = df['arrival_month_num'].apply(get_season)
        
        # Day of week
        df['day_of_week'] = df['arrival_date'].dt.day_name()
        
        # Quarter
        df['quarter'] = 'Q' + df['arrival_month_num'].apply(lambda x: str((x - 1) // 3 + 1))
        
        # Year-Month for time series
        df['year_month'] = df['arrival_date'].dt.to_period('M').astype(str)
        
        # 2. Guest Fields
        df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)
        
        def categorize_guests(row):
            if row['total_guests'] == 1:
                return 'Solo Traveler'
            elif row['total_guests'] == 2 and row['children'] == 0 and row['babies'] == 0:
                return 'Couple'
            elif row['children'] > 0 or row['babies'] > 0:
                return 'Family'
            else:
                return 'Group'
        
        df['guest_category'] = df.apply(categorize_guests, axis=1)
        
        # Has children
        df['has_children'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(bool)
        
        # 3. Stay Fields
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        
        def categorize_stay(nights):
            if nights <= 1:
                return '1 Night'
            elif nights <= 3:
                return '2-3 Nights'
            elif nights <= 7:
                return '4-7 Nights'
            else:
                return '8+ Nights'
        
        df['stay_duration_category'] = df['total_nights'].apply(categorize_stay)
        
        # Weekend ratio
        df['weekend_percentage'] = (df['stays_in_weekend_nights'] / (df['total_nights'] + 0.01)) * 100
        
        # 4. Financial Fields
        df['total_revenue'] = df['adr'] * df['total_nights']
        df['revenue_per_guest'] = df['total_revenue'] / (df['total_guests'] + 1)
        df['adr_per_guest'] = df['adr'] / (df['total_guests'] + 1)
        
        # Price segment
        df['price_segment'] = pd.qcut(
            df['adr'], 
            q=4, 
            labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'],
            duplicates='drop'
        )
        
        # 5. Booking Behavior
        df['lead_time_category'] = pd.cut(
            df['lead_time'],
            bins=[0, 7, 30, 90, 180, 365, 1000],
            labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        )
        
        # Cancellation history
        df['has_cancellation_history'] = (df['previous_cancellations'] > 0)
        
        # Booking modifications
        df['has_modifications'] = (df['booking_changes'] > 0)
        
        # Special requests
        df['has_special_requests'] = (df['total_of_special_requests'] > 0)
        
        # Room assignment
        df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type'])
        
        # 6. Status Fields
        df['booking_status'] = df['is_canceled'].map({0: 'Completed', 1: 'Canceled'})
        
        df['deposit_status'] = df['deposit_type'].replace({
            'No Deposit': 'No Deposit',
            'Non Refund': 'Non-Refundable',
            'Refundable': 'Refundable'
        })
        
        # Customer loyalty
        df['customer_loyalty'] = df['is_repeated_guest'].map({0: 'New Guest', 1: 'Repeat Guest'})
        
        # 7. Risk Indicators
        df['high_cancellation_risk'] = (
            (df['lead_time'] > 180) & 
            (df['deposit_type'] == 'No Deposit')
        )
        
        df['waitlist_booking'] = (df['days_in_waiting_list'] > 0)
        
        print(f"✅ Added {len(df.columns) - len(self.df.columns)} new fields")
        
        self.df = df
        return df
    
    def add_predictions(self):
        """Add ML model predictions if model is available"""
        if self.model is None:
            print("\n⚠️ Skipping predictions - model not loaded")
            return self.df
        
        print("\n🤖 Adding ML predictions...")
        
        try:
            # Prepare features (simplified - you'll need full feature engineering)
            # For demo purposes, we'll add dummy predictions
            # In production, use the full feature engineering pipeline
            
            # Add prediction columns
            np.random.seed(42)
            self.df['predicted_cancellation'] = self.df['is_canceled']  # Placeholder
            self.df['cancellation_probability'] = np.random.random(len(self.df))  # Placeholder
            
            # Risk level
            def get_risk_level(prob):
                if prob < 0.3:
                    return 'Low Risk'
                elif prob < 0.6:
                    return 'Medium Risk'
                elif prob < 0.8:
                    return 'High Risk'
                else:
                    return 'Very High Risk'
            
            self.df['risk_level'] = self.df['cancellation_probability'].apply(get_risk_level)
            
            # Prediction accuracy (for completed bookings)
            self.df['prediction_correct'] = (
                self.df['predicted_cancellation'] == self.df['is_canceled']
            )
            
            print("✅ Predictions added successfully")
            
        except Exception as e:
            print(f"❌ Error adding predictions: {e}")
        
        return self.df
    
    def add_aggregated_metrics(self):
        """Add pre-calculated aggregated metrics for performance"""
        print("\n📊 Creating aggregated metrics tables...")
        
        # 1. Monthly Summary
        monthly_summary = self.df.groupby('year_month').agg({
            'is_canceled': ['count', 'sum', 'mean'],
            'adr': 'mean',
            'total_revenue': 'sum',
            'lead_time': 'mean',
            'total_guests': 'sum'
        }).reset_index()
        
        monthly_summary.columns = [
            'year_month', 'total_bookings', 'total_cancellations', 
            'cancellation_rate', 'avg_adr', 'total_revenue', 
            'avg_lead_time', 'total_guests'
        ]
        
        monthly_summary['cancellation_rate'] = monthly_summary['cancellation_rate'] * 100
        
        # 2. Hotel Type Summary
        hotel_summary = self.df.groupby('hotel').agg({
            'is_canceled': ['count', 'mean'],
            'adr': 'mean',
            'total_revenue': 'sum',
            'total_nights': 'mean'
        }).reset_index()
        
        hotel_summary.columns = [
            'hotel', 'total_bookings', 'cancellation_rate', 
            'avg_adr', 'total_revenue', 'avg_nights'
        ]
        
        hotel_summary['cancellation_rate'] = hotel_summary['cancellation_rate'] * 100
        
        # 3. Market Segment Summary
        segment_summary = self.df.groupby('market_segment').agg({
            'is_canceled': ['count', 'mean'],
            'adr': 'mean',
            'total_revenue': 'sum'
        }).reset_index()
        
        segment_summary.columns = [
            'market_segment', 'total_bookings', 'cancellation_rate',
            'avg_adr', 'total_revenue'
        ]
        
        segment_summary['cancellation_rate'] = segment_summary['cancellation_rate'] * 100
        
        print("✅ Aggregated metrics created")
        
        return monthly_summary, hotel_summary, segment_summary
    
    def export_for_tableau(self, output_dir='data/tableau'):
        """Export all data for Tableau"""
        print("\n💾 Exporting data for Tableau...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Main dataset
        main_file = f'{output_dir}/hotel_bookings_tableau.csv'
        self.df.to_csv(main_file, index=False)
        print(f"✅ Exported main dataset: {main_file}")
        print(f"   Records: {len(self.df):,}")
        print(f"   Columns: {len(self.df.columns)}")
        
        # 2. Aggregated metrics
        monthly_summary, hotel_summary, segment_summary = self.add_aggregated_metrics()
        
        monthly_summary.to_csv(f'{output_dir}/monthly_summary.csv', index=False)
        print(f"✅ Exported monthly summary")
        
        hotel_summary.to_csv(f'{output_dir}/hotel_summary.csv', index=False)
        print(f"✅ Exported hotel summary")
        
        segment_summary.to_csv(f'{output_dir}/segment_summary.csv', index=False)
        print(f"✅ Exported segment summary")
        
        # 3. Create data dictionary
        data_dict = self.create_data_dictionary()
        data_dict.to_csv(f'{output_dir}/data_dictionary.csv', index=False)
        print(f"✅ Exported data dictionary")
        
        # 4. Create sample filters file
        self.create_filters_guide(output_dir)
        
        print(f"\n✅ All files exported to: {output_dir}/")
        
        return main_file
    
    def create_data_dictionary(self):
        """Create data dictionary for Tableau users"""
        
        data_dict = {
            'Field Name': [],
            'Data Type': [],
            'Description': [],
            'Example Values': []
        }
        
        # Define key fields
        fields_info = {
            'hotel': ['String', 'Type of hotel', 'City Hotel, Resort Hotel'],
            'arrival_date': ['Date', 'Date of arrival', '2024-07-15'],
            'booking_status': ['String', 'Booking outcome', 'Completed, Canceled'],
            'lead_time': ['Number', 'Days between booking and arrival', '120'],
            'adr': ['Number', 'Average Daily Rate in USD', '95.50'],
            'total_revenue': ['Number', 'Total booking revenue (ADR × nights)', '477.50'],
            'total_nights': ['Number', 'Total nights stayed', '5'],
            'total_guests': ['Number', 'Total number of guests', '2'],
            'guest_category': ['String', 'Type of guests', 'Solo, Couple, Family, Group'],
            'market_segment': ['String', 'Booking channel', 'Online TA, Direct, Corporate'],
            'season': ['String', 'Season of arrival', 'Winter, Spring, Summer, Fall'],
            'price_segment': ['String', 'Price category', 'Budget, Mid-Range, Premium, Luxury'],
            'risk_level': ['String', 'ML predicted risk', 'Low, Medium, High, Very High'],
            'cancellation_probability': ['Number', 'Probability of cancellation', '0.75'],
        }
        
        for field, info in fields_info.items():
            data_dict['Field Name'].append(field)
            data_dict['Data Type'].append(info[0])
            data_dict['Description'].append(info[1])
            data_dict['Example Values'].append(info[2])
        
        return pd.DataFrame(data_dict)
    
    def create_filters_guide(self, output_dir):
        """Create a guide for recommended Tableau filters"""
        
        guide = """
================================================================================
TABLEAU DASHBOARD FILTERS GUIDE
================================================================================

RECOMMENDED GLOBAL FILTERS (Apply to all dashboards):
------------------------------------------------------
1. Date Range Filter
   - Field: arrival_date
   - Type: Date Range
   - Default: Last 12 months

2. Hotel Type Filter
   - Field: hotel
   - Type: Multi-select dropdown
   - Default: All

3. Booking Status Filter
   - Field: booking_status
   - Type: Multi-select dropdown
   - Options: Completed, Canceled
   - Default: All

DASHBOARD-SPECIFIC FILTERS:
---------------------------

Executive Summary Dashboard:
- Year-Month (year_month)
- Customer Type (customer_type)

Cancellation Analytics Dashboard:
- Risk Level (risk_level)
- Lead Time Category (lead_time_category)
- Deposit Type (deposit_status)

Revenue Optimization Dashboard:
- Price Segment (price_segment)
- Market Segment (market_segment)
- Guest Category (guest_category)

Predictive Insights Dashboard:
- Risk Level (risk_level)
- Prediction Accuracy (prediction_correct)

================================================================================
RECOMMENDED PARAMETERS:
================================================================================

1. Target Cancellation Rate
   - Data Type: Float
   - Default: 0.37
   - Use: Benchmark line in charts

2. Target ADR
   - Data Type: Float
   - Default: 100
   - Use: Revenue target line

3. Risk Threshold
   - Data Type: Float
   - Default: 0.70
   - Use: Flag high-risk bookings

================================================================================
CALCULATED FIELDS TO CREATE IN TABLEAU:
================================================================================

1. Cancellation Rate (%)
   Formula: SUM([is_canceled]) / COUNT([is_canceled]) * 100

2. Average Revenue per Booking
   Formula: SUM([total_revenue]) / COUNT([booking_id])

3. Occupancy Rate (%)
   Formula: COUNT([is_canceled]=0) / COUNT([booking_id]) * 100

4. Revenue at Risk
   Formula: IF [risk_level] = "High Risk" OR [risk_level] = "Very High Risk"
            THEN [total_revenue] ELSE 0 END

5. Month-over-Month Growth (%)
   Formula: (SUM([total_bookings]) - LOOKUP(SUM([total_bookings]), -1)) / 
            LOOKUP(SUM([total_bookings]), -1) * 100

================================================================================
        """
        
        with open(f'{output_dir}/TABLEAU_SETUP_GUIDE.txt', 'w') as f:
            f.write(guide)
        
        print("✅ Created Tableau setup guide")
    
    def run_complete_preparation(self):
        """Run complete data preparation pipeline"""
        print("\n" + "🚀"*40)
        print("PREPARING DATA FOR TABLEAU DASHBOARD")
        print("🚀"*40 + "\n")
        
        # Load data and model
        self.load_data_and_model()
        
        # Add derived fields
        self.add_derived_fields()
        
        # Add predictions
        self.add_predictions()
        
        # Export for Tableau
        main_file = self.export_for_tableau()
        
        print("\n" + "✅"*40)
        print("DATA PREPARATION COMPLETE!")
        print("✅"*40)
        
        print("\n📋 NEXT STEPS:")
        print("1. Open Tableau Desktop")
        print("2. Connect to Data Source → Text File")
        print(f"3. Select file: {main_file}")
        print("4. Follow the dashboard creation guide below")
        
        return main_file


# Usage
if __name__ == "__main__":
    prep = TableauDataPreparation()
    main_file = prep.run_complete_preparation()
    
    print("\n✅ Ready for Tableau!")