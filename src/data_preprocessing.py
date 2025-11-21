import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HotelDataPreprocessor:
    """
    Comprehensive data preprocessing for hotel booking dataset
    """
    
    def __init__(self, filepath):
        """Initialize with dataset filepath"""
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        return self.df
    
    def initial_exploration(self):
        """Perform initial data exploration"""
        print("\n" + "="*80)
        print("INITIAL DATA EXPLORATION")
        print("="*80)
        
        # Basic info
        print(f"\n Dataset Shape: {self.df.shape}")
        print(f" Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n Data Types:")
        print(self.df.dtypes.value_counts())
        
        # Missing values
        print("\n Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Basic statistics
        print("\n Numerical Columns Statistics:")
        print(self.df.describe())
        
        # Categorical columns
        print("\n Categorical Columns:")
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols[:5]:  # First 5 categorical columns
            print(f"\n{col}: {self.df[col].nunique()} unique values")
            print(self.df[col].value_counts().head())
            
        return missing_df
    
    def handle_missing_values(self):
        """Handle missing values intelligently"""
        print("\n🔧 Handling missing values...")
        self.df_clean = self.df.copy()
        
        # Children: Fill with 0 (no children)
        if 'children' in self.df_clean.columns:
            self.df_clean['children'] = self.df_clean['children'].fillna(0)
            print(" Filled 'children' with 0")
        
        # Country: Fill with 'Unknown'
        if 'country' in self.df_clean.columns:
            self.df_clean['country'] = self.df_clean['country'].fillna('Unknown')
            print("Filled 'country' with 'Unknown'")
        
        # Agent: Fill with 0 (no agent)
        if 'agent' in self.df_clean.columns:
            self.df_clean['agent'] = self.df_clean['agent'].fillna(0)
            print("Filled 'agent' with 0")
        
        # Company: Fill with 0 (no company)
        if 'company' in self.df_clean.columns:
            self.df_clean['company'] = self.df_clean['company'].fillna(0)
            print(" Filled 'company' with 0")
        
        print(f" Missing values handled. Remaining: {self.df_clean.isnull().sum().sum()}")
        return self.df_clean
    
    def remove_outliers(self, columns=['adr', 'lead_time'], method='IQR'):
        """Remove outliers using IQR method"""
        print(f"\n Removing outliers from: {columns}")
        
        initial_rows = len(self.df_clean)
        
        for col in columns:
            if col in self.df_clean.columns:
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                self.df_clean = self.df_clean[
                    (self.df_clean[col] >= lower_bound) & 
                    (self.df_clean[col] <= upper_bound)
                ]
                
                removed = initial_rows - len(self.df_clean)
                print(f"   {col}: Removed {removed:,} outliers (Range: {lower_bound:.2f} - {upper_bound:.2f})")
                initial_rows = len(self.df_clean)
        
        # Also remove negative ADR
        if 'adr' in self.df_clean.columns:
            self.df_clean = self.df_clean[self.df_clean['adr'] >= 0]
            print(f"  Removed negative ADR values")
        
        print(f" Final dataset: {len(self.df_clean):,} records")
        return self.df_clean
    
    def create_eda_visualizations(self, save_path='notebooks/eda_plots/'):
        """Create comprehensive EDA visualizations"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\n Creating EDA visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 1. Cancellation Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        self.df_clean['is_canceled'].value_counts().plot(
            kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c']
        )
        axes[0].set_title('Booking Status Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Status (0=Not Canceled, 1=Canceled)')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['Not Canceled', 'Canceled'], rotation=0)
        
        # Percentage
        cancel_pct = self.df_clean['is_canceled'].value_counts(normalize=True) * 100
        axes[1].pie(cancel_pct, labels=['Not Canceled', 'Canceled'], 
                    autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                    startangle=90)
        axes[1].set_title('Cancellation Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}01_cancellation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 01_cancellation_distribution.png")
        
        # 2. Lead Time Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution
        axes[0, 0].hist(self.df_clean['lead_time'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Lead Time Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Frequency')
        
        # Boxplot by cancellation
        self.df_clean.boxplot(column='lead_time', by='is_canceled', ax=axes[0, 1])
        axes[0, 1].set_title('Lead Time by Cancellation Status', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Canceled')
        axes[0, 1].set_ylabel('Lead Time (days)')
        
        # Lead time categories
        lead_time_cats = pd.cut(self.df_clean['lead_time'], 
                                bins=[0, 7, 30, 90, 180, 365, 1000],
                                labels=['0-7 days', '8-30 days', '31-90 days', 
                                       '91-180 days', '181-365 days', '365+ days'])
        cancel_by_lead = self.df_clean.groupby(lead_time_cats)['is_canceled'].mean() * 100
        
        cancel_by_lead.plot(kind='bar', ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Cancellation Rate by Lead Time Category', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Lead Time Category')
        axes[1, 0].set_ylabel('Cancellation Rate (%)')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        
        # Average lead time by month
        monthly_lead = self.df_clean.groupby('arrival_date_month')['lead_time'].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_lead = monthly_lead.reindex([m for m in month_order if m in monthly_lead.index])
        
        monthly_lead.plot(kind='line', ax=axes[1, 1], marker='o', color='purple')
        axes[1, 1].set_title('Average Lead Time by Arrival Month', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Avg Lead Time (days)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}02_lead_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 02_lead_time_analysis.png")
        
        # 3. ADR Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution
        axes[0, 0].hist(self.df_clean['adr'], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 0].set_title('ADR Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('ADR ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Boxplot by cancellation
        self.df_clean.boxplot(column='adr', by='is_canceled', ax=axes[0, 1])
        axes[0, 1].set_title('ADR by Cancellation Status', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Canceled')
        axes[0, 1].set_ylabel('ADR ($)')
        
        # ADR by month
        monthly_adr = self.df_clean.groupby('arrival_date_month')['adr'].mean()
        monthly_adr = monthly_adr.reindex([m for m in month_order if m in monthly_adr.index])
        
        monthly_adr.plot(kind='bar', ax=axes[1, 0], color='orange')
        axes[1, 0].set_title('Average ADR by Month', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Avg ADR ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # ADR by customer type
        if 'customer_type' in self.df_clean.columns:
            customer_adr = self.df_clean.groupby('customer_type')['adr'].mean().sort_values()
            customer_adr.plot(kind='barh', ax=axes[1, 1], color='teal')
            axes[1, 1].set_title('Average ADR by Customer Type', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Avg ADR ($)')
            axes[1, 1].set_ylabel('Customer Type')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}03_adr_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 03_adr_analysis.png")
        
        # 4. Market Segment Analysis
        if 'market_segment' in self.df_clean.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Distribution
            market_counts = self.df_clean['market_segment'].value_counts()
            axes[0].barh(market_counts.index, market_counts.values, color='steelblue')
            axes[0].set_title('Bookings by Market Segment', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Count')
            
            # Cancellation rate
            market_cancel = self.df_clean.groupby('market_segment')['is_canceled'].mean() * 100
            market_cancel = market_cancel.sort_values()
            axes[1].barh(market_cancel.index, market_cancel.values, color='indianred')
            axes[1].set_title('Cancellation Rate by Market Segment', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Cancellation Rate (%)')
            
            plt.tight_layout()
            plt.savefig(f'{save_path}04_market_segment_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   Saved: 04_market_segment_analysis.png")
        
        # 5. Correlation Heatmap
        plt.figure(figsize=(14, 10))
        
        # Select numerical columns
        num_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df_clean[num_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 05_correlation_heatmap.png")
        
        # 6. Guest Composition
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Adults distribution
        self.df_clean['adults'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='lightcoral')
        axes[0, 0].set_title('Adults per Booking', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Adults')
        axes[0, 0].set_ylabel('Count')
        
        # Children distribution
        self.df_clean['children'].value_counts().sort_index().head(5).plot(kind='bar', ax=axes[0, 1], color='lightblue')
        axes[0, 1].set_title('Children per Booking (Top 5)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Children')
        axes[0, 1].set_ylabel('Count')
        
        # Total guests
        self.df_clean['total_guests'] = self.df_clean['adults'] + self.df_clean['children'] + self.df_clean['babies']
        self.df_clean['total_guests'].value_counts().sort_index().head(10).plot(
            kind='bar', ax=axes[1, 0], color='mediumpurple'
        )
        axes[1, 0].set_title('Total Guests per Booking (Top 10)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Total Guests')
        axes[1, 0].set_ylabel('Count')
        
        # Stay duration
        self.df_clean['total_nights'] = (self.df_clean['stays_in_weekend_nights'] + 
                                         self.df_clean['stays_in_week_nights'])
        self.df_clean['total_nights'].value_counts().sort_index().head(15).plot(
            kind='bar', ax=axes[1, 1], color='gold'
        )
        axes[1, 1].set_title('Length of Stay (Top 15)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Nights')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}06_guest_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: 06_guest_composition.png")
        
        # 7. Deposit Type Impact
        if 'deposit_type' in self.df_clean.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Distribution
            deposit_counts = self.df_clean['deposit_type'].value_counts()
            axes[0].pie(deposit_counts, labels=deposit_counts.index, autopct='%1.1f%%',
                       colors=['#3498db', '#e74c3c', '#2ecc71'], startangle=90)
            axes[0].set_title('Deposit Type Distribution', fontsize=12, fontweight='bold')
            
            # Cancellation by deposit
            deposit_cancel = self.df_clean.groupby('deposit_type')['is_canceled'].mean() * 100
            deposit_cancel.plot(kind='bar', ax=axes[1], color='crimson')
            axes[1].set_title('Cancellation Rate by Deposit Type', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Deposit Type')
            axes[1].set_ylabel('Cancellation Rate (%)')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}07_deposit_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   Saved: 07_deposit_analysis.png")
        
        print(f"\nAll visualizations saved to '{save_path}'")
        
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary statistics"""
        print("\n" + "="*80)
        print(" EDA SUMMARY STATISTICS")
        print("="*80)
        
        summary = {}
        
        # Overall stats
        summary['total_bookings'] = len(self.df_clean)
        summary['cancellation_rate'] = (self.df_clean['is_canceled'].mean() * 100)
        summary['avg_adr'] = self.df_clean['adr'].mean()
        summary['avg_lead_time'] = self.df_clean['lead_time'].mean()
        
        # Guest composition
        summary['avg_adults'] = self.df_clean['adults'].mean()
        summary['avg_children'] = self.df_clean['children'].mean()
        summary['avg_total_guests'] = self.df_clean['total_guests'].mean()
        summary['avg_stay_nights'] = self.df_clean['total_nights'].mean()
        
        # Special requests
        summary['avg_special_requests'] = self.df_clean['total_of_special_requests'].mean()
        summary['pct_with_special_requests'] = (
            (self.df_clean['total_of_special_requests'] > 0).mean() * 100
        )
        
        # Repeat guests
        if 'is_repeated_guest' in self.df_clean.columns:
            summary['repeat_guest_rate'] = self.df_clean['is_repeated_guest'].mean() * 100
        
        # Print summary
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
        
        return summary
    
    def save_processed_data(self, output_path='data/processed/hotel_bookings_clean.csv'):
        """Save cleaned dataset"""
        self.df_clean.to_csv(output_path, index=False)
        print(f"\n💾 Cleaned data saved to: {output_path}")
        print(f"   Shape: {self.df_clean.shape}")
        
    def run_full_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "🚀"*40)
        print("STARTING COMPLETE EDA PIPELINE")
        print("🚀"*40 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Initial exploration
        self.initial_exploration()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Remove outliers
        self.remove_outliers()
        
        # Step 5: Create visualizations
        self.create_eda_visualizations()
        
        # Step 6: Generate summary
        self.generate_eda_summary()
        
        # Step 7: Save processed data
        self.save_processed_data()
        
        print("\n" + "✅"*40)
        print("EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅"*40 + "\n")
        
        return self.df_clean


# Usage Example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = HotelDataPreprocessor('data/raw/hotel_bookings.csv')
    
    # Run full pipeline
    df_clean = preprocessor.run_full_pipeline()
    
    print("\n Sample of cleaned data:")
    print(df_clean.head())