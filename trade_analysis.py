import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class BangladeshTradeAnalyzer:
    def __init__(self):
        self.trade_data = None
        self.country_codes = None
        self.product_codes = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
        # Load trade data
        self.trade_data = pd.read_csv('bd_trade_data.csv')
        print(f"Trade data loaded: {len(self.trade_data):,} records")
        
        # Load country codes
        self.country_codes = pd.read_csv('country_codes_V202501.csv')
        print(f"Country codes loaded: {len(self.country_codes)} countries")
        
        # Load product codes
        self.product_codes = pd.read_csv('product_codes_HS92_V202501.csv')
        print(f"Product codes loaded: {len(self.product_codes)} products")
        
        # Basic data info
        print("\n=== TRADE DATA OVERVIEW ===")
        print(f"Columns: {list(self.trade_data.columns)}")
        print(f"Years covered: {self.trade_data['t'].min()} - {self.trade_data['t'].max()}")
        print(f"Total trade value: ${self.trade_data['v'].sum():,.0f} thousand USD")
        print(f"Total quantity: {self.trade_data['q'].sum():,.0f} metric tons")
        
    def merge_data(self):
        """Merge trade data with country and product names"""
        print("\nMerging datasets...")
        
        # Merge with exporter countries (i)
        self.trade_data = self.trade_data.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='i', right_on='country_code', how='left'
        ).rename(columns={'country_name': 'exporter_name', 'country_iso3': 'exporter_iso3'})
        
        # Merge with importer countries (j) - Bangladesh should be 50
        self.trade_data = self.trade_data.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='j', right_on='country_code', how='left', suffixes=('', '_imp')
        ).rename(columns={'country_name': 'importer_name', 'country_iso3': 'importer_iso3'})
        
        # Merge with product descriptions
        self.trade_data['k_str'] = self.trade_data['k'].astype(str).str.zfill(6)
        self.trade_data = self.trade_data.merge(
            self.product_codes, left_on='k_str', right_on='code', how='left'
        )
        
        # Clean up columns
        self.trade_data = self.trade_data.drop(['country_code', 'country_code_imp'], axis=1)
        
        print("Data merged successfully!")
        
    def analyze_trade_direction(self):
        """Analyze Bangladesh's imports vs exports"""
        print("\n=== TRADE DIRECTION ANALYSIS ===")
        
        # Bangladesh country code is 50
        bangladesh_exports = self.trade_data[self.trade_data['i'] == 50]
        bangladesh_imports = self.trade_data[self.trade_data['j'] == 50]
        
        print(f"Bangladesh Exports: {len(bangladesh_exports):,} records")
        print(f"Bangladesh Imports: {len(bangladesh_imports):,} records")
        
        export_value = bangladesh_exports['v'].sum()
        import_value = bangladesh_imports['v'].sum()
        
        print(f"Total Export Value: ${export_value:,.0f} thousand USD")
        print(f"Total Import Value: ${import_value:,.0f} thousand USD")
        print(f"Trade Balance: ${export_value - import_value:,.0f} thousand USD")
        
        return bangladesh_exports, bangladesh_imports
        
    def top_trading_partners(self, bangladesh_exports, bangladesh_imports, top_n=10):
        """Analyze top trading partners"""
        print(f"\n=== TOP {top_n} TRADING PARTNERS ===")
        
        # Top export destinations
        top_export_partners = bangladesh_exports.groupby(['j', 'importer_name'])['v'].sum().sort_values(ascending=False).head(top_n)
        print("\nTop Export Destinations:")
        for (code, name), value in top_export_partners.items():
            print(f"{name}: ${value:,.0f} thousand USD")
            
        # Top import sources
        top_import_partners = bangladesh_imports.groupby(['i', 'exporter_name'])['v'].sum().sort_values(ascending=False).head(top_n)
        print("\nTop Import Sources:")
        for (code, name), value in top_import_partners.items():
            print(f"{name}: ${value:,.0f} thousand USD")
            
        return top_export_partners, top_import_partners
        
    def top_products(self, bangladesh_exports, bangladesh_imports, top_n=10):
        """Analyze top traded products"""
        print(f"\n=== TOP {top_n} TRADED PRODUCTS ===")
        
        # Top export products
        top_export_products = bangladesh_exports.groupby(['k', 'description'])['v'].sum().sort_values(ascending=False).head(top_n)
        print("\nTop Export Products:")
        for (code, desc), value in top_export_products.items():
            print(f"{code} - {desc}: ${value:,.0f} thousand USD")
            
        # Top import products
        top_import_products = bangladesh_imports.groupby(['k', 'description'])['v'].sum().sort_values(ascending=False).head(top_n)
        print("\nTop Import Products:")
        for (code, desc), value in top_import_products.items():
            print(f"{code} - {desc}: ${value:,.0f} thousand USD")
            
        return top_export_products, top_import_products
        
    def yearly_trends(self, bangladesh_exports, bangladesh_imports):
        """Analyze yearly trade trends"""
        print("\n=== YEARLY TRADE TRENDS ===")
        
        # Yearly export trends
        yearly_exports = bangladesh_exports.groupby('t')['v'].sum()
        yearly_imports = bangladesh_imports.groupby('t')['v'].sum()
        
        print("\nYearly Trade Values:")
        for year in sorted(yearly_exports.index):
            exp_val = yearly_exports.get(year, 0)
            imp_val = yearly_imports.get(year, 0)
            balance = exp_val - imp_val
            print(f"{year}: Exports ${exp_val:,.0f}k, Imports ${imp_val:,.0f}k, Balance ${balance:,.0f}k")
            
        return yearly_exports, yearly_imports
        
    def create_visualizations(self, bangladesh_exports, bangladesh_imports):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Trade Balance Overview
        ax1 = plt.subplot(2, 3, 1)
        export_value = bangladesh_exports['v'].sum()
        import_value = bangladesh_imports['v'].sum()
        
        plt.bar(['Exports', 'Imports'], [export_value, import_value], 
                color=['green', 'red'], alpha=0.7)
        plt.title('Bangladesh Trade Overview\n(Total Value in Thousand USD)', fontsize=14, fontweight='bold')
        plt.ylabel('Value (Thousand USD)')
        for i, v in enumerate([export_value, import_value]):
            plt.text(i, v + max(export_value, import_value) * 0.01, f'${v:,.0f}k', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Top Export Destinations
        ax2 = plt.subplot(2, 3, 2)
        top_export_partners = bangladesh_exports.groupby('importer_name')['v'].sum().sort_values(ascending=False).head(8)
        plt.barh(range(len(top_export_partners)), top_export_partners.values, color='lightgreen')
        plt.yticks(range(len(top_export_partners)), [name[:20] + '...' if len(name) > 20 else name for name in top_export_partners.index])
        plt.title('Top Export Destinations', fontsize=14, fontweight='bold')
        plt.xlabel('Value (Thousand USD)')
        
        # 3. Top Import Sources
        ax3 = plt.subplot(2, 3, 3)
        top_import_partners = bangladesh_imports.groupby('exporter_name')['v'].sum().sort_values(ascending=False).head(8)
        plt.barh(range(len(top_import_partners)), top_import_partners.values, color='lightcoral')
        plt.yticks(range(len(top_import_partners)), [name[:20] + '...' if len(name) > 20 else name for name in top_import_partners.index])
        plt.title('Top Import Sources', fontsize=14, fontweight='bold')
        plt.xlabel('Value (Thousand USD)')
        
        # 4. Top Export Products (by HS2 category)
        ax4 = plt.subplot(2, 3, 4)
        bangladesh_exports['hs2'] = bangladesh_exports['k'].astype(str).str[:2]
        top_export_hs2 = bangladesh_exports.groupby('hs2')['v'].sum().sort_values(ascending=False).head(10)
        plt.pie(top_export_hs2.values, labels=top_export_hs2.index, autopct='%1.1f%%', startangle=90)
        plt.title('Top Export Product Categories\n(HS2 Level)', fontsize=14, fontweight='bold')
        
        # 5. Top Import Products (by HS2 category)
        ax5 = plt.subplot(2, 3, 5)
        bangladesh_imports['hs2'] = bangladesh_imports['k'].astype(str).str[:2]
        top_import_hs2 = bangladesh_imports.groupby('hs2')['v'].sum().sort_values(ascending=False).head(10)
        plt.pie(top_import_hs2.values, labels=top_import_hs2.index, autopct='%1.1f%%', startangle=90)
        plt.title('Top Import Product Categories\n(HS2 Level)', fontsize=14, fontweight='bold')
        
        # 6. Yearly Trends
        ax6 = plt.subplot(2, 3, 6)
        yearly_exports = bangladesh_exports.groupby('t')['v'].sum()
        yearly_imports = bangladesh_imports.groupby('t')['v'].sum()
        
        years = sorted(set(yearly_exports.index) | set(yearly_imports.index))
        exp_values = [yearly_exports.get(year, 0) for year in years]
        imp_values = [yearly_imports.get(year, 0) for year in years]
        
        plt.plot(years, exp_values, marker='o', linewidth=2, label='Exports', color='green')
        plt.plot(years, imp_values, marker='s', linewidth=2, label='Imports', color='red')
        plt.title('Yearly Trade Trends', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Value (Thousand USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bangladesh_trade_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self, bangladesh_exports, bangladesh_imports):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("         BANGLADESH TRADE ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        # Basic statistics
        total_export_value = bangladesh_exports['v'].sum()
        total_import_value = bangladesh_imports['v'].sum()
        trade_balance = total_export_value - total_import_value
        
        print(f"\n📊 OVERALL TRADE STATISTICS:")
        print(f"   • Total Export Value: ${total_export_value:,.0f} thousand USD")
        print(f"   • Total Import Value: ${total_import_value:,.0f} thousand USD")
        print(f"   • Trade Balance: ${trade_balance:,.0f} thousand USD")
        print(f"   • Trade Coverage Ratio: {(total_export_value/total_import_value)*100:.1f}%")
        
        # Top partners
        top_export_partner = bangladesh_exports.groupby('importer_name')['v'].sum().idxmax()
        top_import_partner = bangladesh_imports.groupby('exporter_name')['v'].sum().idxmax()
        
        print(f"\n🌍 KEY TRADING PARTNERS:")
        print(f"   • Largest Export Destination: {top_export_partner}")
        print(f"   • Largest Import Source: {top_import_partner}")
        
        # Product diversity
        unique_export_products = bangladesh_exports['k'].nunique()
        unique_import_products = bangladesh_imports['k'].nunique()
        
        print(f"\n📦 PRODUCT DIVERSITY:")
        print(f"   • Unique Export Products: {unique_export_products}")
        print(f"   • Unique Import Products: {unique_import_products}")
        
        # Geographic reach
        export_countries = bangladesh_exports['j'].nunique()
        import_countries = bangladesh_imports['i'].nunique()
        
        print(f"\n🗺️  GEOGRAPHIC REACH:")
        print(f"   • Export Destinations: {export_countries} countries")
        print(f"   • Import Sources: {import_countries} countries")
        
        print("\n" + "="*60)
        
    def run_complete_analysis(self):
        """Run the complete trade analysis"""
        print("🇧🇩 BANGLADESH TRADE ANALYSIS STARTING...")
        print("="*50)
        
        # Load and prepare data
        self.load_data()
        self.merge_data()
        
        # Analyze trade patterns
        bangladesh_exports, bangladesh_imports = self.analyze_trade_direction()
        
        # Detailed analysis
        self.top_trading_partners(bangladesh_exports, bangladesh_imports)
        self.top_products(bangladesh_exports, bangladesh_imports)
        self.yearly_trends(bangladesh_exports, bangladesh_imports)
        
        # Create visualizations
        self.create_visualizations(bangladesh_exports, bangladesh_imports)
        
        # Generate summary
        self.generate_summary_report(bangladesh_exports, bangladesh_imports)
        
        print("\n✅ Analysis completed! Check 'bangladesh_trade_analysis.png' for visualizations.")
        
        return bangladesh_exports, bangladesh_imports

# Run the analysis
if __name__ == "__main__":
    analyzer = BangladeshTradeAnalyzer()
    exports, imports = analyzer.run_complete_analysis()