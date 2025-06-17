import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedBengaliTradeAnalyzer:
    def __init__(self):
        """বাংলাদেশের বর্ধিত বাণিজ্য বিশ্লেষণের জন্য ক্লাস"""
        self.trade_data = None
        self.country_codes = None
        self.product_codes = None
        self.bangladesh_exports = None
        self.bangladesh_imports = None
        
    def load_and_prepare_data(self):
        """ডেটা লোড এবং প্রস্তুত করা"""
        print("ডেটা লোড করা হচ্ছে...")
        
        # Load datasets
        self.trade_data = pd.read_csv('bd_trade_data.csv')
        self.country_codes = pd.read_csv('country_codes_V202501.csv')
        self.product_codes = pd.read_csv('product_codes_HS92_V202501.csv')
        
        # Filter Bangladesh data
        self.bangladesh_exports = self.trade_data[self.trade_data['i'] == 50].copy()
        self.bangladesh_imports = self.trade_data[self.trade_data['j'] == 50].copy()
        
        # Merge with country information
        self.bangladesh_exports = self.bangladesh_exports.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='j', right_on='country_code', how='left'
        )
        self.bangladesh_exports.rename(columns={
            'country_name': 'importer_name',
            'country_iso3': 'importer_iso3'
        }, inplace=True)
        
        self.bangladesh_imports = self.bangladesh_imports.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='i', right_on='country_code', how='left'
        )
        self.bangladesh_imports.rename(columns={
            'country_name': 'exporter_name',
            'country_iso3': 'exporter_iso3'
        }, inplace=True)
        
        # Merge with product information
        self.bangladesh_exports['k_str'] = self.bangladesh_exports['k'].astype(str).str.zfill(6)
        self.bangladesh_imports['k_str'] = self.bangladesh_imports['k'].astype(str).str.zfill(6)
        
        self.bangladesh_exports = self.bangladesh_exports.merge(
            self.product_codes[['code', 'description']], 
            left_on='k_str', right_on='code', how='left'
        )
        
        self.bangladesh_imports = self.bangladesh_imports.merge(
            self.product_codes[['code', 'description']], 
            left_on='k_str', right_on='code', how='left'
        )
        
        # Add derived variables
        self.bangladesh_exports['hs2'] = self.bangladesh_exports['k'].astype(str).str[:2]
        self.bangladesh_imports['hs2'] = self.bangladesh_imports['k'].astype(str).str[:2]
        
        print(f"ডেটা লোড সম্পন্ন: {len(self.trade_data):,} মোট রেকর্ড")
        
    def get_top_trading_partners(self):
        """শীর্ষ বাণিজ্যিক অংশীদার বিশ্লেষণ"""
        print("শীর্ষ বাণিজ্যিক অংশীদার বিশ্লেষণ করা হচ্ছে...")
        
        # Top export destinations
        top_export_partners = self.bangladesh_exports.groupby(['j', 'importer_name']).agg({
            'v': 'sum',
            'k': 'nunique',
            't': ['min', 'max']
        }).reset_index()
        
        top_export_partners.columns = ['country_code', 'country_name', 'total_value', 'product_count', 'first_year', 'last_year']
        top_export_partners['total_value_usd'] = top_export_partners['total_value'] * 1000
        top_export_partners['trade_duration'] = top_export_partners['last_year'] - top_export_partners['first_year'] + 1
        top_export_partners = top_export_partners.sort_values('total_value_usd', ascending=False).head(20)
        
        # Top import sources
        top_import_partners = self.bangladesh_imports.groupby(['i', 'exporter_name']).agg({
            'v': 'sum',
            'k': 'nunique',
            't': ['min', 'max']
        }).reset_index()
        
        top_import_partners.columns = ['country_code', 'country_name', 'total_value', 'product_count', 'first_year', 'last_year']
        top_import_partners['total_value_usd'] = top_import_partners['total_value'] * 1000
        top_import_partners['trade_duration'] = top_import_partners['last_year'] - top_import_partners['first_year'] + 1
        top_import_partners = top_import_partners.sort_values('total_value_usd', ascending=False).head(20)
        
        return top_export_partners, top_import_partners
        
    def identify_emerging_partners(self):
        """উদীয়মান বাণিজ্যিক অংশীদার চিহ্নিতকরণ"""
        print("উদীয়মান বাণিজ্যিক অংশীদার চিহ্নিত করা হচ্ছে...")
        
        # Calculate growth rates for recent years (2020-2023)
        recent_years = [2020, 2021, 2022, 2023]
        
        # Export partners growth
        export_growth = []
        for country in self.bangladesh_exports['j'].unique():
            country_data = self.bangladesh_exports[
                (self.bangladesh_exports['j'] == country) & 
                (self.bangladesh_exports['t'].isin(recent_years))
            ]
            
            if len(country_data) >= 2:
                yearly_values = country_data.groupby('t')['v'].sum()
                if len(yearly_values) >= 2:
                    growth_rate = ((yearly_values.iloc[-1] - yearly_values.iloc[0]) / yearly_values.iloc[0]) * 100
                    
                    country_name = country_data['importer_name'].iloc[0] if not country_data['importer_name'].isna().all() else f"Country_{country}"
                    total_value = country_data['v'].sum() * 1000
                    
                    export_growth.append({
                        'country_code': country,
                        'country_name': country_name,
                        'growth_rate': growth_rate,
                        'total_value': total_value,
                        'years_active': len(yearly_values)
                    })
        
        export_growth_df = pd.DataFrame(export_growth)
        
        # Filter emerging partners (high growth, moderate value)
        emerging_exports = export_growth_df[
            (export_growth_df['growth_rate'] > 20) & 
            (export_growth_df['total_value'] > 1000000)  # $1M+
        ].sort_values('growth_rate', ascending=False).head(15)
        
        # Import partners growth
        import_growth = []
        for country in self.bangladesh_imports['i'].unique():
            country_data = self.bangladesh_imports[
                (self.bangladesh_imports['i'] == country) & 
                (self.bangladesh_imports['t'].isin(recent_years))
            ]
            
            if len(country_data) >= 2:
                yearly_values = country_data.groupby('t')['v'].sum()
                if len(yearly_values) >= 2:
                    growth_rate = ((yearly_values.iloc[-1] - yearly_values.iloc[0]) / yearly_values.iloc[0]) * 100
                    
                    country_name = country_data['exporter_name'].iloc[0] if not country_data['exporter_name'].isna().all() else f"Country_{country}"
                    total_value = country_data['v'].sum() * 1000
                    
                    import_growth.append({
                        'country_code': country,
                        'country_name': country_name,
                        'growth_rate': growth_rate,
                        'total_value': total_value,
                        'years_active': len(yearly_values)
                    })
        
        import_growth_df = pd.DataFrame(import_growth)
        
        # Filter emerging partners
        emerging_imports = import_growth_df[
            (import_growth_df['growth_rate'] > 20) & 
            (import_growth_df['total_value'] > 1000000)  # $1M+
        ].sort_values('growth_rate', ascending=False).head(15)
        
        return emerging_exports, emerging_imports
        
    def analyze_top_products(self):
        """শীর্ষ রপ্তানি ও আমদানি পণ্য বিশ্লেষণ"""
        print("শীর্ষ পণ্য বিশ্লেষণ করা হচ্ছে...")
        
        # Top export products
        top_export_products = self.bangladesh_exports.groupby(['k', 'description']).agg({
            'v': 'sum',
            'j': 'nunique',
            't': ['min', 'max']
        }).reset_index()
        
        top_export_products.columns = ['product_code', 'product_name', 'total_value', 'market_count', 'first_year', 'last_year']
        top_export_products['total_value_usd'] = top_export_products['total_value'] * 1000
        top_export_products['market_span'] = top_export_products['last_year'] - top_export_products['first_year'] + 1
        top_export_products = top_export_products.sort_values('total_value_usd', ascending=False).head(25)
        
        # Top import products
        top_import_products = self.bangladesh_imports.groupby(['k', 'description']).agg({
            'v': 'sum',
            'i': 'nunique',
            't': ['min', 'max']
        }).reset_index()
        
        top_import_products.columns = ['product_code', 'product_name', 'total_value', 'supplier_count', 'first_year', 'last_year']
        top_import_products['total_value_usd'] = top_import_products['total_value'] * 1000
        top_import_products['supplier_span'] = top_import_products['last_year'] - top_import_products['first_year'] + 1
        top_import_products = top_import_products.sort_values('total_value_usd', ascending=False).head(25)
        
        return top_export_products, top_import_products
        
    def product_market_analysis(self):
        """পণ্য-বাজার বিশ্লেষণ"""
        print("পণ্য-বাজার ম্যাট্রিক্স বিশ্লেষণ করা হচ্ছে...")
        
        # Create product-market matrix
        product_market = self.bangladesh_exports.groupby(['k', 'description', 'j', 'importer_name'])['v'].sum().reset_index()
        product_market['value_usd'] = product_market['v'] * 1000
        
        # Get top 10 products and top 15 markets
        top_products = self.bangladesh_exports.groupby('k')['v'].sum().nlargest(10).index
        top_markets = self.bangladesh_exports.groupby('j')['v'].sum().nlargest(15).index
        
        # Filter matrix
        matrix_data = product_market[
            (product_market['k'].isin(top_products)) & 
            (product_market['j'].isin(top_markets))
        ]
        
        # Create pivot table
        matrix_pivot = matrix_data.pivot_table(
            index=['k', 'description'], 
            columns=['j', 'importer_name'], 
            values='value_usd', 
            fill_value=0
        )
        
        return matrix_pivot
        
    def trade_balance_by_partner(self):
        """অংশীদারভিত্তিক বাণিজ্য ভারসাম্য"""
        print("অংশীদারভিত্তিক বাণিজ্য ভারসাম্য বিশ্লেষণ করা হচ্ছে...")
        
        # Export values by partner
        exports_by_partner = self.bangladesh_exports.groupby(['j', 'importer_name'])['v'].sum().reset_index()
        exports_by_partner.columns = ['country_code', 'country_name', 'exports']
        exports_by_partner['exports_usd'] = exports_by_partner['exports'] * 1000
        
        # Import values by partner
        imports_by_partner = self.bangladesh_imports.groupby(['i', 'exporter_name'])['v'].sum().reset_index()
        imports_by_partner.columns = ['country_code', 'country_name', 'imports']
        imports_by_partner['imports_usd'] = imports_by_partner['imports'] * 1000
        
        # Merge and calculate trade balance
        trade_balance = exports_by_partner.merge(
            imports_by_partner, 
            on=['country_code', 'country_name'], 
            how='outer'
        ).fillna(0)
        
        trade_balance['trade_balance'] = trade_balance['exports_usd'] - trade_balance['imports_usd']
        trade_balance['total_trade'] = trade_balance['exports_usd'] + trade_balance['imports_usd']
        
        # Sort by total trade volume
        trade_balance = trade_balance.sort_values('total_trade', ascending=False).head(30)
        
        return trade_balance
        
    def generate_enhanced_html_report(self):
        """বর্ধিত HTML রিপোর্ট তৈরি করা"""
        print("বর্ধিত বাংলা HTML রিপোর্ট তৈরি করা হচ্ছে...")
        
        # Load and analyze data
        self.load_and_prepare_data()
        
        # Get all analyses
        top_export_partners, top_import_partners = self.get_top_trading_partners()
        emerging_exports, emerging_imports = self.identify_emerging_partners()
        top_export_products, top_import_products = self.analyze_top_products()
        trade_balance_partners = self.trade_balance_by_partner()
        
        # Calculate summary metrics
        total_exports = self.bangladesh_exports['v'].sum() * 1000
        total_imports = self.bangladesh_imports['v'].sum() * 1000
        trade_balance = total_exports - total_imports
        
        # Create visualizations
        # 1. Top Trading Partners Chart
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('শীর্ষ রপ্তানি গন্তব্য', 'শীর্ষ আমদানি উৎস', 'উদীয়মান রপ্তানি বাজার', 'উদীয়মান আমদানি উৎস'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Top export destinations
        fig1.add_trace(
            go.Bar(x=top_export_partners.head(10)['total_value_usd']/1e9, 
                   y=top_export_partners.head(10)['country_name'],
                   orientation='h', name='রপ্তানি (বিলিয়ন USD)', marker_color='green'),
            row=1, col=1
        )
        
        # Top import sources
        fig1.add_trace(
            go.Bar(x=top_import_partners.head(10)['total_value_usd']/1e9, 
                   y=top_import_partners.head(10)['country_name'],
                   orientation='h', name='আমদানি (বিলিয়ন USD)', marker_color='red'),
            row=1, col=2
        )
        
        # Emerging export markets
        if not emerging_exports.empty:
            fig1.add_trace(
                go.Scatter(x=emerging_exports['total_value']/1e6, 
                          y=emerging_exports['growth_rate'],
                          mode='markers+text',
                          text=emerging_exports['country_name'],
                          textposition="top center",
                          name='উদীয়মান রপ্তানি', marker_color='blue', marker_size=8),
                row=2, col=1
            )
        
        # Emerging import sources
        if not emerging_imports.empty:
            fig1.add_trace(
                go.Scatter(x=emerging_imports['total_value']/1e6, 
                          y=emerging_imports['growth_rate'],
                          mode='markers+text',
                          text=emerging_imports['country_name'],
                          textposition="top center",
                          name='উদীয়মান আমদানি', marker_color='orange', marker_size=8),
                row=2, col=2
            )
        
        fig1.update_layout(height=800, title_text="বাণিজ্যিক অংশীদার বিশ্লেষণ", showlegend=False)
        
        # 2. Top Products Analysis
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('শীর্ষ রপ্তানি পণ্য', 'শীর্ষ আমদানি পণ্য'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig2.add_trace(
            go.Bar(x=top_export_products.head(10)['total_value_usd']/1e9,
                   y=[name[:50] + '...' if len(str(name)) > 50 else str(name) for name in top_export_products.head(10)['product_name']],
                   orientation='h', name='রপ্তানি', marker_color='green'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Bar(x=top_import_products.head(10)['total_value_usd']/1e9,
                   y=[name[:50] + '...' if len(str(name)) > 50 else str(name) for name in top_import_products.head(10)['product_name']],
                   orientation='h', name='আমদানি', marker_color='red'),
            row=1, col=2
        )
        
        fig2.update_layout(height=600, title_text="শীর্ষ পণ্য বিশ্লেষণ", showlegend=False)
        
        # 3. Trade Balance by Partner
        fig3 = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in trade_balance_partners.head(15)['trade_balance']]
        
        fig3.add_trace(go.Bar(
            x=trade_balance_partners.head(15)['trade_balance']/1e9,
            y=trade_balance_partners.head(15)['country_name'],
            orientation='h',
            marker_color=colors,
            name='বাণিজ্য ভারসাম্য'
        ))
        
        fig3.update_layout(
            title="অংশীদারভিত্তিক বাণিজ্য ভারসাম্য (বিলিয়ন USD)",
            xaxis_title="বাণিজ্য ভারসাম্য (বিলিয়ন USD)",
            height=600
        )
        
        # Convert plots to HTML
        plot1_html = pyo.plot(fig1, output_type='div', include_plotlyjs=False)
        plot2_html = pyo.plot(fig2, output_type='div', include_plotlyjs=False)
        plot3_html = pyo.plot(fig3, output_type='div', include_plotlyjs=False)
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>বাংলাদেশ বাণিজ্য বিশ্লেষণ - বর্ধিত রিপোর্ট</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@300;400;500;600;700&display=swap');
        body {{
            font-family: 'Noto Sans Bengali', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.8;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            display: inline-block;
            width: 200px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .highlight-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #2196f3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇧🇩 বাংলাদেশের আন্তর্জাতিক বাণিজ্য</h1>
            <h2>বর্ধিত বিশ্লেষণ রিপোর্ট</h2>
            <p>বাণিজ্যিক অংশীদার, উদীয়মান বাজার এবং পণ্য বিশ্লেষণ</p>
            <p><em>{datetime.now().strftime('%B %d, %Y')} তারিখে তৈরি</em></p>
        </div>
        
        <div class="section">
            <h2>📊 মূল পরিসংখ্যান</h2>
            <div style="text-align: center;">
                <div class="metric-card">
                    <div class="metric-value">${total_exports/1000000000:.1f}B</div>
                    <div class="metric-label">মোট রপ্তানি</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${total_imports/1000000000:.1f}B</div>
                    <div class="metric-label">মোট আমদানি</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${trade_balance/1000000000:.1f}B</div>
                    <div class="metric-label">বাণিজ্য ভারসাম্য</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(top_export_partners)}</div>
                    <div class="metric-label">প্রধান রপ্তানি গন্তব্য</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(top_import_partners)}</div>
                    <div class="metric-label">প্রধান আমদানি উৎস</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🌍 বাণিজ্যিক অংশীদার বিশ্লেষণ</h2>
            <div class="plot-container">
                {plot1_html}
            </div>
            
            <div class="highlight-box">
                <h4>🎯 মূল অন্তর্দৃষ্টি</h4>
                <ul>
                    <li><strong>শীর্ষ রপ্তানি গন্তব্য:</strong> {top_export_partners.iloc[0]['country_name']} (${top_export_partners.iloc[0]['total_value_usd']/1e9:.1f}B)</li>
                    <li><strong>শীর্ষ আমদানি উৎস:</strong> {top_import_partners.iloc[0]['country_name']} (${top_import_partners.iloc[0]['total_value_usd']/1e9:.1f}B)</li>
                    <li><strong>উদীয়মান বাজার:</strong> {len(emerging_exports)} টি নতুন রপ্তানি সুযোগ চিহ্নিত</li>
                    <li><strong>নতুন সরবরাহকারী:</strong> {len(emerging_imports)} টি নতুন আমদানি উৎস</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📦 শীর্ষ বাণিজ্যিক অংশীদার</h2>
            
            <h3>🚀 রপ্তানি গন্তব্য</h3>
            <table>
                <tr><th>র‍্যাঙ্ক</th><th>দেশ</th><th>মূল্য (USD)</th><th>পণ্য সংখ্যা</th><th>বাণিজ্য সময়কাল</th></tr>
        """
        
        # Add top export partners table
        for i, row in top_export_partners.head(10).iterrows():
            html_content += f"""
                <tr>
                    <td>{top_export_partners.index.get_loc(i) + 1}</td>
                    <td>{row['country_name']}</td>
                    <td>${row['total_value_usd']:,.0f}</td>
                    <td>{row['product_count']}</td>
                    <td>{row['trade_duration']} বছর</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h3>📥 আমদানি উৎস</h3>
            <table>
                <tr><th>র‍্যাঙ্ক</th><th>দেশ</th><th>মূল্য (USD)</th><th>পণ্য সংখ্যা</th><th>বাণিজ্য সময়কাল</th></tr>
        """
        
        # Add top import partners table
        for i, row in top_import_partners.head(10).iterrows():
            html_content += f"""
                <tr>
                    <td>{top_import_partners.index.get_loc(i) + 1}</td>
                    <td>{row['country_name']}</td>
                    <td>${row['total_value_usd']:,.0f}</td>
                    <td>{row['product_count']}</td>
                    <td>{row['trade_duration']} বছর</td>
                </tr>
            """
        
        html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>🏭 পণ্য বিশ্লেষণ</h2>
            <div class="plot-container">
                {plot2_html}
            </div>
            
            <h3>📈 শীর্ষ রপ্তানি পণ্য</h3>
            <table>
                <tr><th>র‍্যাঙ্ক</th><th>পণ্যের নাম</th><th>মূল্য (USD)</th><th>বাজার সংখ্যা</th></tr>
        """
        
        # Add top export products table
        for i, row in top_export_products.head(10).iterrows():
            product_name = str(row['product_name'])[:80] + '...' if len(str(row['product_name'])) > 80 else str(row['product_name'])
            html_content += f"""
                <tr>
                    <td>{top_export_products.index.get_loc(i) + 1}</td>
                    <td>{product_name}</td>
                    <td>${row['total_value_usd']:,.0f}</td>
                    <td>{row['market_count']}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h3>📥 শীর্ষ আমদানি পণ্য</h3>
            <table>
                <tr><th>র‍্যাঙ্ক</th><th>পণ্যের নাম</th><th>মূল্য (USD)</th><th>সরবরাহকারী সংখ্যা</th></tr>
        """
        
        # Add top import products table
        for i, row in top_import_products.head(10).iterrows():
            product_name = str(row['product_name'])[:80] + '...' if len(str(row['product_name'])) > 80 else str(row['product_name'])
            html_content += f"""
                <tr>
                    <td>{top_import_products.index.get_loc(i) + 1}</td>
                    <td>{product_name}</td>
                    <td>${row['total_value_usd']:,.0f}</td>
                    <td>{row['supplier_count']}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>⚖️ অংশীদারভিত্তিক বাণিজ্য ভারসাম্য</h2>
            <div class="plot-container">
                {plot3_html}
            </div>
            
            <div class="highlight-box">
                <h4>💡 বাণিজ্য ভারসাম্য অন্তর্দৃষ্টি</h4>
                <ul>
                    <li><strong>উদ্বৃত্ত বাজার:</strong> {len(trade_balance_partners[trade_balance_partners['trade_balance'] > 0])} টি দেশের সাথে বাণিজ্য উদ্বৃত্ত</li>
                    <li><strong>ঘাটতি বাজার:</strong> {len(trade_balance_partners[trade_balance_partners['trade_balance'] < 0])} টি দেশের সাথে বাণিজ্য ঘাটতি</li>
                    <li><strong>সর্বোচ্চ উদ্বৃত্ত:</strong> {trade_balance_partners.loc[trade_balance_partners['trade_balance'].idxmax(), 'country_name']} (${trade_balance_partners['trade_balance'].max()/1e9:.1f}B)</li>
                    <li><strong>সর্বোচ্চ ঘাটতি:</strong> {trade_balance_partners.loc[trade_balance_partners['trade_balance'].idxmin(), 'country_name']} (${abs(trade_balance_partners['trade_balance'].min())/1e9:.1f}B)</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>🌟 উদীয়মান বাজার সুযোগ</h2>
        """
        
        if not emerging_exports.empty:
            html_content += f"""
            <h3>🚀 উদীয়মান রপ্তানি বাজার</h3>
            <table>
                <tr><th>দেশ</th><th>বৃদ্ধির হার (%)</th><th>বর্তমান মূল্য (USD)</th><th>সক্রিয় বছর</th></tr>
            """
            
            for _, row in emerging_exports.head(10).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['country_name']}</td>
                        <td>{row['growth_rate']:.1f}%</td>
                        <td>${row['total_value']:,.0f}</td>
                        <td>{row['years_active']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        if not emerging_imports.empty:
            html_content += f"""
            <h3>📥 উদীয়মান আমদানি উৎস</h3>
            <table>
                <tr><th>দেশ</th><th>বৃদ্ধির হার (%)</th><th>বর্তমান মূল্য (USD)</th><th>সক্রিয় বছর</th></tr>
            """
            
            for _, row in emerging_imports.head(10).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['country_name']}</td>
                        <td>{row['growth_rate']:.1f}%</td>
                        <td>${row['total_value']:,.0f}</td>
                        <td>{row['years_active']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        html_content += f"""
        </div>
        
        <div class="section">
            <h2>🎯 কৌশলগত সুপারিশ</h2>
            
            <div class="highlight-box">
                <h4>✅ রপ্তানি বৃদ্ধির সুযোগ</h4>
                <ul>
                    <li><strong>উদীয়মান বাজার:</strong> উচ্চ বৃদ্ধির হার সহ নতুন বাজারে মনোনিবেশ করুন</li>
                    <li><strong>পণ্য বৈচিত্র্যকরণ:</strong> বিদ্যমান বাজারে নতুন পণ্য প্রবর্তন করুন</li>
                    <li><strong>মূল্য সংযোজন:</strong> উচ্চ মূল্যের পণ্যে স্থানান্তর করুন</li>
                </ul>
            </div>
            
            <div class="highlight-box">
                <h4>⚠️ আমদানি অপ্টিমাইজেশন</h4>
                <ul>
                    <li><strong>সরবরাহকারী বৈচিত্র্যকরণ:</strong> একক উৎসের উপর নির্ভরতা কমান</li>
                    <li><strong>স্থানীয় বিকল্প:</strong> আমদানি প্রতিস্থাপনের সুযোগ খুঁজুন</li>
                    <li><strong>খরচ অপ্টিমাইজেশন:</strong> সাশ্রয়ী সরবরাহকারী চিহ্নিত করুন</li>
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <p><em>এই বর্ধিত রিপোর্টটি বাংলাদেশের আন্তর্জাতিক বাণিজ্যের গভীর বিশ্লেষণ প্রদান করে এবং 
            কৌশলগত সিদ্ধান্ত গ্রহণে সহায়তা করে।</em></p>
            <p><strong>রিপোর্ট তৈরি:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        with open('bangladesh_enhanced_trade_analysis_bengali.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print("✅ বর্ধিত বাংলা HTML রিপোর্ট তৈরি: bangladesh_enhanced_trade_analysis_bengali.html")
        return html_content

# Run the enhanced analysis
if __name__ == "__main__":
    analyzer = EnhancedBengaliTradeAnalyzer()
    analyzer.generate_enhanced_html_report() 