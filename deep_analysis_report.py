import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DeepTradeAnalyzer:
    def __init__(self):
        self.trade_data = None
        self.country_codes = None
        self.product_codes = None
        self.bangladesh_exports = None
        self.bangladesh_imports = None
        self.html_content = ""
        
    def load_and_prepare_data(self):
        """Load and prepare all datasets"""
        print("Loading datasets for deep analysis...")
        
        # Load datasets
        self.trade_data = pd.read_csv('bd_trade_data.csv')
        self.country_codes = pd.read_csv('country_codes_V202501.csv')
        self.product_codes = pd.read_csv('product_codes_HS92_V202501.csv')
        
        # Merge data
        self.trade_data = self.trade_data.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='i', right_on='country_code', how='left'
        ).rename(columns={'country_name': 'exporter_name', 'country_iso3': 'exporter_iso3'})
        
        self.trade_data = self.trade_data.merge(
            self.country_codes[['country_code', 'country_name', 'country_iso3']], 
            left_on='j', right_on='country_code', how='left', suffixes=('', '_imp')
        ).rename(columns={'country_name': 'importer_name', 'country_iso3': 'importer_iso3'})
        
        self.trade_data['k_str'] = self.trade_data['k'].astype(str).str.zfill(6)
        self.trade_data = self.trade_data.merge(
            self.product_codes, left_on='k_str', right_on='code', how='left'
        )
        
        # Separate Bangladesh trade flows
        self.bangladesh_exports = self.trade_data[self.trade_data['i'] == 50].copy()
        self.bangladesh_imports = self.trade_data[self.trade_data['j'] == 50].copy()
        
        # Add derived variables
        self.bangladesh_exports['hs2'] = self.bangladesh_exports['k'].astype(str).str[:2]
        self.bangladesh_imports['hs2'] = self.bangladesh_imports['k'].astype(str).str[:2]
        
        # Calculate unit values only for records with positive quantities
        self.bangladesh_exports['unit_value'] = np.where(
            self.bangladesh_exports['q'] > 0,
            self.bangladesh_exports['v'] / self.bangladesh_exports['q'],
            np.nan
        )
        self.bangladesh_imports['unit_value'] = np.where(
            self.bangladesh_imports['q'] > 0,
            self.bangladesh_imports['v'] / self.bangladesh_imports['q'],
            np.nan
        )
        
        print(f"Data loaded: {len(self.trade_data):,} total records")
        print(f"Bangladesh exports: {len(self.bangladesh_exports):,} records")
        print(f"Bangladesh imports: {len(self.bangladesh_imports):,} records")
        
    def statistical_analysis(self):
        """Perform advanced statistical analysis"""
        print("Performing statistical analysis...")
        
        # Basic statistics (convert to actual USD)
        export_values = self.bangladesh_exports['v'] * 1000
        import_values = self.bangladesh_imports['v'] * 1000
        export_stats = export_values.describe()
        import_stats = import_values.describe()
        
        # Distribution analysis (using actual USD values)
        export_skewness = stats.skew(export_values)
        export_kurtosis = stats.kurtosis(export_values)
        import_skewness = stats.skew(import_values)
        import_kurtosis = stats.kurtosis(import_values)
        
        # Concentration analysis (Herfindahl-Hirschman Index)
        export_country_shares = self.bangladesh_exports.groupby('j')['v'].sum() / self.bangladesh_exports['v'].sum()
        export_hhi = (export_country_shares ** 2).sum()
        
        import_country_shares = self.bangladesh_imports.groupby('i')['v'].sum() / self.bangladesh_imports['v'].sum()
        import_hhi = (import_country_shares ** 2).sum()
        
        export_product_shares = self.bangladesh_exports.groupby('hs2')['v'].sum() / self.bangladesh_exports['v'].sum()
        export_product_hhi = (export_product_shares ** 2).sum()
        
        import_product_shares = self.bangladesh_imports.groupby('hs2')['v'].sum() / self.bangladesh_imports['v'].sum()
        import_product_hhi = (import_product_shares ** 2).sum()
        
        return {
            'export_stats': export_stats,
            'import_stats': import_stats,
            'export_skewness': export_skewness,
            'export_kurtosis': export_kurtosis,
            'import_skewness': import_skewness,
            'import_kurtosis': import_kurtosis,
            'export_country_hhi': export_hhi,
            'import_country_hhi': import_hhi,
            'export_product_hhi': export_product_hhi,
            'import_product_hhi': import_product_hhi
        }
        
    def trade_complexity_analysis(self):
        """Analyze trade complexity and sophistication"""
        print("Analyzing trade complexity...")
        
        # Product complexity (based on number of exporters)
        product_complexity = self.trade_data.groupby('k')['i'].nunique().reset_index()
        product_complexity.columns = ['product', 'num_exporters']
        product_complexity['complexity_score'] = 1 / product_complexity['num_exporters']
        
        # Bangladesh's export complexity
        bd_export_complexity = self.bangladesh_exports.merge(
            product_complexity, left_on='k', right_on='product', how='left'
        )
        
        avg_export_complexity = (bd_export_complexity['v'] * bd_export_complexity['complexity_score']).sum() / bd_export_complexity['v'].sum()
        
        # Market penetration analysis
        total_importers_by_product = self.trade_data.groupby('k')['j'].nunique().reset_index()
        total_importers_by_product.columns = ['product', 'total_importers']
        
        bd_importers_by_product = self.bangladesh_exports.groupby('k')['j'].nunique().reset_index()
        bd_importers_by_product.columns = ['product', 'bd_importers']
        
        market_penetration = bd_importers_by_product.merge(
            total_importers_by_product, on='product', how='left'
        )
        market_penetration['penetration_rate'] = market_penetration['bd_importers'] / market_penetration['total_importers']
        
        return {
            'avg_export_complexity': avg_export_complexity,
            'market_penetration': market_penetration,
            'product_complexity': product_complexity
        }
        
    def clustering_analysis(self):
        """Perform clustering analysis on trading partners"""
        print("Performing clustering analysis...")
        
        # Prepare data for clustering - export partners
        export_partner_data = self.bangladesh_exports.groupby(['j', 'importer_name']).agg({
            'v': 'sum',
            'q': 'sum',
            'k': 'nunique'
        }).reset_index()
        export_partner_data.columns = ['country_code', 'country_name', 'total_value', 'total_quantity', 'product_diversity']
        export_partner_data['unit_value'] = export_partner_data['total_value'] / (export_partner_data['total_quantity'] + 0.001)
        
        # Standardize features for clustering
        features = ['total_value', 'total_quantity', 'product_diversity', 'unit_value']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(export_partner_data[features].fillna(0))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        export_partner_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)
        export_partner_data['pca1'] = pca_features[:, 0]
        export_partner_data['pca2'] = pca_features[:, 1]
        
        return export_partner_data, pca.explained_variance_ratio_
        
    def time_series_analysis(self):
        """Analyze temporal trends and patterns"""
        print("Analyzing time series patterns...")
        
        # Yearly trends (convert to actual USD by multiplying by 1000)
        yearly_exports = self.bangladesh_exports.groupby('t')['v'].sum().reset_index()
        yearly_imports = self.bangladesh_imports.groupby('t')['v'].sum().reset_index()
        yearly_exports['v'] = yearly_exports['v'] * 1000
        yearly_imports['v'] = yearly_imports['v'] * 1000
        
        # Growth rates
        yearly_exports['export_growth'] = yearly_exports['v'].pct_change() * 100
        yearly_imports['import_growth'] = yearly_imports['v'].pct_change() * 100
        
        # Product lifecycle analysis (convert to actual USD)
        product_trends = self.bangladesh_exports.groupby(['t', 'hs2'])['v'].sum().reset_index()
        product_trends['v'] = product_trends['v'] * 1000
        product_pivot = product_trends.pivot(index='t', columns='hs2', values='v').fillna(0)
        
        return yearly_exports, yearly_imports, product_pivot
        
    def competitive_analysis(self):
        """Analyze Bangladesh's competitive position"""
        print("Analyzing competitive position...")
        
        # Market share analysis for top products
        top_export_products = self.bangladesh_exports.groupby('k')['v'].sum().nlargest(20).index
        
        competitive_data = []
        for product in top_export_products:
            # Global trade in this product (all exporters)
            global_product_trade = self.trade_data[self.trade_data['k'] == product]
            # Group by exporter (i) to get total exports by each country
            global_exports_by_country = global_product_trade.groupby('i')['v'].sum()
            total_global_exports = global_exports_by_country.sum()
            
            # Bangladesh's exports for this product (convert to actual USD)
            bd_exports = self.bangladesh_exports[self.bangladesh_exports['k'] == product]['v'].sum() * 1000
            total_global_exports = total_global_exports * 1000
            market_share = bd_exports / total_global_exports if total_global_exports > 0 else 0
            
            # Top competitors (excluding Bangladesh if it appears)
            competitors = global_exports_by_country.drop(50, errors='ignore').nlargest(5)
            
            competitive_data.append({
                'product': product,
                'bd_exports': bd_exports,
                'global_exports': total_global_exports,
                'market_share': market_share,
                'top_competitors': competitors
            })
            
        return competitive_data
        
    def create_interactive_visualizations(self, stats, complexity, clustering_data, time_data, competitive_data):
        """Create interactive Plotly visualizations"""
        print("Creating interactive visualizations...")
        
        # 1. Trade Overview Dashboard
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trade Balance Trend', 'Export Concentration (HHI)', 
                          'Product Complexity Distribution', 'Market Penetration'),
            specs=[[{"secondary_y": True}, {"type": "indicator"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Trade balance trend
        yearly_exports, yearly_imports, _ = time_data
        fig1.add_trace(
            go.Scatter(x=yearly_exports['t'], y=yearly_exports['v'], 
                      name='Exports', line=dict(color='green')),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(x=yearly_imports['t'], y=yearly_imports['v'], 
                      name='Imports', line=dict(color='red')),
            row=1, col=1
        )
        
        # HHI Indicator
        fig1.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=stats['export_country_hhi'],
                title={"text": "Export Concentration (HHI)"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.15], 'color': "lightgray"},
                                {'range': [0.15, 0.25], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.25}}
            ),
            row=1, col=2
        )
        
        # Product complexity histogram
        fig1.add_trace(
            go.Histogram(x=complexity['product_complexity']['complexity_score'], 
                        name='Product Complexity'),
            row=2, col=1
        )
        
        # Clustering visualization
        fig1.add_trace(
            go.Scatter(x=clustering_data['pca1'], y=clustering_data['pca2'],
                      mode='markers', 
                      marker=dict(color=clustering_data['cluster'], 
                                 colorscale='viridis', size=8),
                      text=clustering_data['country_name'],
                      name='Trading Partners'),
            row=2, col=2
        )
        
        fig1.update_layout(height=800, title_text="Bangladesh Trade Analysis Dashboard")
        
        # 2. Product Analysis
        top_exports = self.bangladesh_exports.groupby(['hs2', 'description'])['v'].sum().nlargest(15)
        
        fig2 = px.treemap(
            values=top_exports.values,
            parents=[""] * len(top_exports),
            names=[f"{idx[0]}: {idx[1][:30]}..." if len(idx[1]) > 30 else f"{idx[0]}: {idx[1]}" for idx in top_exports.index],
            title="Export Product Portfolio (Treemap)"
        )
        
        # 3. Geographic Analysis
        export_by_country = self.bangladesh_exports.groupby(['importer_iso3', 'importer_name'])['v'].sum().reset_index()
        
        fig3 = px.choropleth(
            export_by_country,
            locations='importer_iso3',
            color='v',
            hover_name='importer_name',
            color_continuous_scale='Viridis',
            title="Bangladesh Export Destinations (World Map)"
        )
        
        # 4. Competitive Position
        comp_df = pd.DataFrame(competitive_data)
        fig4 = px.scatter(
            comp_df, x='global_exports', y='market_share',
            size='bd_exports', hover_data=['product'],
            title="Competitive Position Analysis",
            labels={'global_exports': 'Global Market Size', 'market_share': 'Bangladesh Market Share'}
        )
        
        return fig1, fig2, fig3, fig4
        
    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        print("Generating HTML report...")
        
        # Perform all analyses
        self.load_and_prepare_data()
        stats = self.statistical_analysis()
        complexity = self.trade_complexity_analysis()
        clustering_data, pca_variance = self.clustering_analysis()
        time_data = self.time_series_analysis()
        competitive_data = self.competitive_analysis()
        
        # Create visualizations
        fig1, fig2, fig3, fig4 = self.create_interactive_visualizations(
            stats, complexity, clustering_data, time_data, competitive_data
        )
        
        # Convert plots to HTML
        plot1_html = pyo.plot(fig1, output_type='div', include_plotlyjs=False)
        plot2_html = pyo.plot(fig2, output_type='div', include_plotlyjs=False)
        plot3_html = pyo.plot(fig3, output_type='div', include_plotlyjs=False)
        plot4_html = pyo.plot(fig4, output_type='div', include_plotlyjs=False)
        
        # Calculate key metrics (v is in thousands USD, so multiply by 1000)
        total_exports = self.bangladesh_exports['v'].sum() * 1000
        total_imports = self.bangladesh_imports['v'].sum() * 1000
        trade_balance = total_exports - total_imports
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangladesh Trade Analysis - Deep Dive Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
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
        .insight-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #2196f3;
        }}
        .warning-box {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }}
        .success-box {{
            background: #d4edda;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
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
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇧🇩 Bangladesh International Trade</h1>
            <h2>Deep Analysis Report</h2>
            <p>Comprehensive analysis of trade patterns, competitiveness, and strategic insights</p>
            <p><em>Generated on {datetime.now().strftime('%B %d, %Y')}</em></p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div style="text-align: center;">
                <div class="metric-card">
                    <div class="metric-value">${total_exports/1000000000:.1f}B</div>
                    <div class="metric-label">Total Exports</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${total_imports/1000000000:.1f}B</div>
                    <div class="metric-label">Total Imports</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${trade_balance/1000000000:.1f}B</div>
                    <div class="metric-label">Trade Balance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.bangladesh_exports['j'].unique())}</div>
                    <div class="metric-label">Export Markets</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.bangladesh_exports['k'].unique())}</div>
                    <div class="metric-label">Export Products</div>
                </div>
            </div>
            
            <div class="insight-box">
                <h4>🎯 Key Findings</h4>
                <ul>
                    <li><strong>Trade Position:</strong> Bangladesh maintains a trade deficit of ${trade_balance/1000000000:.1f}B, indicating higher import dependency</li>
                    <li><strong>Export Concentration:</strong> HHI of {stats['export_country_hhi']:.3f} shows {'high' if stats['export_country_hhi'] > 0.25 else 'moderate' if stats['export_country_hhi'] > 0.15 else 'low'} geographic concentration</li>
                    <li><strong>Product Diversification:</strong> HHI of {stats['export_product_hhi']:.3f} indicates {'high' if stats['export_product_hhi'] > 0.25 else 'moderate' if stats['export_product_hhi'] > 0.15 else 'low'} product concentration</li>
                    <li><strong>Market Reach:</strong> Exports to {len(self.bangladesh_exports['j'].unique())} countries demonstrate strong global presence</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Interactive Dashboard</h2>
            <div class="plot-container">
                {plot1_html}
            </div>
        </div>
        
        <div class="section">
            <h2>🏭 Product Portfolio Analysis</h2>
            <div class="plot-container">
                {plot2_html}
            </div>
            
            <div class="insight-box">
                <h4>💡 Product Insights</h4>
                <p><strong>Export Complexity Score:</strong> {complexity['avg_export_complexity']:.4f}</p>
                <p>This score indicates the sophistication level of Bangladesh's export basket. Higher scores suggest more complex, knowledge-intensive products.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🌍 Geographic Distribution</h2>
            <div class="plot-container">
                {plot3_html}
            </div>
        </div>
        
        <div class="section">
            <h2>🏆 Competitive Position Analysis</h2>
            <div class="plot-container">
                {plot4_html}
            </div>
            
            <div class="warning-box">
                <h4>⚠️ Strategic Considerations</h4>
                <ul>
                    <li><strong>Market Concentration Risk:</strong> High dependence on few markets increases vulnerability</li>
                    <li><strong>Product Sophistication:</strong> Focus on moving up the value chain in existing products</li>
                    <li><strong>Diversification Opportunities:</strong> Explore new markets and products to reduce risk</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Statistical Analysis</h2>
            <table>
                <tr><th>Metric</th><th>Exports</th><th>Imports</th></tr>
                <tr><td>Mean Transaction Value</td><td>${stats['export_stats']['mean']:,.0f}</td><td>${stats['import_stats']['mean']:,.0f}</td></tr>
                <tr><td>Median Transaction Value</td><td>${stats['export_stats']['50%']:,.0f}</td><td>${stats['import_stats']['50%']:,.0f}</td></tr>
                <tr><td>Standard Deviation</td><td>${stats['export_stats']['std']:,.0f}</td><td>${stats['import_stats']['std']:,.0f}</td></tr>
                <tr><td>Skewness</td><td>{stats['export_skewness']:.2f}</td><td>{stats['import_skewness']:.2f}</td></tr>
                <tr><td>Kurtosis</td><td>{stats['export_kurtosis']:.2f}</td><td>{stats['import_kurtosis']:.2f}</td></tr>
            </table>
            
            <div class="insight-box">
                <h4>📈 Distribution Analysis</h4>
                <p>The high skewness values indicate that both export and import transactions are heavily right-skewed, 
                meaning most transactions are small with few very large transactions dominating the trade volumes.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Strategic Recommendations</h2>
            
            <div class="success-box">
                <h4>✅ Strengths to Leverage</h4>
                <ul>
                    <li><strong>Global Reach:</strong> Strong presence in {len(self.bangladesh_exports['j'].unique())} markets</li>
                    <li><strong>Product Range:</strong> Diverse portfolio with {len(self.bangladesh_exports['k'].unique())} different products</li>
                    <li><strong>Established Relationships:</strong> Strong ties with major economies (USA, EU, China)</li>
                </ul>
            </div>
            
            <div class="warning-box">
                <h4>⚠️ Areas for Improvement</h4>
                <ul>
                    <li><strong>Trade Balance:</strong> Address ${trade_balance/1000000000:.1f}B deficit through export promotion</li>
                    <li><strong>Product Sophistication:</strong> Move towards higher value-added products</li>
                    <li><strong>Market Diversification:</strong> Reduce dependence on top markets</li>
                </ul>
            </div>
            
            <div class="insight-box">
                <h4>🚀 Strategic Actions</h4>
                <ol>
                    <li><strong>Export Promotion:</strong> Focus on high-growth, high-value markets</li>
                    <li><strong>Product Development:</strong> Invest in R&D for product sophistication</li>
                    <li><strong>Market Intelligence:</strong> Develop better understanding of emerging markets</li>
                    <li><strong>Trade Facilitation:</strong> Improve logistics and reduce trade costs</li>
                    <li><strong>Capacity Building:</strong> Enhance export capabilities of SMEs</li>
                </ol>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Methodology & Data Sources</h2>
            <p><strong>Data Sources:</strong></p>
            <ul>
                <li>Bangladesh Trade Data: {len(self.trade_data):,} records</li>
                <li>Country Classification: {len(self.country_codes)} countries</li>
                <li>Product Classification: {len(self.product_codes)} HS6 products</li>
            </ul>
            
            <p><strong>Analytical Methods:</strong></p>
            <ul>
                <li>Descriptive Statistics & Distribution Analysis</li>
                <li>Concentration Analysis (Herfindahl-Hirschman Index)</li>
                <li>Product Complexity Analysis</li>
                <li>K-means Clustering of Trading Partners</li>
                <li>Principal Component Analysis (PCA)</li>
                <li>Competitive Position Analysis</li>
                <li>Time Series Trend Analysis</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <p><em>This report was generated using advanced statistical and machine learning techniques to provide 
            comprehensive insights into Bangladesh's international trade patterns and competitive position.</em></p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        with open('bangladesh_deep_trade_analysis.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print("✅ Deep analysis HTML report generated: bangladesh_deep_trade_analysis.html")
        return html_content

# Run the deep analysis
if __name__ == "__main__":
    analyzer = DeepTradeAnalyzer()
    analyzer.generate_html_report()