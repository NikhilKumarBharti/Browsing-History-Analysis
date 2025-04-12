import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import networkx as nx

class BrowsingHistoryAnalyzer:
    """
    A class to analyze and visualize browsing history data
    """
    
    def __init__(self, csv_path=None, df=None):
        """
        Initialize with either a CSV file path or a pandas DataFrame
        """
        if df is not None:
            self.raw_data = df
        elif csv_path:
            self.raw_data = pd.read_csv(csv_path, skiprows=9)  # Skip the summary section
        else:
            raise ValueError("Either csv_path or df must be provided")
        
        self.summary_data = None
        self.browsing_data = None
        self.clean_data()
        
    # def clean_data(self):
    #     """
    #     Clean and preprocess the data for analysis
    #     """
    #     # Extract browsing data (skipping summary rows)
    #     self.browsing_data = self.raw_data.copy()
        
    #     # Convert timestamps to datetime objects
    #     self.browsing_data['eventtimeutc'] = pd.to_datetime(self.browsing_data['eventtimeutc'])
    #     self.browsing_data['eventtime'] = pd.to_datetime(self.browsing_data['eventtime'])
        
    #     # Extract domain from URLs
    #     self.browsing_data['domain'] = self.browsing_data['url'].apply(
    #         lambda x: urlparse(x).netloc if pd.notna(x) else None
    #     )
        
    #     # Create a simplified category column for the URLs
    #     self.browsing_data['category'] = self.browsing_data['url'].apply(self.categorize_url)
        
    #     # Sort by timestamp
    #     self.browsing_data = self.browsing_data.sort_values('eventtimeutc')
        
    #     # Calculate time spent on each page
    #     self.browsing_data['next_timestamp'] = self.browsing_data['eventtimeutc'].shift(-1)
    #     self.browsing_data['time_spent'] = (
    #         self.browsing_data['next_timestamp'] - self.browsing_data['eventtimeutc']
    #     ).apply(lambda x: x.total_seconds() if pd.notna(x) else 0)
        
    #     # Cap unreasonable time spent values (more than 30 minutes)
    #     self.browsing_data.loc[self.browsing_data['time_spent'] > 1800, 'time_spent'] = 0
        
    def clean_data(self):
        """
        Clean and preprocess the data for analysis
        """
        # Extract browsing data (skipping summary rows)
        self.browsing_data = self.raw_data.copy()
        
        # Drop any NaN rows that might exist in a larger dataset
        self.browsing_data = self.browsing_data.dropna(subset=['url', 'eventtimeutc'])
        
        # Convert timestamps to datetime objects
        self.browsing_data['eventtimeutc'] = pd.to_datetime(self.browsing_data['eventtimeutc'], errors='coerce')
        self.browsing_data['eventtime'] = pd.to_datetime(self.browsing_data['eventtime'], errors='coerce')
        
        # Drop rows with invalid timestamps after conversion
        self.browsing_data = self.browsing_data.dropna(subset=['eventtimeutc'])
        
        # Extract domain from URLs
        self.browsing_data['domain'] = self.browsing_data['url'].apply(
            lambda x: urlparse(x).netloc if pd.notna(x) else None
        )
        
        # Create a simplified category column for the URLs
        self.browsing_data['category'] = self.browsing_data['url'].apply(self.categorize_url)
        
        # Sort by timestamp
        self.browsing_data = self.browsing_data.sort_values('eventtimeutc')
        
        # Calculate time spent on each page (with safeguards for large datasets)
        self.browsing_data['next_timestamp'] = self.browsing_data['eventtimeutc'].shift(-1)
        
        # For large datasets, limit time calculation to reasonable sessions (same day)
        self.browsing_data['same_session'] = (
            (self.browsing_data['next_timestamp'] - self.browsing_data['eventtimeutc']) < pd.Timedelta(hours=2)
        )
        
        # Calculate time spent only for same session pages
        self.browsing_data['time_spent'] = np.where(
            self.browsing_data['same_session'],
            (self.browsing_data['next_timestamp'] - self.browsing_data['eventtimeutc']).dt.total_seconds(),
            0
        )
        
        # Cap unreasonable time spent values (more than 30 minutes)
        self.browsing_data.loc[self.browsing_data['time_spent'] > 1800, 'time_spent'] = 0


    def categorize_url(self, url):
        """
        Categorize URLs into meaningful groups for a larger dataset
        """
        if pd.isna(url):
            return "Unknown"
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()  # Convert to lowercase for consistent matching
        path = parsed_url.path.lower()      # Convert to lowercase for consistent matching
        
        # More comprehensive categorization for larger datasets
        if 'chrome-extension' in domain or 'chromewebstore' in domain:
            return "Browser Extension"
        elif 'google' in domain:
            if 'mail' in domain or 'gmail' in domain:
                return "Email"
            elif 'docs' in domain:
                return "Productivity"
            elif 'drive' in domain:
                return "Storage"
            elif 'calendar' in domain:
                return "Calendar"
            elif 'meet' in domain:
                return "Video Conferencing"
            else:
                return "Google"
        elif 'microsoft' in domain or 'office' in domain or 'live' in domain or 'outlook' in domain:
            return "Microsoft Services"
        elif 'youtube' in domain:
            return "Video"
        elif 'linkedin' in domain:
            return "Professional Social Media"
        elif 'facebook' in domain or 'instagram' in domain or 'twitter' in domain or 'x.com' in domain:
            return "Social Media"
        elif 'amazon' in domain or 'ebay' in domain or 'walmart' in domain or 'shop' in domain:
            return "Shopping"
        elif 'py-insights.com' in domain:
            if 'product' in path:
                return "PY Insights Product"
            elif 'myquotes' in path:
                return "PY Insights Quotes"
            elif 'account' in path:
                return "PY Insights Account"
            else:
                return "PY Insights Other"
        elif 'salesforce' in domain:
            return "CRM"
        elif 'github' in domain or 'gitlab' in domain or 'bitbucket' in domain:
            return "Development"
        elif 'stackoverflow' in domain or 'docs.' in domain:
            return "Technical Documentation"
        elif 'zoom' in domain or 'teams' in domain or 'webex' in domain:
            return "Video Conferencing"
        elif 'news' in domain or 'cnn' in domain or 'bbc' in domain:
            return "News"
        else:
            # Try to extract TLD for unknown domains
            tld_parts = domain.split('.')
            if len(tld_parts) > 1:
                return f"Other (.{tld_parts[-1]})"
            return "Other"
    
    def get_time_series_data(self):
        """
        Aggregate browsing data by minute for time series analysis
        """
        # Create a timestamp rounded to the minute
        self.browsing_data['minute'] = self.browsing_data['eventtimeutc'].dt.floor('min')
        
        # Count visits per minute
        time_series = self.browsing_data.groupby('minute').size().reset_index(name='visits')
        return time_series
    
    def get_domain_statistics(self):
        """
        Get statistics about domains visited
        """
        domain_stats = self.browsing_data.groupby('domain').agg({
            'url': 'count',
            'time_spent': 'sum'
        }).reset_index()
        
        domain_stats.columns = ['domain', 'visit_count', 'time_spent_seconds']
        domain_stats['time_spent_minutes'] = domain_stats['time_spent_seconds'] / 60
        
        return domain_stats.sort_values('visit_count', ascending=False)
    
    def get_category_statistics(self):
        """
        Get statistics about URL categories
        """
        category_stats = self.browsing_data.groupby('category').agg({
            'url': 'count',
            'time_spent': 'sum'
        }).reset_index()
        
        category_stats.columns = ['category', 'visit_count', 'time_spent_seconds']
        category_stats['time_spent_minutes'] = category_stats['time_spent_seconds'] / 60
        
        return category_stats.sort_values('visit_count', ascending=False)
    
    def get_transition_matrix(self):
        """
        Create a transition matrix between domains
        """
        # Create pairs of consecutive domains
        transitions = []
        domains = self.browsing_data['domain'].tolist()
        
        for i in range(len(domains) - 1):
            transitions.append((domains[i], domains[i+1]))
        
        # Count transitions
        transition_counts = Counter(transitions)
        
        # Create a set of all domains
        all_domains = set(domains)
        
        # Initialize matrix
        matrix = {d1: {d2: 0 for d2 in all_domains} for d1 in all_domains}
        
        # Fill matrix
        for (d1, d2), count in transition_counts.items():
            matrix[d1][d2] = count
            
        return pd.DataFrame(matrix)
    
    def visualize_time_series(self):
        """
        Create a time series visualization of browsing activity
        Optimized for larger datasets by aggregating by hour
        """
        # For larger datasets, group by hour instead of minute
        self.browsing_data['hour'] = self.browsing_data['eventtimeutc'].dt.floor('H')
        time_data = self.browsing_data.groupby('hour').size().reset_index(name='visits')
        
        # If dataset spans multiple days, add date to the x-axis label
        if time_data['hour'].dt.date.nunique() > 1:
            time_data['label'] = time_data['hour'].dt.strftime('%Y-%m-%d %H:%M')
        else:
            time_data['label'] = time_data['hour'].dt.strftime('%H:%M')
        
        plt.figure(figsize=(14, 6))
        plt.plot(time_data['hour'], time_data['visits'], marker='o', linestyle='-')
        plt.title('Browsing Activity Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Page Visits', fontsize=12)
        
        # Set x-axis labels based on the data range
        if len(time_data) > 24:
            # For many data points, show fewer labels
            plt.xticks(time_data['hour'][::6], time_data['label'][::6], rotation=45)
        else:
            plt.xticks(time_data['hour'], time_data['label'], rotation=45)
        
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        return plt
    
    def visualize_domains_bar(self):
        """
        Create a bar chart of domains visited
        Optimized for larger datasets by limiting to top domains
        """
        domain_stats = self.get_domain_statistics()
        
        # For large datasets, limit to top 15 domains
        top_domains = domain_stats.head(15)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='visit_count', y='domain', data=top_domains)
        plt.title('Top Domains by Number of Visits', fontsize=16)
        plt.xlabel('Number of Visits', fontsize=12)
        plt.ylabel('Domain', fontsize=12)
        plt.tight_layout()
        return plt
    
    def visualize_categories_pie(self):
        """
        Create a pie chart of URL categories
        Optimized for larger datasets by combining small categories
        """
        category_stats = self.get_category_statistics()
        
        # For large datasets with many categories, combine small ones
        if len(category_stats) > 8:
            threshold = category_stats['visit_count'].sum() * 0.02  # 2% threshold
            small_categories = category_stats[category_stats['visit_count'] < threshold]
            
            if not small_categories.empty:
                # Create a new row for "Other" categories
                other_row = pd.DataFrame({
                    'category': ['Other (Minor Categories)'],
                    'visit_count': [small_categories['visit_count'].sum()],
                    'time_spent_seconds': [small_categories['time_spent_seconds'].sum()],
                    'time_spent_minutes': [small_categories['time_spent_minutes'].sum()]
                })
                
                # Remove small categories and add the "Other" row
                category_stats = category_stats[category_stats['visit_count'] >= threshold]
                category_stats = pd.concat([category_stats, other_row], ignore_index=True)
        
        plt.figure(figsize=(12, 12))
        plt.pie(category_stats['visit_count'], labels=category_stats['category'], 
                autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('Distribution of Browsing Activity by Category', fontsize=16)
        plt.axis('equal')
        return plt
    
    def visualize_time_spent_bar(self):
        """
        Create a bar chart of time spent by category
        """
        category_stats = self.get_category_statistics()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='time_spent_minutes', y='category', data=category_stats)
        plt.title('Time Spent by Category (Minutes)', fontsize=16)
        plt.xlabel('Time Spent (Minutes)', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        return plt
    
    def visualize_transition_network(self):
        """
        Create a network visualization of domain transitions
        Optimized for larger datasets by focusing on significant transitions
        """
        # Get domain transitions
        transitions = []
        domains = self.browsing_data['domain'].tolist()
        
        for i in range(len(domains) - 1):
            if domains[i] != domains[i+1]:  # Only consider transitions between different domains
                transitions.append((domains[i], domains[i+1]))
        
        # Count transitions
        transition_counts = Counter(transitions)
        
        # For large datasets, filter to only include significant transitions
        if len(transition_counts) > 30:  # If there are many transitions
            # Find a reasonable threshold (e.g., transitions that happened at least twice)
            threshold = 2
            transition_counts = {k: v for k, v in transition_counts.items() if v >= threshold}
        
        # Create graph with only significant domains
        G = nx.DiGraph()
        
        # Add nodes for domains that appear in significant transitions
        significant_domains = set()
        for (source, target) in transition_counts.keys():
            significant_domains.add(source)
            significant_domains.add(target)
        
        for domain in significant_domains:
            G.add_node(domain)
        
        # Add edges with weights
        for (source, target), weight in transition_counts.items():
            G.add_edge(source, target, weight=weight)
        
        plt.figure(figsize=(14, 10))
        
        # For large graphs, use a different layout algorithm
        if len(G.nodes) > 20:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with size based on frequency
        node_sizes = []
        for node in G.nodes():
            count = self.browsing_data[self.browsing_data['domain'] == node].shape[0]
            node_sizes.append(max(1000, min(5000, count * 100)))
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8)
        
        # Draw edges with width based on transition count
        edges = G.edges(data=True)
        weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]  # Scale down weights for larger datasets
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color="gray", 
                            arrowsize=15, connectionstyle="arc3,rad=0.1")
        
        # For large graphs, show fewer labels to prevent overcrowding
        if len(G.nodes) > 20:
            # Only label the top N domains by frequency
            top_domains = self.browsing_data['domain'].value_counts().head(10).index.tolist()
            labels = {node: node if node in top_domains else '' for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family="sans-serif")
        else:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        plt.title("Domain Transition Network", fontsize=16)
        plt.axis("off")
        return plt

    
    def visualize_heatmap_by_time(self):
        """
        Create a heatmap of browsing activity by hour and category
        Optimized for larger datasets by using more appropriate time buckets
        """
        # Extract hour from timestamp
        self.browsing_data['hour'] = self.browsing_data['eventtimeutc'].dt.hour
        
        # For multi-day datasets, also consider the day
        date_range = (self.browsing_data['eventtimeutc'].max() - 
                    self.browsing_data['eventtimeutc'].min()).days
        
        if date_range > 0:
            # If data spans multiple days, use day of week instead
            self.browsing_data['day_of_week'] = self.browsing_data['eventtimeutc'].dt.day_name()
            
            # Count visits by day and category
            heatmap_data = self.browsing_data.pivot_table(
                index='category', 
                columns='day_of_week', 
                values='url', 
                aggfunc='count',
                fill_value=0
            )
            
            # Reorder days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            existing_days = [day for day in day_order if day in heatmap_data.columns]
            heatmap_data = heatmap_data[existing_days]
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='g')
            plt.title('Browsing Activity by Day of Week and Category', fontsize=16)
            plt.xlabel('Day of Week', fontsize=12)
            
        else:
            # For single day datasets, use hours
            heatmap_data = self.browsing_data.pivot_table(
                index='category', 
                columns='hour', 
                values='url', 
                aggfunc='count',
                fill_value=0
            )
            
            plt.figure(figsize=(16, 10))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='g')
            plt.title('Browsing Activity by Hour and Category', fontsize=16)
            plt.xlabel('Hour of Day (UTC)', fontsize=12)
        
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        return plt

    
    def visualize_interactive_timeline(self):
        """
        Create an interactive timeline of browsing activity using Plotly
        Optimized for larger datasets
        """
        # For large datasets, sample data to show only key points
        sample_data = self.browsing_data
        if len(self.browsing_data) > 1000:
            # Group by hour and category and count
            grouped = self.browsing_data.groupby([
                pd.Grouper(key='eventtimeutc', freq='1H'),
                'category'
            ]).size().reset_index(name='count')
            
            # Create hover text for aggregated data
            grouped['hover_text'] = (
                'Time: ' + grouped['eventtimeutc'].dt.strftime('%Y-%m-%d %H:%M') + '<br>' +
                'Category: ' + grouped['category'] + '<br>' +
                'Visit Count: ' + grouped['count'].astype(str)
            )
            
            # Create figure for aggregated data
            fig = px.scatter(
                grouped,
                x='eventtimeutc',
                y='category',
                size='count',  # Size reflects number of visits
                color='category',
                hover_name='category',
                hover_data=['hover_text'],
                title='Interactive Timeline of Browsing Activity (Aggregated by Hour)',
            )
        else:
            # For smaller datasets, show individual points
            # Ensure we have titles for all entries
            sample_data['display_title'] = sample_data['title'].fillna(
                sample_data['url'].apply(lambda x: urlparse(x).path)
            )
            
            # Create hover text
            sample_data['hover_text'] = (
                'Time: ' + sample_data['eventtimeutc'].dt.strftime('%Y-%m-%d %H:%M:%S') + '<br>' +
                'URL: ' + sample_data['url'] + '<br>' +
                'Category: ' + sample_data['category']
            )
            
            # Create figure
            fig = px.scatter(
                sample_data,
                x='eventtimeutc',
                y='category',
                color='domain',
                size=[10] * len(sample_data),  # Fixed size
                hover_name='display_title',
                hover_data=['hover_text'],
                title='Interactive Timeline of Browsing Activity',
            )
        
        fig.update_layout(
            xaxis_title='Time (UTC)',
            yaxis_title='Category',
            height=600,
            legend_title='Domain or Category'
        )
        
        return fig
    
    def visualize_sunburst(self):
        """
        Create a sunburst chart showing the hierarchy of domains and categories
        """
        # Prepare data for sunburst chart
        sunburst_data = self.browsing_data.copy()
        
        # Extract more detailed path information
        sunburst_data['path_level1'] = sunburst_data['url'].apply(
            lambda x: urlparse(x).path.split('/')[1] if urlparse(x).path and len(urlparse(x).path.split('/')) > 1 else 'root'
        )
        
        fig = px.sunburst(
            sunburst_data,
            path=['category', 'domain', 'path_level1'],
            values='time_spent',
            title='Hierarchy of Browsing Activity',
            color='category',
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        
        fig.update_layout(height=700, width=700)
        return fig
    
    def visualize_funnel(self):
        """
        Create a funnel chart showing user navigation flow
        """
        # Get sequential page views in order
        page_sequence = self.browsing_data['category'].value_counts().reset_index()
        page_sequence.columns = ['category', 'count']
        
        # Sort by count descending
        page_sequence = page_sequence.sort_values('count', ascending=False)
        
        fig = go.Figure(go.Funnel(
            y=page_sequence['category'],
            x=page_sequence['count'],
            textinfo="value+percent initial",
            marker={"color": px.colors.sequential.Viridis},
        ))
        
        fig.update_layout(
            title="User Navigation Flow",
            height=500,
        )
        
        return fig
    
    def visualize_radar_chart(self):
        """
        Create a radar chart showing activity metrics by category
        """
        category_stats = self.get_category_statistics()
        
        # Normalize metrics
        category_stats['visits_normalized'] = category_stats['visit_count'] / category_stats['visit_count'].max()
        category_stats['time_normalized'] = category_stats['time_spent_minutes'] / category_stats['time_spent_minutes'].max()
        
        categories = category_stats['category'].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=category_stats['visits_normalized'],
            theta=categories,
            fill='toself',
            name='Visit Count (Normalized)',
            line_color='indianred'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=category_stats['time_normalized'],
            theta=categories,
            fill='toself',
            name='Time Spent (Normalized)',
            line_color='cornflowerblue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Category Metrics Comparison",
            showlegend=True
        )
        
        return fig
    
    def generate_full_report(self):
        """
        Generate a comprehensive report with all visualizations
        """
        # Basic statistics
        print("Browsing History Analysis Report")
        print("=" * 40)
        print(f"Total Records: {len(self.browsing_data)}")
        print(f"Unique Domains: {self.browsing_data['domain'].nunique()}")
        print(f"Unique Categories: {self.browsing_data['category'].nunique()}")
        print(f"Date Range: {self.browsing_data['eventtimeutc'].min()} to {self.browsing_data['eventtimeutc'].max()}")
        
        # Generate all visualizations
        print("\nGenerating Visualizations...")
        self.visualize_time_series()
        plt.savefig("time_series.png")
        
        self.visualize_domains_bar()
        plt.savefig("domains_bar.png")
        
        self.visualize_categories_pie()
        plt.savefig("categories_pie.png")
        
        self.visualize_time_spent_bar()
        plt.savefig("time_spent_bar.png")
        
        self.visualize_transition_network()
        plt.savefig("transition_network.png")
        
        self.visualize_heatmap_by_time()
        plt.savefig("heatmap_by_time.png")
        
        interactive_timeline = self.visualize_interactive_timeline()
        interactive_timeline.write_html("interactive_timeline.html")
        
        sunburst = self.visualize_sunburst()
        sunburst.write_html("sunburst.html")
        
        funnel = self.visualize_funnel()
        funnel.write_html("funnel.html")
        
        radar = self.visualize_radar_chart()
        radar.write_html("radar_chart.html")
        
        print("Report generated successfully! All visualizations have been saved.")

# Convert the CSV data to a DataFrame
# csv_data = """Summary,,,,,,,,,
# OrgId,ParticipantId,DeviceId,InstalledDate,AcceptanceDate,Extension,BrowsingCount,BookmarkCount,CookieCount,
# py_demo_client,demoUser@py-insights.com,2nwjevbvxzm7ehb254,2025-02-20T23:58:02-08:00,2025-02-20T23:58:20-08:00,Chrome,5104,0,0,
# ,,,,,,,,,
# Browsing,,,,,,,,,
# OrgId,ParticipantId,DeviceId,url,eventtimeutc,transition,title,visitId,referringVisitId,eventtime
# py_demo_client,demo,2nwjevbvxzm7ehb254,chrome-extension://hkmmnfimlpcphpgnmgdecpdpaefjnlga/snapshot.html,2025-02-21T07:58:02.688Z,link,,166328,0,2025-02-20T23:58:02-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://chromewebstore.google.com/detail/snapshot/hkmmnfimlpcphpgnmgdecpdpaefjnlga?orgId=py_demo_client&product=snapshot&participantId=user@email.com&pli=1,2025-02-21T07:57:51.308Z,link,Snapshot - Chrome Web Store,166327,166326,2025-02-20T23:57:51-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://py-insights.com/account/demo/product?source=d_snapshot,2025-02-21T07:57:40.972Z,link,PY Insights | Product,166319,0,2025-02-20T23:57:40-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://py-insights.com/account/demo/product?source=d_snapshot,2025-02-21T07:57:40.988Z,link,PY Insights | Product,166321,0,2025-02-20T23:57:40-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://py-insights.com/account/demo/product,2025-02-21T07:57:38.017Z,link,PY Insights | Product,166318,0,2025-02-20T23:57:38-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://py-insights.com/account/demo/product,2025-02-21T07:57:40.975Z,link,PY Insights | Product,166320,0,2025-02-20T23:57:40-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://cs-rtl.my.salesforce-sites.com/cts/timetrade_sf1__meetingstatus?Id=CfOz60kLVHx92uS61EkgZcO4XajjlVxO4nTis64nfgU-Z&timeZone=America%2FLos_Angeles&sfdcIFrameOrigin=null,2025-02-21T07:43:10.710Z,link,Manage Meeting,166316,166317,2025-02-20T23:43:10-08:00
# py_demo_client,demo,2nwjevbvxzm7ehb254,https://py-insights.com/account/instants/myquotes,2025-02-21T07:39:12.035Z,link,PY Insights | New Quote,166315,0,2025-02-20T23:39:12-08:00"""

# Create a DataFrame from the CSV data
from io import StringIO
df = pd.read_csv('original.csv', skiprows=9)  # Skip the Summary section

# Create an instance of the analyzer with the DataFrame
analyzer = BrowsingHistoryAnalyzer(df=df)

# Run the analysis for demonstration
print("Browsing History Analysis Report")
print("=" * 40)
print(f"Total Records: {len(analyzer.browsing_data)}")
print(f"Unique Domains: {analyzer.browsing_data['domain'].nunique()}")
print(f"Unique Categories: {analyzer.browsing_data['category'].nunique()}")
print(f"Date Range: {analyzer.browsing_data['eventtimeutc'].min()} to {analyzer.browsing_data['eventtimeutc'].max()}")

# Display domain statistics
print("\nDomain Statistics:")
print(analyzer.get_domain_statistics().head())

# Display category statistics
print("\nCategory Statistics:")
print(analyzer.get_category_statistics())

# Demo visualizations
plt.figure(figsize=(18, 12))

# Plot 1: Time Series
plt.subplot(2, 2, 1)
time_series = analyzer.get_time_series_data()
plt.plot(time_series['minute'], time_series['visits'], marker='o', linestyle='-')
plt.title('Browsing Activity Over Time')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Domains bar chart
plt.subplot(2, 2, 2)
domain_stats = analyzer.get_domain_statistics()
sns.barplot(x='visit_count', y='domain', data=domain_stats)
plt.title('Domains by Number of Visits')

# Plot 3: Categories pie chart
plt.subplot(2, 2, 3)
category_stats = analyzer.get_category_statistics()
plt.pie(category_stats['visit_count'], labels=category_stats['category'], 
        autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Browsing Activity by Category')
plt.axis('equal')

# Plot 4: Time spent bar chart
plt.subplot(2, 2, 4)
sns.barplot(x='time_spent_minutes', y='category', data=category_stats)
plt.title('Time Spent by Category (Minutes)')

plt.tight_layout()

# Generate interactive visualizations for demonstration
timeline_fig = analyzer.visualize_interactive_timeline()
sunburst_fig = analyzer.visualize_sunburst()
funnel_fig = analyzer.visualize_funnel()
radar_fig = analyzer.visualize_radar_chart()

# Display a complete visualization example
plt.figure(figsize=(14, 10))
analyzer.visualize_transition_network()
plt.tight_layout()

# In a real scenario, we would save these visualizations to files
# and create a comprehensive report

# Interactive visualization example
analyzer.visualize_interactive_timeline()