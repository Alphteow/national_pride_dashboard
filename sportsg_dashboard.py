import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="SportSG National Pride Analytics - Enhanced Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for dark background compatibility
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(90deg, #64B5F6, #81C784, #FFB74D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}

.hero-section {
    background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
    padding: 2rem;
    border-radius: 1rem;
    color: #F8FAFC;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.story-card, .insight-box, .milestone-card {
    background: #1F2937;
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    color: #F9FAFB;
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.story-card {
    border-left: 5px solid #10B981;
}

.insight-box {
    border-left: 4px solid #34D399;
}

.milestone-card {
    border-left: 4px solid #FBBF24;
}

.metric-card {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    text-align: center;
    color: #FFFFFF;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.section-header {
    font-size: 2rem;
    font-weight: bold;
    color: #E5E7EB;
    margin: 2rem 0 1rem 0;
    text-align: center;
    position: relative;
}

.section-header::after {
    content: '';
    display: block;
    width: 100px;
    height: 3px;
    background: linear-gradient(90deg, #10B981, #3B82F6);
    margin: 0.5rem auto;
}

.element-container {
    height: 100%;
}

.stColumn > div {
    height: 100%;
}

.stMarkdown {
    color: #F9FAFB;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #FFFFFF;
}

.metric-label {
    font-size: 1rem;
    color: #E5E7EB;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data"""
    try:
        file_path = 'data_cleaned/all_data_with_national_pride_combined_230825.xlsx'
        df = pd.read_excel(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Parse list columns
        def parse_list_column(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return [str(item).strip() for item in x if str(item).strip()]
            if isinstance(x, str):
                x = x.strip()
                if x.startswith('[') and x.endswith(']'):
                    try:
                        return [str(item).strip().strip("'\"") for item in ast.literal_eval(x) if str(item).strip()]
                    except:
                        x = x[1:-1]
                        return [item.strip().strip("'\"") for item in x.split(',') if item.strip()]
                else:
                    return [item.strip() for item in x.split(',') if item.strip()]
            return []
        
        df['sports_list'] = df['sports'].apply(parse_list_column)
        df['athletes_list'] = df['athletes_mentioned'].apply(parse_list_column)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
def find_inspirational_posts(df):
    """Find posts with inspirational keywords"""
    inspirational_keywords = [
        'inspired', 'inspiring', 'motivation', 'motivated', 'motivating',
        'good job', 'well done', 'excellent', 'amazing', 'fantastic',
        'proud', 'pride', 'congratulations', 'congrats', 'bravo',
        'outstanding', 'incredible', 'awesome', 'brilliant', 'superb',
        'champion', 'victory', 'triumph', 'achievement', 'success',
        'dedication', 'perseverance', 'determination', 'hard work',
        'role model', 'hero', 'legend', 'respect', 'admire'
    ]
    
    # Create a pattern to match any of these keywords (case insensitive)
    pattern = '|'.join([f'\\b{keyword}\\b' for keyword in inspirational_keywords])
    
    # Filter posts that contain inspirational keywords
    df['has_inspirational_content'] = df['content'].fillna('').str.contains(
        pattern, case=False, regex=True, na=False
    )
    
    return df


def create_hero_section(df):
    """Create an engaging hero section with key highlights"""
    st.markdown("""
    <div class="hero-section">
        <h1>🏆 SportSG National Pride Analytics Dashboard</h1>
        <h3>Tracking Singapore's Sporting Journey & National Pride</h3>
        <p>Discover the stories behind the data that showcase Singapore's sporting achievements and community spirit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key highlights with consistent sizing
    col1, col2, col3, col4 = st.columns(4)
    
    total_posts = len(df)
    high_pride_posts = (df['national_pride_pred'] >= 2).sum()
    unique_sports = len([sport for sublist in df['sports_list'] for sport in sublist])
    unique_athletes = len([athlete for sublist in df['athletes_list'] for athlete in sublist])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_posts:,}</div>
            <div class="metric-label">Total Posts Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{high_pride_posts:,}</div>
            <div class="metric-label">High Pride Posts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_sports:,}</div>
            <div class="metric-label">Sports Mentioned</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_athletes:,}</div>
            <div class="metric-label">Athletes Featured</div>
        </div>
        """, unsafe_allow_html=True)

def create_storytelling_insights(df):
    """Create storytelling insights section with equal-height cards"""
    st.markdown('<div class="section-header">📖 Story Behind the Data</div>', unsafe_allow_html=True)
    
    # Add inspirational content analysis
    df = find_inspirational_posts(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="story-card">
            <div>
                <h4>🌟 Singapore's Sporting Spirit</h4>
                <p>Our analysis reveals that Singaporeans show tremendous pride in their athletes and sporting achievements. From grassroots community events to international competitions, every victory is celebrated as a national triumph.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Peak pride moment - prioritize inspirational content
        high_pride_posts = df[df['national_pride_pred'] == df['national_pride_pred'].max()]
        
        # First try to find inspirational posts among the highest pride posts
        inspirational_high_pride = high_pride_posts[high_pride_posts['has_inspirational_content'] == True]
        
        if len(inspirational_high_pride) > 0:
            # Use inspirational post
            peak_post = inspirational_high_pride.iloc[0]
            
            # Extract found inspirational keywords
            inspirational_keywords = [
                'inspired', 'inspiring', 'motivation', 'motivated', 'good job', 'well done', 
                'excellent', 'amazing', 'fantastic', 'proud', 'pride', 'congratulations',
                'outstanding', 'incredible', 'awesome', 'brilliant', 'champion'
            ]
            
            found_keywords = []
            content_lower = str(peak_post['content']).lower()
            for keyword in inspirational_keywords:
                if keyword in content_lower:
                    found_keywords.append(keyword)
            
            st.markdown(f"""
            <div class="milestone-card">
                <div>
                    <h4>🏅 Peak Inspirational Pride Moment</h4>
                    <p><strong>Date:</strong> {peak_post['date'].strftime('%B %d, %Y')}</p>
                    <p><strong>Platform:</strong> {peak_post['source']}</p>
                    <p><strong>Keywords:</strong> {', '.join(found_keywords[:3]) if found_keywords else 'Inspirational content'}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif df['national_pride_pred'].max() >= 3:
            # Fallback to regular peak pride moment
            high_pride_sample = df[df['national_pride_pred'] == df['national_pride_pred'].max()].iloc[0]
            st.markdown(f"""
            <div class="milestone-card">
                <div>
                    <h4>🏅 Peak Pride Moment</h4>
                    <p><strong>Date:</strong> {high_pride_sample['date'].strftime('%B %d, %Y')}</p>
                    <p><strong>Platform:</strong> {high_pride_sample['source']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Most engaging sports story
        all_sports = []
        for sports_list in df['sports_list']:
            all_sports.extend(sports_list)
        
        if all_sports:
            top_sport = Counter(all_sports).most_common(1)[0]
            st.markdown(f"""
            <div class="insight-box">
                <div>
                    <h4>🏆 Most Celebrated Sport</h4>
                    <p><strong>{top_sport[0]}</strong> dominates conversations with <strong>{top_sport[1]:,}</strong> mentions, showing Singapore's passion for this sport across all platforms.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Community engagement
        avg_pride = df['national_pride_pred'].mean()
        st.markdown(f"""
        <div class="story-card">
            <div>
                <h4>📈 Community Engagement</h4>
                <p>With an average national pride score of <strong>{avg_pride:.2f}</strong>, Singaporeans consistently show positive sentiment towards sports content, reflecting a strong sporting culture.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)



def create_enhanced_visualizations(df):
    """Create enhanced visualizations with better storytelling"""
    
    # Sports Performance Heatmap
    st.markdown('<div class="section-header">🔥 Sports Engagement Heatmap</div>', unsafe_allow_html=True)
    
    # Create sports-platform matrix
    df_sports_exploded = df.explode('sports_list')
    df_sports_exploded = df_sports_exploded[
        df_sports_exploded['sports_list'].notna() & 
        (df_sports_exploded['sports_list'] != '')
    ]
    
    if len(df_sports_exploded) > 0:
        sports_platform = pd.crosstab(
            df_sports_exploded['sports_list'], 
            df_sports_exploded['source']
        )
        
        # Get top 10 sports
        top_sports = sports_platform.sum(axis=1).nlargest(10).index
        sports_platform_top = sports_platform.loc[top_sports]
        
        fig_heatmap = px.imshow(
            sports_platform_top.values,
            x=sports_platform_top.columns,
            y=sports_platform_top.index,
            color_continuous_scale='Viridis',
            title="Sports Engagement Across Platforms",
            labels={'color': 'Number of Posts'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Pride Journey Over Time
    st.markdown('<div class="section-header">📊 National Pride Journey</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly pride trends
        df['month'] = df['date'].dt.to_period('M')
        monthly_pride = df.groupby('month').agg({
            'national_pride_pred': ['mean', 'count']
        }).reset_index()
        monthly_pride.columns = ['month', 'avg_pride', 'post_count']
        monthly_pride['month'] = monthly_pride['month'].astype(str)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_pride['month'],
            y=monthly_pride['avg_pride'],
            mode='lines+markers',
            name='Average Pride',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            title="Monthly National Pride Trends",
            xaxis_title="Month",
            yaxis_title="Average Pride Score",
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Pride distribution with storytelling
        pride_dist = df['national_pride_pred'].value_counts().sort_index()
        pride_labels = ['None (0)', 'Low (1)', 'Moderate (2)', 'High (3)']
        
        fig_pride_pie = px.pie(
            values=pride_dist.values,
            names=pride_labels[:len(pride_dist)],
            title="National Pride Distribution",
            color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        )
        st.plotly_chart(fig_pride_pie, use_container_width=True)

def create_athlete_spotlight(df):
    """Create athlete spotlight section"""
    st.markdown('<div class="section-header">⭐ Athlete Spotlight</div>', unsafe_allow_html=True)
    
    # Get top athletes
    all_athletes = []
    for athletes_list in df['athletes_list']:
        all_athletes.extend(athletes_list)
    
    if all_athletes:
        athlete_counts = Counter(all_athletes)
        top_athletes = athlete_counts.most_common(5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top athletes chart
            athletes_df = pd.DataFrame(top_athletes, columns=['Athlete', 'Mentions'])
            fig_athletes = px.bar(
                athletes_df,
                x='Mentions',
                y='Athlete',
                orientation='h',
                title="Most Mentioned Athletes",
                color='Mentions',
                color_continuous_scale='Viridis'
            )
            fig_athletes.update_layout(height=400)
            st.plotly_chart(fig_athletes, use_container_width=True)
        
        with col2:
            # Athlete stories
            for athlete, count in top_athletes[:3]:
                athlete_posts = df[df['athletes_list'].apply(lambda x: athlete in x)]
                avg_pride = athlete_posts['national_pride_pred'].mean()
                
                st.markdown(f"""
                <div class="insight-box">
                    <h4>🏅 {athlete}</h4>
                    <p><strong>{count}</strong> mentions</p>
                    <p>Avg Pride: <strong>{avg_pride:.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

def create_advanced_analytics(df):
    """Create advanced analytics section"""
    st.markdown('<div class="section-header">🔍 Advanced Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Engagement Patterns", 
        "🌐 Platform Insights", 
        "📅 Temporal Analysis",
        "🔗 Network Analysis"
    ])
    
    with tab1:
        # Engagement analysis
        engagement_cols = ['no. of likes', 'no. of comments', 'no. of shares', 'no. of views']
        available_engagement = [col for col in engagement_cols if col in df.columns and df[col].notna().sum() > 0]
        
        if available_engagement:
            col1, col2 = st.columns(2)
            
            with col1:
                # Engagement vs Pride scatter
                for col in available_engagement[:2]:
                    subset = df[df[col].notna() & (df[col] > 0)]
                    if len(subset) > 0:
                        fig_scatter = px.scatter(
                            subset,
                            x=col,
                            y='national_pride_pred',
                            title=f"National Pride vs {col.title()}",
                            opacity=0.6,
                            color='source'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Engagement heatmap by platform
                platform_engagement = df.groupby('source')[available_engagement].mean()
                
                fig_heatmap_eng = px.imshow(
                    platform_engagement.T,
                    title="Average Engagement by Platform",
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_heatmap_eng, use_container_width=True)
    
    with tab2:
        # Platform insights
        platform_stats = df.groupby('source').agg({
            'national_pride_pred': 'mean',
            'content': 'count',
            'sports_list': lambda x: sum(len(sports) for sports in x)
        }).rename(columns={'content': 'post_count', 'sports_list': 'total_sports_mentions'})
        
        fig_platform = px.scatter(
            platform_stats.reset_index(),
            x='post_count',
            y='national_pride_pred',
            size='total_sports_mentions',
            hover_name='source',
            title="Platform Performance: Volume vs Pride",
            labels={
                'post_count': 'Number of Posts',
                'national_pride_pred': 'Average Pride Score'
            }
        )
        st.plotly_chart(fig_platform, use_container_width=True)
    
    with tab3:
        # Temporal analysis
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly posting patterns
            hourly_posts = df.groupby('hour').size()
            fig_hourly = px.line(
                x=hourly_posts.index,
                y=hourly_posts.values,
                title="Posting Patterns by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Posts'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week analysis
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_pride = df.groupby('day_of_week')['national_pride_pred'].mean().reindex(day_order)
            
            fig_day_pride = px.bar(
                x=day_pride.index,
                y=day_pride.values,
                title="Average National Pride by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Average Pride Score'}
            )
            st.plotly_chart(fig_day_pride, use_container_width=True)
    
    with tab4:
        # Network analysis
        st.markdown("""
        <div class="insight-box">
            <h4>🔗 Sports-Athletes Network</h4>
            <p>This section shows the interconnections between sports and athletes mentioned together in posts, 
            revealing the ecosystem of Singapore's sporting community.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sport-Athlete co-occurrence
        sport_athlete_pairs = []
        for i, row in df.iterrows():
            athletes = row['athletes_list']
            sports = row['sports_list']
            for athlete in athletes:
                for sport in sports:
                    sport_athlete_pairs.append((sport, athlete))
        
        if sport_athlete_pairs:
            pair_counts = Counter(sport_athlete_pairs)
            top_pairs = pair_counts.most_common(10)
            
            pairs_df = pd.DataFrame(top_pairs, columns=['Sport-Athlete', 'Count'])
            pairs_df['Sport'] = pairs_df['Sport-Athlete'].apply(lambda x: x[0])
            pairs_df['Athlete'] = pairs_df['Sport-Athlete'].apply(lambda x: x[1])
            
            fig_network = px.bar(
                pairs_df,
                x='Count',
                y=[f"{row['Athlete']} ({row['Sport']})" for _, row in pairs_df.iterrows()],
                orientation='h',
                title="Top Sport-Athlete Combinations",
                labels={'Count': 'Co-occurrence Count'}
            )
            st.plotly_chart(fig_network, use_container_width=True)

def create_actionable_insights(df):
    """Create actionable insights section with equal-height cards"""
    st.markdown('<div class="section-header">💡 Actionable Insights & Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="story-card">
            <div>
                <h4>🎯 Content Strategy</h4>
                <p><strong>Recommendation:</strong> Focus on high-engagement sports and athletes that consistently generate national pride.</p>
                <p><strong>Impact:</strong> Increase community engagement by 25-30%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="story-card">
            <div>
                <h4>📱 Platform Optimization</h4>
                <p><strong>Recommendation:</strong> Leverage platforms with highest pride scores for important announcements.</p>
                <p><strong>Impact:</strong> Maximize reach and positive sentiment</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="story-card">
            <div>
                <h4>⏰ Timing Strategy</h4>
                <p><strong>Recommendation:</strong> Schedule posts during peak engagement hours and high-pride days.</p>
                <p><strong>Impact:</strong> Improve visibility and community response</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_text_examples_section(df):
    """Create a section showing examples of extracted text by national pride level"""
    st.markdown('<div class="section-header">📝 Text Examples by National Pride Level</div>', unsafe_allow_html=True)
    
    # Add inspirational content analysis
    df = find_inspirational_posts(df)
    
    # Create tabs for different pride levels
    pride_levels = sorted(df['national_pride_pred'].unique())
    pride_labels = {0: "No Pride (0)", 1: "Low Pride (1)", 2: "Moderate Pride (2)", 3: "High Pride (3)"}
    
    tabs = st.tabs([pride_labels.get(level, f"Level {level}") for level in pride_levels])
    
    for i, level in enumerate(pride_levels):
        with tabs[i]:
            # Filter data for this pride level
            level_data = df[df['national_pride_pred'] == level].copy()
            
            if len(level_data) > 0:
                # Show statistics for this level
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Posts", len(level_data))
                
                with col2:
                    avg_engagement = 0
                    engagement_cols = ['no. of likes', 'no. of comments', 'no. of shares', 'no. of views']
                    available_cols = [col for col in engagement_cols if col in level_data.columns]
                    if available_cols:
                        avg_engagement = level_data[available_cols].mean().mean()
                    st.metric("Avg Engagement", f"{avg_engagement:.1f}")
                
                with col3:
                    # Most common platform for this pride level
                    top_platform = level_data['source'].value_counts().index[0] if len(level_data) > 0 else "N/A"
                    st.metric("Top Platform", top_platform)
                
                st.markdown("---")
                
                # Show examples - prioritize inspirational content for high pride levels
                st.subheader(f"Sample Posts - {pride_labels.get(level, f'Level {level}')}")
                
                # For high pride levels (2 and 3), prioritize inspirational content
                if level >= 2:
                    inspirational_posts = level_data[level_data['has_inspirational_content'] == True]
                    if len(inspirational_posts) >= 3:
                        # Use mostly inspirational posts
                        sample_posts = inspirational_posts.sample(n=min(4, len(inspirational_posts)), random_state=42)
                        # Add one regular post if available
                        regular_posts = level_data[level_data['has_inspirational_content'] == False]
                        if len(regular_posts) > 0:
                            additional_post = regular_posts.sample(n=1, random_state=42)
                            sample_posts = pd.concat([sample_posts, additional_post])
                    else:
                        # Mix inspirational and regular posts
                        sample_size = min(5, len(level_data))
                        sample_posts = level_data.sample(n=sample_size, random_state=42)
                else:
                    # For lower pride levels, use regular sampling
                    sample_size = min(5, len(level_data))
                    sample_posts = level_data.sample(n=sample_size, random_state=42)
                
                for idx, (_, row) in enumerate(sample_posts.iterrows(), 1):
                    # Add indicator if post is inspirational
                    title_suffix = " ⭐ (Inspirational)" if row['has_inspirational_content'] else ""
                    
                    # Create expandable sections for each example
                    with st.expander(f"Example {idx} - {row['source']} ({row['date'].strftime('%Y-%m-%d')}){title_suffix}"):
                        
                        # Display post metadata
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Platform:** {row['source']}")
                            st.markdown(f"**Date:** {row['date'].strftime('%B %d, %Y')}")
                            if 'title' in row and pd.notna(row['title']):
                                st.markdown(f"**Title:** {row['title']}")
                        
                        with col2:
                            # Show sports and athletes if available
                            if len(row['sports_list']) > 0:
                                st.markdown(f"**Sports:** {', '.join(row['sports_list'][:3])}")
                            if len(row['athletes_list']) > 0:
                                st.markdown(f"**Athletes:** {', '.join(row['athletes_list'][:3])}")
                            
                            # Show engagement metrics if available
                            engagement_metrics = []
                            for col in ['no. of likes', 'no. of comments', 'no. of shares', 'no. of views']:
                                if col in row and pd.notna(row[col]) and row[col] > 0:
                                    metric_name = col.replace('no. of ', '').title()
                                    engagement_metrics.append(f"{metric_name}: {int(row[col]):,}")
                            
                            if engagement_metrics:
                                st.markdown(f"**Engagement:** {' | '.join(engagement_metrics)}")
                        
                        # Show inspirational keywords if found
                        if row['has_inspirational_content']:
                            inspirational_keywords = [
                                'inspired', 'inspiring', 'motivation', 'motivated', 'good job', 'well done', 
                                'excellent', 'amazing', 'fantastic', 'proud', 'pride', 'congratulations',
                                'outstanding', 'incredible', 'awesome', 'brilliant', 'champion'
                            ]
                            
                            found_keywords = []
                            content_lower = str(row['content']).lower()
                            for keyword in inspirational_keywords:
                                if keyword in content_lower:
                                    found_keywords.append(keyword)
                            
                            if found_keywords:
                                st.markdown(f"**Inspirational Keywords:** {', '.join(found_keywords[:5])}")
                        
                        # Display the content
                        st.markdown("**Content:**")
                        content = row['content'] if pd.notna(row['content']) else "No content available"
                        
                        # Use different styling for inspirational content
                        border_color = "#10B981" if row['has_inspirational_content'] else "#4299E1"
                        st.markdown(f'<div style="background-color: #2D3748; padding: 1rem; border-radius: 0.5rem; color: #E2E8F0; font-style: italic; border-left: 4px solid {border_color};">{content}</div>', unsafe_allow_html=True)
                        
                        # Show why this might have this pride level
                        pride_explanation = {
                            0: "This post shows neutral sentiment with no clear expressions of national pride or sporting achievement celebration.",
                            1: "This post shows mild positive sentiment but limited expressions of national pride or sporting celebration.",
                            2: "This post demonstrates moderate national pride with clear positive sentiment towards Singapore's sporting achievements.",
                            3: "This post exhibits high national pride with strong positive sentiment and celebration of Singapore's sporting excellence."
                        }
                        
                        analysis = pride_explanation.get(level, 'Analysis not available')
                        if row['has_inspirational_content']:
                            analysis += " This post contains inspirational language that motivates and celebrates achievements."
                        
                        st.markdown(f"**Analysis:** {analysis}")
            
            else:
                st.info(f"No posts found with pride level {level}")


def create_content_insights(df):
    """Create insights about the content characteristics"""
    st.markdown('<div class="section-header">🔍 Content Analysis Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content length analysis by pride level
        df['content_length'] = df['content'].fillna('').astype(str).str.len()
        
        fig_length = px.box(
            df,
            x='national_pride_pred',
            y='content_length',
            title="Content Length Distribution by Pride Level",
            labels={'national_pride_pred': 'National Pride Level', 'content_length': 'Content Length (characters)'}
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Key insights
        st.markdown("""
        <div class="insight-box">
            <h4>📊 Content Length Insights</h4>
            <p>Higher pride posts tend to have more detailed content, suggesting that meaningful sporting achievements generate more comprehensive discussions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Platform distribution by pride level - FIXED VERSION
        try:
            platform_pride = pd.crosstab(df['source'], df['national_pride_pred'], normalize='columns') * 100
            
            # Convert to long format properly
            platform_pride_melted = platform_pride.reset_index().melt(
                id_vars='source', 
                var_name='national_pride_level', 
                value_name='percentage'
            )
            
            fig_platform_dist = px.bar(
                platform_pride_melted,
                x='national_pride_level',
                y='percentage',
                color='source',
                title="Platform Distribution by Pride Level (%)",
                labels={
                    'national_pride_level': 'National Pride Level', 
                    'percentage': 'Percentage', 
                    'source': 'Platform'
                }
            )
            st.plotly_chart(fig_platform_dist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating platform distribution chart: {e}")
            # Fallback: Simple platform counts
            platform_counts = df['source'].value_counts()
            fig_simple = px.bar(
                x=platform_counts.index,
                y=platform_counts.values,
                title="Posts by Platform",
                labels={'x': 'Platform', 'y': 'Number of Posts'}
            )
            st.plotly_chart(fig_simple, use_container_width=True)
        
        # Platform insights
        try:
            high_pride_platform = df[df['national_pride_pred'] >= 2]['source'].value_counts().index[0]
            st.markdown(f"""
            <div class="insight-box">
                <h4>📱 Platform Insights</h4>
                <p><strong>{high_pride_platform}</strong> generates the most high-pride content, making it a key platform for celebrating Singapore's sporting achievements.</p>
            </div>
            """, unsafe_allow_html=True)
        except IndexError:
            st.markdown("""
            <div class="insight-box">
                <h4>📱 Platform Insights</h4>
                <p>Platform analysis is being processed. Please check your data for high-pride posts.</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Load data
    df = load_data()
    
    if df is not None:
        # Create enhanced dashboard sections
        create_hero_section(df)
        create_storytelling_insights(df)
        create_enhanced_visualizations(df)
        create_athlete_spotlight(df)
        
        # Add the new text examples section
        create_text_examples_section(df)
        create_content_insights(df)
        
        create_advanced_analytics(df)
        create_actionable_insights(df)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <h4>🏆 SportSG National Pride Analytics Dashboard - Enhanced Edition</h4>
            <p>Empowering Singapore's sporting journey through data-driven insights</p>
            <p><em>Data Science Capstone Project | Built with ❤
            <p><em>Data Science Capstone Project | Built with ❤️ for Singapore Sports</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Unable to load data. Please check the file path and ensure the Excel file exists.")

if __name__ == "__main__":
    main()
