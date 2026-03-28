import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #E23744; font-weight: bold; text-align: center;}
    .sub-header {font-size: 1.5rem; color: #555; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# Sentiment analysis function
def analyze_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return 'NEUTRAL', 0.0
    
    text_lower = str(text).lower()
    
    # Negative patterns that TextBlob might miss
    negative_patterns = ['deleted', 'delete', 'worst', 'terrible', 'horrible', 'pathetic', 
                        'useless', 'waste', 'never', 'disappointed', 'disgusting', 'awful',
                        'poor', 'bad', 'hate', 'scam', 'fraud', 'cheat', 'rude', 'late']
    
    # Check for strong negative indicators
    if any(pattern in text_lower for pattern in negative_patterns):
        blob = TextBlob(str(text))
        polarity = min(blob.sentiment.polarity, -0.2)  # Force negative
    else:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
    
    if polarity > 0.15:
        return 'POSITIVE', polarity
    elif polarity < -0.15:
        return 'NEGATIVE', polarity
    return 'NEUTRAL', polarity

# Extract keywords
def extract_keywords(texts, top_n=15):
    keywords = ['delivery', 'price', 'quality', 'service', 'food', 'order', 'time', 'late', 
                'bad', 'good', 'excellent', 'poor', 'return', 'refund', 'issue', 'problem',
                'customer', 'support', 'app', 'restaurant', 'experience', 'taste', 'cold']
    word_freq = Counter()
    for text in texts:
        if pd.notna(text):
            words = re.findall(r'\b\w+\b', str(text).lower())
            word_freq.update([w for w in words if w in keywords and len(w) > 3])
    return dict(word_freq.most_common(top_n))

# Categorize negative reviews
def categorize_review(text):
    if pd.isna(text):
        return 'Other'
    text = str(text).lower()
    if any(word in text for word in ['delivery', 'late', 'delay', 'time', 'deliver']):
        return 'Delivery Issues'
    elif any(word in text for word in ['food', 'quality', 'taste', 'cold', 'bad', 'stale']):
        return 'Food Quality'
    elif any(word in text for word in ['service', 'support', 'customer', 'rude', 'response']):
        return 'Customer Service'
    elif any(word in text for word in ['price', 'expensive', 'cost', 'charge', 'refund']):
        return 'Pricing Issues'
    elif any(word in text for word in ['app', 'website', 'crash', 'bug', 'error', 'technical']):
        return 'Technical Issues'
    return 'Other'

# Load and process data
@st.cache_data
def load_data(file_path, source_type):
    try:
        df = pd.read_csv(file_path)
        
        # Identify text column
        text_cols = [col for col in df.columns if any(x in col.lower() for x in ['text', 'content', 'body', 'comment', 'tweet', 'title', 'description'])]
        if text_cols:
            df['text'] = df[text_cols[0]]
        elif 'text' not in df.columns:
            df['text'] = df.iloc[:, 0].astype(str)
        
        df['source'] = source_type
        df[['sentiment', 'polarity']] = df['text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
        df['category'] = df.apply(lambda row: categorize_review(row['text']) if row['sentiment'] == 'NEGATIVE' else 'N/A', axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading {source_type} data: {str(e)}")
        return None

# Main app
def main():
    st.markdown('<p class="main-header">📊 Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Social Media & News Sentiment Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    analysis_mode = st.sidebar.radio("Select Mode", ["Default Dataset (Zomato)", "Custom Dataset Upload"])
    
    if analysis_mode == "Default Dataset (Zomato)":
        st.sidebar.success("Using Zomato dataset from RealData folder")
        
        # Load default data
        twitter_df = load_data('RealData/zomato_tweets.csv', 'Twitter')
        reddit_df = load_data('RealData/reddit_posts_and_comments (1).csv', 'Reddit')
        news_df = load_data('RealData/zomato_news_last25days.csv', 'News')
        
        if twitter_df is not None and reddit_df is not None and news_df is not None:
            combined_df = pd.concat([twitter_df, reddit_df, news_df], ignore_index=True)
            company_name = "Zomato"
        else:
            st.error("Failed to load default datasets")
            return
    
    else:
        st.sidebar.info("Upload your custom datasets")
        company_name = st.sidebar.text_input("Company Name", "Your Company")
        
        twitter_file = st.sidebar.file_uploader("Upload Twitter/X Data (CSV)", type=['csv'], key='twitter')
        reddit_file = st.sidebar.file_uploader("Upload Reddit Data (CSV)", type=['csv'], key='reddit')
        news_file = st.sidebar.file_uploader("Upload News Data (CSV)", type=['csv'], key='news')
        
        dfs = []
        if twitter_file:
            df = load_data(twitter_file, 'Twitter')
            if df is not None:
                dfs.append(df)
        if reddit_file:
            df = load_data(reddit_file, 'Reddit')
            if df is not None:
                dfs.append(df)
        if news_file:
            df = load_data(news_file, 'News')
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            st.warning("Please upload at least one dataset to begin analysis")
            return
        
        combined_df = pd.concat(dfs, ignore_index=True)
    
    # Analysis
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_records = len(combined_df)
    positive_pct = (combined_df['sentiment'] == 'POSITIVE').sum() / total_records * 100
    negative_pct = (combined_df['sentiment'] == 'NEGATIVE').sum() / total_records * 100
    neutral_pct = (combined_df['sentiment'] == 'NEUTRAL').sum() / total_records * 100
    avg_polarity = combined_df['polarity'].mean()
    
    col1.metric("📝 Total Records", f"{total_records:,}")
    col2.metric("😊 Positive", f"{positive_pct:.1f}%", delta=f"{positive_pct-33:.1f}%")
    col3.metric("😞 Negative", f"{negative_pct:.1f}%", delta=f"{33-negative_pct:.1f}%", delta_color="inverse")
    col4.metric("😐 Neutral", f"{neutral_pct:.1f}%")
    col5.metric("📊 Avg Polarity", f"{avg_polarity:.3f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Deep Dive", "☁️ Word Clouds", "📈 Trends", "💬 Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = combined_df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                        color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'},
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Source-wise Distribution")
            source_counts = combined_df['source'].value_counts()
            fig = px.bar(x=source_counts.index, y=source_counts.values, 
                        color=source_counts.values, color_continuous_scale='viridis')
            fig.update_layout(xaxis_title="Source", yaxis_title="Count", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sentiment by Source")
        source_sentiment = pd.crosstab(combined_df['source'], combined_df['sentiment'], normalize='index') * 100
        fig = px.bar(source_sentiment, barmode='group', 
                    color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
        fig.update_layout(xaxis_title="Source", yaxis_title="Percentage (%)", legend_title="Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Top 3 Negative Review Categories")
        
        negative_df = combined_df[combined_df['sentiment'] == 'NEGATIVE']
        category_counts = negative_df['category'].value_counts().head(3)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(x=category_counts.values, y=category_counts.index, orientation='h',
                        color=category_counts.values, color_continuous_scale='Reds')
            fig.update_layout(xaxis_title="Count", yaxis_title="Category", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Category Breakdown")
            for idx, (cat, count) in enumerate(category_counts.items(), 1):
                pct = count / len(negative_df) * 100
                st.metric(f"#{idx} {cat}", f"{count}", f"{pct:.1f}%")
        
        st.subheader("🔑 Top Keywords in Negative Reviews")
        keywords = extract_keywords(negative_df['text'])
        if keywords:
            fig = px.bar(x=list(keywords.values()), y=list(keywords.keys()), orientation='h',
                        color=list(keywords.values()), color_continuous_scale='RdYlGn_r')
            fig.update_layout(xaxis_title="Frequency", yaxis_title="Keyword", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("☁️ Word Clouds by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment, color in zip([col1, col2, col3], 
                                         ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                                         ['Greens', 'Reds', 'Greys']):
            with col:
                st.markdown(f"**{sentiment}**")
                texts = ' '.join(combined_df[combined_df['sentiment'] == sentiment]['text'].dropna().astype(str))
                if texts.strip():
                    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                         colormap=color, max_words=50).generate(texts)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
    
    with tab4:
        st.subheader("📈 Sentiment Trends & Analysis")
        
        # Sentiment Over Time
        date_cols = [col for col in combined_df.columns if 'date' in col.lower() or 'created' in col.lower()]
        if date_cols:
            try:
                combined_df['date_parsed'] = pd.to_datetime(combined_df[date_cols[0]], errors='coerce')
                trend_df = combined_df.dropna(subset=['date_parsed']).copy()
                
                if len(trend_df) > 0:
                    trend_df['date'] = trend_df['date_parsed'].dt.date
                    
                    # Daily sentiment counts
                    daily_sentiment = trend_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
                    
                    # Only show if we have multiple dates
                    if len(daily_sentiment) > 1:
                        fig = go.Figure()
                        
                        if 'POSITIVE' in daily_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=daily_sentiment.index, 
                                y=daily_sentiment['POSITIVE'], 
                                mode='lines+markers', 
                                name='Positive', 
                                line=dict(color='#2ecc71', width=3),
                                marker=dict(size=10, symbol='circle'),
                                fill='tozeroy',
                                fillcolor='rgba(46, 204, 113, 0.1)'
                            ))
                        
                        if 'NEGATIVE' in daily_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=daily_sentiment.index, 
                                y=daily_sentiment['NEGATIVE'], 
                                mode='lines+markers', 
                                name='Negative',
                                line=dict(color='#e74c3c', width=3),
                                marker=dict(size=10, symbol='circle'),
                                fill='tozeroy',
                                fillcolor='rgba(231, 76, 60, 0.1)'
                            ))
                        
                        if 'NEUTRAL' in daily_sentiment.columns:
                            fig.add_trace(go.Scatter(
                                x=daily_sentiment.index, 
                                y=daily_sentiment['NEUTRAL'], 
                                mode='lines+markers', 
                                name='Neutral',
                                line=dict(color='#95a5a6', width=3),
                                marker=dict(size=10, symbol='circle'),
                                fill='tozeroy',
                                fillcolor='rgba(149, 165, 166, 0.1)'
                            ))
                        
                        fig.update_layout(
                            title="Sentiment Trends Over Time", 
                            xaxis_title="Date", 
                            yaxis_title="Number of Reviews",
                            hovermode='x unified', 
                            height=450,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sentiment percentage over time
                        daily_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
                        
                        fig2 = go.Figure()
                        
                        if 'POSITIVE' in daily_pct.columns:
                            fig2.add_trace(go.Bar(
                                x=daily_pct.index, 
                                y=daily_pct['POSITIVE'], 
                                name='Positive', 
                                marker_color='#2ecc71',
                                text=daily_pct['POSITIVE'].round(1),
                                texttemplate='%{text}%',
                                textposition='inside'
                            ))
                        
                        if 'NEGATIVE' in daily_pct.columns:
                            fig2.add_trace(go.Bar(
                                x=daily_pct.index, 
                                y=daily_pct['NEGATIVE'], 
                                name='Negative', 
                                marker_color='#e74c3c',
                                text=daily_pct['NEGATIVE'].round(1),
                                texttemplate='%{text}%',
                                textposition='inside'
                            ))
                        
                        if 'NEUTRAL' in daily_pct.columns:
                            fig2.add_trace(go.Bar(
                                x=daily_pct.index, 
                                y=daily_pct['NEUTRAL'], 
                                name='Neutral', 
                                marker_color='#95a5a6',
                                text=daily_pct['NEUTRAL'].round(1),
                                texttemplate='%{text}%',
                                textposition='inside'
                            ))
                        
                        fig2.update_layout(
                            title="Sentiment Distribution Over Time (%)", 
                            xaxis_title="Date", 
                            yaxis_title="Percentage",
                            barmode='stack', 
                            height=450,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("📊 Need data from multiple dates for trend analysis. Showing overall metrics.")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive", f"{positive_pct:.1f}%")
                        col2.metric("Negative", f"{negative_pct:.1f}%")
                        col3.metric("Neutral", f"{neutral_pct:.1f}%")
                else:
                    st.info("📊 No valid date information for trend analysis.")
                    
            except Exception as e:
                st.warning(f"📊 Unable to generate time-based trends: {str(e)}")
        else:
            st.info("📊 No date column found. Showing overall metrics.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{positive_pct:.1f}%")
            col2.metric("Negative", f"{negative_pct:.1f}%")
            col3.metric("Neutral", f"{neutral_pct:.1f}%")
        
        st.markdown("---")
        st.subheader("📊 Polarity Distribution")
        
        fig = px.histogram(combined_df, x='polarity', color='sentiment', nbins=50,
                          color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
        fig.update_layout(xaxis_title="Polarity Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Sentiment Score by Source")
        fig = px.box(combined_df, x='source', y='polarity', color='sentiment',
                    color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader(f"💡 Strategic Insights for {company_name}")
        
        # Key Focus Areas
        st.markdown("### 🎯 Priority Focus Areas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💪 Strengths to Leverage")
            positive_keywords = extract_keywords(combined_df[combined_df['sentiment'] == 'POSITIVE']['text'], top_n=5)
            for idx, (word, count) in enumerate(list(positive_keywords.items())[:5], 1):
                st.success(f"**{idx}. {word.title()}**: {count} positive mentions")
        
        with col2:
            st.markdown("#### ⚠️ Critical Issues to Address")
            negative_keywords = extract_keywords(negative_df['text'], top_n=5)
            for idx, (word, count) in enumerate(list(negative_keywords.items())[:5], 1):
                st.error(f"**{idx}. {word.title()}**: {count} complaints")
        
        st.markdown("---")
        
        # Actionable Recommendations
        st.markdown("### 📋 Actionable Recommendations for {company_name}")
        
        top_3_issues = category_counts.head(3)
        
        for idx, (category, count) in enumerate(top_3_issues.items(), 1):
            pct = count / len(negative_df) * 100
            
            with st.expander(f"🔴 Priority {idx}: {category} ({pct:.1f}% of complaints)", expanded=(idx==1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'delivery' in category.lower():
                        st.markdown("""
                        **Immediate Actions:**
                        - Implement real-time delivery tracking
                        - Partner with reliable delivery services
                        - Set realistic delivery time expectations
                        - Provide compensation for late deliveries
                        
                        **Long-term Strategy:**
                        - Optimize delivery route algorithms
                        - Increase delivery partner training
                        - Establish delivery quality metrics
                        """)
                    elif 'quality' in category.lower() or 'food' in category.lower():
                        st.markdown("""
                        **Immediate Actions:**
                        - Implement restaurant quality audits
                        - Temperature-controlled packaging
                        - Partner quality standards enforcement
                        - Quick complaint resolution process
                        
                        **Long-term Strategy:**
                        - Restaurant rating system improvements
                        - Quality certification program
                        - Customer feedback loop to restaurants
                        """)
                    elif 'service' in category.lower():
                        st.markdown("""
                        **Immediate Actions:**
                        - 24/7 customer support availability
                        - Reduce response time to <2 hours
                        - Empower support team with refund authority
                        - Multi-channel support (chat, phone, email)
                        
                        **Long-term Strategy:**
                        - AI-powered chatbot for instant responses
                        - Customer service training programs
                        - Proactive issue detection system
                        """)
                    elif 'price' in category.lower():
                        st.markdown("""
                        **Immediate Actions:**
                        - Transparent pricing breakdown
                        - Introduce loyalty rewards program
                        - First-time user discounts
                        - Price comparison with competitors
                        
                        **Long-term Strategy:**
                        - Dynamic pricing optimization
                        - Subscription-based models
                        - Partner with budget-friendly restaurants
                        """)
                    else:
                        st.markdown("""
                        **Immediate Actions:**
                        - Conduct detailed analysis of complaints
                        - Set up dedicated task force
                        - Implement quick-win improvements
                        
                        **Long-term Strategy:**
                        - Continuous monitoring and improvement
                        - Customer feedback integration
                        """)
                
                with col2:
                    st.metric("Complaints", count)
                    st.metric("Impact", f"{pct:.1f}%")
                    st.metric("Priority", f"P{idx}")
        
        st.markdown("---")
        
        # Personalized Response Templates
        st.markdown("### 💬 Sample Personalized Response Templates")
        
        response_templates = {
            "Delivery Issues": {
                "template": "Dear Valued Customer,\n\nWe sincerely apologize for the delay in your order delivery. We understand how frustrating this must be. We're taking immediate action to improve our delivery times and have credited ₹{amount} to your account as a goodwill gesture.\n\nYour satisfaction is our priority.\n\nBest regards,\n{company_name} Support Team",
                "icon": "🚚"
            },
            "Food Quality": {
                "template": "Dear Customer,\n\nWe're sorry to hear about your experience with food quality. This is not the standard we strive for. We've shared your feedback with the restaurant partner and issued a full refund of ₹{amount}.\n\nWe'd love to make it right - please use code QUALITY50 for 50% off your next order.\n\nThank you for helping us improve,\n{company_name} Team",
                "icon": "🍽️"
            },
            "Customer Service": {
                "template": "Dear {customer_name},\n\nThank you for bringing this to our attention. We apologize for the inconvenience caused. Our team has reviewed your case and we're implementing immediate improvements to our support process.\n\nWe've resolved your issue and added ₹{amount} credits to your account.\n\nWarm regards,\n{company_name} Customer Care",
                "icon": "💁"
            },
            "Pricing Issues": {
                "template": "Hi {customer_name},\n\nWe appreciate your feedback about pricing. Transparency is important to us. We've reviewed the charges on your order and confirmed [explanation]. As a valued customer, we're offering you a {discount}% discount on your next 3 orders.\n\nThank you for being with us,\n{company_name}",
                "icon": "💰"
            },
            "Technical Issues": {
                "template": "Hello,\n\nWe're sorry you experienced technical difficulties with our app. Our tech team has been notified and is working on a fix. Meanwhile, we've ensured your order was processed correctly and added ₹{amount} credits for the inconvenience.\n\nThank you for your patience,\n{company_name} Tech Support",
                "icon": "⚙️"
            }
        }
        
        for category, details in response_templates.items():
            with st.expander(f"{details['icon']} {category} Response Template"):
                st.code(details['template'], language=None)
                st.caption("💡 Personalize with customer name, order details, and specific compensation amounts")
        
        st.markdown("---")
        
        # Performance Summary
        st.markdown("### 📊 Overall Performance Summary")
        
        sentiment_score = positive_pct - negative_pct  # Changed from division
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sentiment_score > 30:
                st.success("🎉 **Excellent Performance**")
                st.write(f"Strong positive sentiment at {positive_pct:.1f}%")
            elif sentiment_score > 0:
                st.info("👍 **Good Performance**")
                st.write(f"More positive ({positive_pct:.1f}%) than negative")
            else:
                st.error("⚠️ **Needs Attention**")
                st.write(f"High negative sentiment at {negative_pct:.1f}%")
        
        with col2:
            st.metric("Net Sentiment Score", f"{sentiment_score:.1f}%")
            st.caption("Positive % - Negative %")
        
        with col3:
            improvement_needed = max(0, 60 - positive_pct)
            st.metric("To Reach 60% Positive", f"{improvement_needed:.1f}%")
            st.caption("Industry benchmark for food delivery")
        
        st.markdown("---")
        
        # Sample Reviews - Filter for truly positive/negative
        st.markdown("### 📝 Representative Customer Feedback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😊 Most Positive Review")
            positive_reviews = combined_df[combined_df['sentiment'] == 'POSITIVE'].copy()
            
            # Exclude false positives
            exclude_words = ['deleted', 'delete', 'worst', 'never', 'bad', 'terrible', 'horrible']
            positive_reviews = positive_reviews[
                ~positive_reviews['text'].str.lower().str.contains('|'.join(exclude_words), na=False)
            ]
            
            if len(positive_reviews) > 0:
                # Get review with highest polarity that's actually positive
                positive_reviews = positive_reviews[positive_reviews['polarity'] > 0.3]
                if len(positive_reviews) > 0:
                    most_positive = positive_reviews.nlargest(1, 'polarity').iloc[0]
                    review_text = str(most_positive['text'])[:300]
                    st.success(f"_{review_text}..._")
                    st.caption(f"📍 Source: {most_positive['source']} | Sentiment Score: {most_positive['polarity']:.3f}")
                else:
                    st.info("No strongly positive reviews found")
            else:
                st.info("No positive reviews available")
        
        with col2:
            st.markdown("#### 😞 Most Critical Review")
            negative_reviews = combined_df[combined_df['sentiment'] == 'NEGATIVE'].copy()
            if len(negative_reviews) > 0:
                most_negative = negative_reviews.nsmallest(1, 'polarity').iloc[0]
                review_text = str(most_negative['text'])[:300]
                st.error(f"_{review_text}..._")
                st.caption(f"📍 Source: {most_negative['source']} | Sentiment Score: {most_negative['polarity']:.3f}")
            else:
                st.info("No negative reviews available")
        
        st.markdown("---")
        
        # Implementation Roadmap
        st.markdown("### 🗓️ 90-Day Implementation Roadmap")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 Days 1-30: Quick Wins")
            st.markdown("""
            - ✅ Set up sentiment monitoring dashboard
            - ✅ Implement response templates
            - ✅ Train support team on new protocols
            - ✅ Launch customer feedback surveys
            - ✅ Address top 3 complaint categories
            """)
        
        with col2:
            st.markdown("#### 📅 Days 31-60: Process Improvements")
            st.markdown("""
            - 🔄 Optimize delivery logistics
            - 🔄 Enhance quality control measures
            - 🔄 Improve app performance
            - 🔄 Launch loyalty program
            - 🔄 Partner training initiatives
            """)
        
        with col3:
            st.markdown("#### 📅 Days 61-90: Long-term Strategy")
            st.markdown("""
            - 🎯 Achieve 15% reduction in negative sentiment
            - 🎯 Increase positive reviews by 20%
            - 🎯 Reduce complaint response time to <2hrs
            - 🎯 Launch predictive analytics
            - 🎯 Establish continuous improvement cycle
            """)
        
        st.success(f"🎯 **Goal**: Achieve 60%+ positive sentiment and <20% negative sentiment within 90 days")
    
    # Download results
    st.markdown("---")
    st.subheader("📥 Download Analysis Results")
    
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Complete Analysis (CSV)",
        data=csv,
        file_name=f"{company_name}_sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
