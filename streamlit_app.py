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
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'POSITIVE', polarity
    elif polarity < -0.1:
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
        st.subheader("📈 Polarity Distribution")
        
        fig = px.histogram(combined_df, x='polarity', color='sentiment', nbins=50,
                          color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
        fig.update_layout(xaxis_title="Polarity Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Sentiment Score by Source")
        fig = px.box(combined_df, x='source', y='polarity', color='sentiment',
                    color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader(f"💡 Key Insights for {company_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Strengths")
            positive_keywords = extract_keywords(combined_df[combined_df['sentiment'] == 'POSITIVE']['text'], top_n=5)
            for word, count in list(positive_keywords.items())[:5]:
                st.success(f"✓ **{word.title()}**: {count} mentions")
        
        with col2:
            st.markdown("### ⚠️ Areas for Improvement")
            negative_keywords = extract_keywords(negative_df['text'], top_n=5)
            for word, count in list(negative_keywords.items())[:5]:
                st.error(f"✗ **{word.title()}**: {count} complaints")
        
        st.markdown("### 📋 Actionable Recommendations")
        
        if category_counts.iloc[0] > len(negative_df) * 0.3:
            top_issue = category_counts.index[0]
            st.warning(f"🚨 **Priority Alert**: {top_issue} accounts for {category_counts.iloc[0]/len(negative_df)*100:.1f}% of negative feedback")
            
            if 'delivery' in top_issue.lower():
                st.info("💡 **Recommendation**: Optimize delivery logistics, partner with more reliable delivery services, implement real-time tracking")
            elif 'quality' in top_issue.lower():
                st.info("💡 **Recommendation**: Enhance quality control, partner training programs, temperature-controlled packaging")
            elif 'service' in top_issue.lower():
                st.info("💡 **Recommendation**: Customer service training, faster response times, dedicated support team")
            elif 'price' in top_issue.lower():
                st.info("💡 **Recommendation**: Review pricing strategy, introduce loyalty programs, transparent pricing")
        
        st.markdown("### 📊 Performance Summary")
        
        sentiment_score = (positive_pct - negative_pct) / 100
        
        if sentiment_score > 0.3:
            st.success(f"🎉 **Excellent Performance**: {company_name} has strong positive sentiment ({positive_pct:.1f}%)")
        elif sentiment_score > 0:
            st.info(f"👍 **Good Performance**: {company_name} has more positive than negative sentiment")
        else:
            st.error(f"⚠️ **Needs Attention**: {company_name} has concerning negative sentiment levels ({negative_pct:.1f}%)")
        
        # Sample reviews
        st.markdown("### 📝 Sample Reviews")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Positive Review**")
            most_positive = combined_df.loc[combined_df['polarity'].idxmax()]
            st.success(f"_{most_positive['text'][:200]}..._")
            st.caption(f"Source: {most_positive['source']} | Polarity: {most_positive['polarity']:.3f}")
        
        with col2:
            st.markdown("**Most Negative Review**")
            most_negative = combined_df.loc[combined_df['polarity'].idxmin()]
            st.error(f"_{most_negative['text'][:200]}..._")
            st.caption(f"Source: {most_negative['source']} | Polarity: {most_negative['polarity']:.3f}")
    
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
