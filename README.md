# 📊 Sentiment Analysis Dashboard - Deployment Package

## 🚀 Quick Start

### Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud
1. Push this folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Main file: `streamlit_app.py`
6. Deploy!

## 📁 Folder Structure
```
deployment/
├── streamlit_app.py          # Main dashboard
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── RealData/                 # Datasets
│   ├── zomato_tweets.csv
│   ├── reddit_posts_and_comments (1).csv
│   └── zomato_news_last25days.csv
└── .streamlit/
    └── config.toml           # Theme configuration
```

## ✨ Features
- Real-time sentiment analysis
- Multi-source data (Twitter, Reddit, News)
- Top 3 negative review categories
- Interactive visualizations
- Custom CSV upload
- Download results

## 🎯 For Presentation
1. Run the dashboard
2. Show default Zomato analysis
3. Navigate through all 5 tabs
4. Demo custom upload feature
5. Highlight actionable insights

**Ready to deploy! 🚀**
