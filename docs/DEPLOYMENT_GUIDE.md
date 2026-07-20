# Streamlit Dashboard Deployment Guide

Hosting your dashboard online is the best way to share your machine learning model with reviewers, colleagues, and the public. We highly recommend using **Streamlit Community Cloud**, as it is 100% free, specifically optimized for Python data apps, and takes less than 5 minutes to set up.

## Prerequisites
Before you begin, ensure you have:
1. Pushed this entire project folder to a public or private repository on **GitHub**.
2. A free account on [Streamlit Community Cloud](https://share.streamlit.io/) (you can sign in with your GitHub account).

---

## Deployment Steps

### Step 1: Connect your GitHub
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New app"** (in the top right corner).
3. If this is your first time, authorize Streamlit to access your GitHub repositories.

### Step 2: Configure the App
You will see a form asking for three pieces of information. Fill them out exactly as follows:
- **Repository:** Select your project repository from the dropdown (e.g., `your-username/AgriGuide`).
- **Branch:** `main` (or whichever branch your code is on).
- **Main file path:** `dashboard.py`

### Step 3: Deploy!
Click the blue **"Deploy!"** button. 

Streamlit will automatically read the `requirements.txt` file we optimized, provision a server, install all the necessary dependencies (like XGBoost, Plotly, and pandas), and launch your application.

> [!TIP]
> The initial deployment will take about **2-3 minutes** while it installs the packages. After the first deployment, any updates you push to your GitHub repository will automatically and instantly update the live web dashboard!

## Troubleshooting
If you encounter any memory errors (e.g., the app suddenly crashing), it means the 1GB RAM limit was exceeded. We have already mitigated this by stripping massive unused libraries (like PyTorch) from `requirements.txt`. If you add new heavy libraries in the future, be mindful of this limit.
