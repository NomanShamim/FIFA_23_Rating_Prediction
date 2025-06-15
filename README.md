---
title: FIFA 23 Rating Prediction
emoji: ⚽
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: true
---

# ⚽ FIFA 23 Player Insights & Market Value Prediction

An interactive Streamlit app to explore, visualize, and predict FIFA 23 player values using machine learning.

## 🔍 Features

- 📊 Exploratory Data Analysis (EDA)  
  Filter players by club, nationality, and position. View histograms, heatmaps, and position-wise value trends.

- 🤖 Market Value Prediction  
  Estimate a player’s market value using a trained Random Forest Regressor based on:
  Age, Overall Rating, Potential, and Wage.

- 🔎 Player Profile Lookup  
  Search any player and view detailed stats and predicted value.

- 🆚 Player Comparison  
  Compare two players side-by-side.

- 📈 Model Performance Dashboard  
  View model evaluation metrics including R², RMSE, prediction error histogram, and actual vs. predicted scatter plot.

## 🧠 Technologies Used

- Python, Pandas, NumPy  
- Streamlit  
- Scikit-learn  
- Seaborn, Matplotlib  
- Hugging Face Spaces for deployment

## 📦 Dataset

- Source: [FIFA 23 Complete Player Dataset on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
- Only male_players.csv was used for relevance and performance.

## 🚀 How to Run the App Locally

1. Clone this repository  
   ```bash
   git clone https://github.com/NomanShamim/FIFA_23_Rating_Prediction.git
   ```

2. Navigate to the folder  
   ```bash
   cd FIFA_23_Rating_Prediction
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app  
   ```bash
   streamlit run app.py
   ```

## 🧾 How to Use

- Choose a section from the sidebar:
  - Home: Overview & stats
  - EDA: Explore distributions, correlations
  - Predict Player Value: Estimate market value
  - Player Lookup: Search & compare players
  - Model Performance: Evaluate ML metrics
- Upload your own custom CSV (optional, under 200MB)
- Download filtered data for further use

## 👤 Author

- Name: Noman Shamim  
- Course: Introduction to Data Science – Term Project  
- Supervised by: [Prof. Nadeem Majeed]

---

✅ App Fully Deployed and Functional  
📦 Model: RandomForestRegressor  
🚀 Hosted on Hugging Face Spaces