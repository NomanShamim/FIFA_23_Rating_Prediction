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

This interactive Streamlit app allows users to explore the FIFA 23 player dataset and predict a player's market value using a machine learning model.

## 🔍 Features

- 📊 **Exploratory Data Analysis (EDA)**  
  Filter players by club, nationality, or position and visualize distributions, correlations, and more.

- 🤖 **Market Value Prediction**  
  Use a trained Random Forest Regressor to estimate a player's market value based on age, overall rating, potential, and wage.

- 🔎 **Player Profile Lookup**  
  Search individual players and view detailed data and predicted value.

- 📈 **Model Performance Dashboard**  
  View actual vs predicted values, R² score, RMSE, and prediction error distribution.

---

## 🧠 Technologies Used

- **Python**
- **Streamlit**
- **Pandas, Seaborn, Matplotlib**
- **Scikit-learn (RandomForestRegressor)**
- **Hugging Face Spaces for Deployment**

---

## 📦 Dataset

**Source:** [FIFA 23 Complete Player Dataset on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)

- Includes player names, ratings, positions, clubs, wages, values, and more
- Used only `male_players.csv` subset for performance and relevance

---

## 🧾 How to Use

1. Choose a section from the sidebar:
   - **Home** for overview
   - **EDA** for interactive visualizations
   - **Predict Player Value** to try predictions
   - **Player Lookup** to search a specific player
   - **Model Performance** for ML evaluation

2. Apply filters and explore insights

3. Optionally download filtered datasets

---

## 👤 Author

**Noman Shamim**  
_Introduction to Data Science – Final Project_  
Supervised by: [Pro.Nadeem Majeed]

---

## 🖼 Thumbnail

Make sure `thumbnail.png` is uploaded for a better Space preview (600x400 recommended).

---

## ✅ Status

✅ App Fully Deployed and Functional  
📦 Model: `RandomForestRegressor`  
🚀 Hosted on Hugging Face Spaces
