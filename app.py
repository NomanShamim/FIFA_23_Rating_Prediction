import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------- Page Configuration -----------------
st.set_page_config(page_title="FIFA 23 Player Insights", page_icon="⚽", layout="wide")

# ----------------- Custom CSS Enhancements -----------------
st.markdown("""
    <style>
    html { zoom: 95%; }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    thead tr th { font-size: 14px !important; }
    tbody tr td { font-size: 13px; }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Theme Toggle -----------------
theme = st.sidebar.selectbox("🌃 Theme", ["light", "dark"], key="theme_selector")
sns.set_theme(style="darkgrid" if theme == "dark" else "whitegrid")

# ----------------- Load Default Data Function -----------------
@st.cache_data
def load_default_data():
    return pd.read_csv("male_players.csv", encoding="utf-8", on_bad_lines="skip", engine="python")

@st.cache_data
def get_club_list(df): return sorted(df["club_name"].dropna().unique())

@st.cache_data
def get_player_list(df): return sorted(df["short_name"].dropna().unique())

# ----------------- Sidebar CSV Upload Option -----------------
st.sidebar.markdown("### 📁 Upload Custom Dataset (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Max 200MB)", type=["csv"])

if uploaded_file:
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
        st.sidebar.error("🚫 File too large! Please upload a file smaller than 200 MB.")
        df = load_default_data()
        st.sidebar.warning("Using default dataset.")
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip", engine="python")
            st.sidebar.success("✅ Custom dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load file: {e}")
            df = load_default_data()
else:
    df = load_default_data()

# ----------------- Reprocess Data -----------------
df["main_position"] = df["player_positions"].apply(lambda x: x.split()[0] if isinstance(x, str) else x)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ----------------- Sidebar Filters -----------------
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio("Go to", ["Home", "EDA", "Predict Player Value", "Player Lookup", "Model Performance"])

st.sidebar.markdown("### 🎛 Filters")
selected_club = st.sidebar.selectbox("🏟️ Club", ["All"] + get_club_list(df))
selected_nation = st.sidebar.selectbox("🌍 Country", ["All"] + sorted(df["nationality_name"].dropna().unique()))
selected_pos = st.sidebar.selectbox("🎯 Position", ["All"] + sorted(df["main_position"].dropna().unique()))

filtered_df = df.copy()
if selected_club != "All":
    filtered_df = filtered_df[filtered_df["club_name"] == selected_club]
if selected_nation != "All":
    filtered_df = filtered_df[filtered_df["nationality_name"] == selected_nation]
if selected_pos != "All":
    filtered_df = filtered_df[filtered_df["main_position"] == selected_pos]

# ------------------- HOME -------------------
if menu == "Home":
    # Sticky top bar and animations
    st.markdown("""
        <style>
        .css-1r6slb0.e1tzin5v2 {
            position: sticky;
            top: 0;
            background: white;
            z-index: 999;
            border-bottom: 1px solid #f0f0f0;
        }
        .fade-in-section {
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease-out forwards;
        }
        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero banner
    st.markdown("""
    <div class="fade-in-section" style='background: linear-gradient(to right, #1d2671, #c33764); padding: 2rem 1rem; border-radius: 12px; margin-bottom: 20px;'>
        <h1 style='text-align: center; color: white; font-size: 3rem;'>⚽ FIFA 23 Player Insights</h1>
        <p style='text-align: center; color: white; font-size: 1.2rem;'>Unlock the power of data to predict, explore & compare football players.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("🧍 Players", f"{len(df)}")
    col2.metric("🏟️ Clubs", f"{df['club_name'].nunique()}")
    col3.metric("🌍 Countries", f"{df['nationality_name'].nunique()}")

    # Features
    st.markdown("""
    <div class="fade-in-section">
        <h3>🚀 Features</h3>
        <div style="display: flex; justify-content: space-between;">
            <div style="flex: 1; padding: 10px;"><strong>🔍 Interactive Exploration</strong><br>Apply filters & view player insights</div>
            <div style="flex: 1; padding: 10px;"><strong>🤖 ML Predictions</strong><br>Estimate player's value using a trained ML model</div>
            <div style="flex: 1; padding: 10px;"><strong>🆚 Compare Players</strong><br>Search and analyze profiles side-by-side</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # About Project
    with st.expander("📘 About This Project"):
        st.markdown("""
        - **Author:** Noman Shamim  
        - **Built With:** Streamlit · Scikit-learn · Pandas · Seaborn  
        - **Dataset:** [FIFA 23 Player Dataset (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)  
        This interactive web app demonstrates how machine learning and data visualization can enhance football scouting and analysis. Designed for football fans, analysts, and learners alike.
        """)

    # 🔐 Admin Panel
    with st.expander("🔐 Admin Access"):
        st.caption("🛡️ Admin-only section to monitor logs or updates")
        password = st.text_input("Enter Admin Password", type="password")
        if password == "fifaadmin2025":
            st.success("✅ Access Granted")
            st.markdown("You can now add admin tools or view logs.")
        elif password:
            st.error("❌ Incorrect Password")

    # Sidebar contact
    with st.sidebar.expander("💬 Feedback or Help"):
        st.markdown("📩 Email: **nomanshamim720@gmail.com**")
        st.markdown("🌐 Hugging Face: [Your Space](https://huggingface.co/spaces/NomanShamim/FIFA_23_Rating_Prediction)")

    # Back to Top Button
    st.markdown("""
        <style>
        #topButton {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 999;
            background-color: #1d2671;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        section.main > div.block-container {
            padding-bottom: 0rem;
        }
        </style>
        <script>
        const scrollButton = document.createElement("button");
        scrollButton.id = "topButton";
        scrollButton.innerHTML = "⬆️ Top";
        scrollButton.onclick = function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };
        document.body.appendChild(scrollButton);
        </script>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <hr style="margin-top: 1rem; margin-bottom: 0.2rem;">
        <div style='text-align:center; font-size:0.9rem; color:gray; font-weight:600;'>
            © 2025 · Built with ❤️ by Noman Shamim
        </div>
    """, unsafe_allow_html=True)


# ------------------- EDA -------------------
elif menu == "EDA":
    st.markdown("<h2>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.caption(f"🎯 {len(filtered_df)} players match current filters")

    col1, col2 = st.columns(2)
    col1.metric("Avg Overall", f"{filtered_df['overall'].mean():.1f}")
    col2.metric("Avg Value (€)", f"{filtered_df['value_eur'].mean():,.0f}")

    st.download_button("⬇️ Download Filtered CSV",
                       data=filtered_df.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_players.csv", mime="text/csv")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📌 Positions & Value", "📈 Correlation"])

    with tab1:
        st.subheader("Age & Rating Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(filtered_df["age"], kde=True, ax=ax)
            ax.set_title("Player Age Distribution")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(filtered_df["overall"], kde=True, ax=ax)
            ax.set_title("Overall Rating Distribution")
            st.pyplot(fig)

    with tab2:
        st.subheader("Top Positions & Clubs")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.barplot(y=filtered_df["main_position"].value_counts().head(10).index,
                        x=filtered_df["main_position"].value_counts().head(10).values,
                        palette="Set2", ax=ax)
            ax.set_title("Top 10 Primary Positions")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.barplot(y=filtered_df["club_name"].value_counts().head(10).index,
                        x=filtered_df["club_name"].value_counts().head(10).values,
                        palette="cool", ax=ax)
            ax.set_title("Top Clubs by Player Count")
            st.pyplot(fig)

        st.subheader("💸 Average Market Value by Position")
        avg_val = filtered_df.groupby("main_position")["value_eur"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(avg_val)

    with tab3:
        st.subheader("📉 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(filtered_df[["age", "overall", "potential", "wage_eur", "value_eur"]].corr(),
                    annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("💵 Wage vs Market Value")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df, x="wage_eur", y="value_eur", hue="main_position", alpha=0.6, ax=ax)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        ax.set_title("Wage vs Value")
        st.pyplot(fig)

    with st.expander("📋 Show Raw Filtered Data"):
        st.dataframe(filtered_df, use_container_width=True)

# ------------------- PREDICT PLAYER VALUE -------------------
elif menu == "Predict Player Value":
    st.markdown("<h2>🧠 Predict Player Market Value</h2>", unsafe_allow_html=True)
    st.markdown("Use the sliders and input to estimate a player's market value using our trained ML model.")

    with st.form("predict_form"):
        age = st.slider("Age", 16, 45, 25)
        overall = st.slider("Overall Rating", 40, 95, 75)
        potential = st.slider("Potential", 40, 95, 80)
        wage = st.number_input("Wage (EUR)", min_value=1000, max_value=500000, value=15000)
        st.caption("💸 Wage in Euros (weekly)")

        if wage > 1_000_000:
            st.warning("⚠️ Wage entered seems unusually high. Please check.")

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([[age, overall, potential, wage]], columns=feature_columns)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        st.success(f"💰 Estimated Market Value: €{int(pred):,}")

        if pred > 10_000_000:
            st.balloons()
            st.info("🎉 Superstar-level player detected!")

        # Save prediction
        log_df = pd.DataFrame([[age, overall, potential, wage, pred]],
                              columns=feature_columns + ["predicted_value"])
        log_df.to_csv("user_predictions.csv", mode='a', header=False, index=False)

        # Visual + Metric
        st.subheader("📊 Predicted Value Overview")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.barplot(x=["Market Value"], y=[pred], palette="viridis", ax=ax)
            ax.set_ylabel("€")
            ax.set_title("Predicted Market Value")
            st.pyplot(fig)
        with col2:
            st.metric(label="Predicted Value", value=f"€{int(pred):,}")
            st.caption("Based on Age, Rating, Potential, Wage")
# ------------------- PLAYER LOOKUP -------------------
elif menu == "Player Lookup":
    st.markdown("<h2>🔍 Player Profile Search</h2>", unsafe_allow_html=True)

    player = st.selectbox("Select a Player", get_player_list(df), key="lookup_select")
    player_row = df[df["short_name"] == player]

    if not player_row.empty:
        st.markdown(f"### 🎯 Player: **{player}**")
        st.dataframe(player_row.reset_index(drop=True), use_container_width=True)

        try:
            X = player_row[feature_columns]
            X_scaled = scaler.transform(X)
            pred_val = model.predict(X_scaled)[0]
            st.success(f"💰 Predicted Market Value: €{int(pred_val):,}")
        except:
            st.warning("⚠️ Prediction failed — check for missing values.")

        st.subheader("📊 Player Rating Map")
        fig, ax = plt.subplots()
        sns.scatterplot(data=player_row, x="potential", y="overall", s=200, ax=ax, color="royalblue")
        ax.set_xlim(40, 100)
        ax.set_ylim(40, 100)
        ax.set_title("Potential vs. Overall")
        st.pyplot(fig)

    # Player Comparison
    st.markdown("---")
    st.subheader("🆚 Compare Players")
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", get_player_list(df), key="cmp1")
    with col2:
        player2 = st.selectbox("Player 2", get_player_list(df), key="cmp2")

    if player1 and player2 and player1 != player2:
        df1 = df[df["short_name"] == player1]
        df2 = df[df["short_name"] == player2]
        combined = pd.concat([df1, df2])
        st.dataframe(combined.reset_index(drop=True), use_container_width=True)

        st.download_button("📥 Download Comparison CSV",
                           data=combined.to_csv(index=False).encode(),
                           file_name="player_comparison.csv")

# ------------------- MODEL PERFORMANCE -------------------
elif menu == "Model Performance":
    st.markdown("<h2>📈 Model Evaluation</h2>", unsafe_allow_html=True)

    try:
        log = pd.read_csv("model_results.csv")

        st.subheader("📋 Prediction Logs")
        with st.expander("🔍 View Sample Predictions"):
            st.dataframe(log.head(20), use_container_width=True)

        st.subheader("📏 Model Metrics")
        r2 = log.corr().loc["actual", "predicted"]
        rmse = ((log["actual"] - log["predicted"])**2).mean()**0.5
        mae = log["actual"].sub(log["predicted"]).abs().mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("Mean Abs Error", f"€{mae:,.0f}")

        st.subheader("📊 Prediction Error Distribution")
        fig, ax = plt.subplots()
        error = log["predicted"] - log["actual"]
        sns.histplot(error, bins=30, kde=True, ax=ax, color="gray")
        ax.set_title("Prediction Error Histogram")
        st.pyplot(fig)

        st.subheader("📌 Actual vs Predicted")
        fig, ax = plt.subplots()
        sns.scatterplot(x=log["actual"], y=log["predicted"], alpha=0.6, ax=ax)
        ax.set_title("Actual vs Predicted Market Value")
        ax.set(xlabel="Actual Value (€)", ylabel="Predicted Value (€)")
        st.pyplot(fig)

    except Exception:
        st.warning("⚠️ model_results.csv not found or cannot be loaded.")
