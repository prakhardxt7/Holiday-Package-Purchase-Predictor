import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Travel_cleaned.csv")

# Step 1: Welcome Page
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    st.markdown(
        """
        <div style="text-align: center; padding: 50px;">
            <h1 style="color: #4CAF50; font-size: 3em;">Welcome to Holiday Bliss Predictor</h1>
            <p style="font-size: 1.5em; color: #555;">"Your Journey Starts with Insights"</p>
            <img src="https://via.placeholder.com/600x200.png?text=Holiday+Packages" alt="Holiday Packages" style="border-radius: 10px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Proceed to Explore", key="proceed_explore"):
        st.session_state.page = "options"

# Step 2: Options Page
elif st.session_state.page == "options":
    st.title("Choose Your Action")
    action = st.radio("What would you like to do?", ("Data Quality Check", "Model Results", "Make a Prediction"), key="action_selection")

    if st.button("Next", key="next_action"):
        st.session_state.page = action.lower().replace(" ", "_")

# Step 3: Data Quality Check
elif st.session_state.page == "data_quality_check":
    st.title("Data Quality Check")

    data = load_data()
    st.write("### Dataset Preview", data.head())

    st.write("**Null Values in Dataset**")
    st.write(data.isnull().sum())

    st.write("**Duplicate Values in Dataset**")
    st.write(data.duplicated().sum())

    st.write("**Unique Values in Each Column**")
    st.write(data.nunique())

    if st.checkbox("Show Correlation Heatmap"):
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    if st.button("Back", key="back_data_quality"):
        st.session_state.page = "options"

# Step 4: Model Results
elif st.session_state.page == "model_results":
    st.title("Model Results")

    data = load_data()

   

    # Now we drop the CustomerID column


    X = data.drop(columns=['ProdTaken'])
    y = data['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include='object').columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_features=7, min_samples_split=2, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### Model Performance")
    st.write("**Accuracy**:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    if st.checkbox("Show Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
        st.pyplot(plt)

    if st.button("Back", key="back_model_results"):
        st.session_state.page = "options"

# Step 5: Prediction Page
elif st.session_state.page == "make_a_prediction":
    st.title("Make a Prediction")

    data = load_data()


    


    # Now we drop the CustomerID column
    

    X = data.drop(columns=['ProdTaken'])
    y = data['ProdTaken']

    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include='object').columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)]
    )

    model = RandomForestClassifier(n_estimators=200, max_features=7, min_samples_split=2, max_depth=None, random_state=42)
    X_processed = preprocessor.fit_transform(X)
    model.fit(X_processed, y)

    st.write("### Input Parameters")

    user_input = {}
    for col in X.columns:
        if col in categorical_features:
            user_input[col] = st.selectbox(f"Select {col}", options=data[col].unique(), help=f"Select the value(s) for {col}")
        else:
            user_input[col] = st.number_input(f"Enter {col}", min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].mean()), step=0.01, help=f"Adjust the value for {col}")

    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <button style="font-size: 1.5em; background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px;" onclick="window.location.reload()">Get Prediction</button>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Get Prediction", key="get_prediction"):
        user_df = pd.DataFrame([user_input])
        user_transformed = preprocessor.transform(user_df)
        prediction = model.predict(user_transformed)
        result = "🎉 Yay! This customer is likely to purchase the package!" if prediction[0] == 1 else "😞 Unfortunately, this customer may not purchase the package."
        st.markdown(f"<div style='text-align: center; font-size: 1.5em; color: #4CAF50;'>{result}</div>", unsafe_allow_html=True)

    if st.button("Back", key="back_prediction"):
        st.session_state.page = "options"
