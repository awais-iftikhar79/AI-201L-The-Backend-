import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import csv

# ======================= CLASSES =======================


class DataHandler:
    """Handles dataset loading, cleaning, and summary"""
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.df = None

    def load_data(self):
        """Loads CSV file and automatically detects delimiter"""
        if not self.uploaded_file:
            return None

        self.uploaded_file.seek(0)
        sample = self.uploaded_file.read(2048).decode('utf-8')
        self.uploaded_file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t'])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','

        self.df = pd.read_csv(self.uploaded_file, delimiter=delimiter)
        return self.df

    def drop_missing(self):
        """Drops rows with missing values"""
        if self.df is not None:
            self.df = self.df.dropna()
        return self.df

    def drop_duplicates(self):
        if self.df is not None:
            self.df = self.df.drop_duplicates()
        return self.df

    def fill_missing(self, strategy='mean'):
        if self.df is not None:
            for col in self.df.select_dtypes(include=np.number).columns:
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self.df

    def encode_categoricals(self):
        if self.df is not None:
            self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df

    def standardize_data(self):
        if self.df is not None:
            scaler = StandardScaler()
            num_cols = self.df.select_dtypes(include=np.number).columns
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self.df

    def get_summary(self):
        if self.df is not None:
            summary = self.df.describe()
            missing = self.df.isnull().sum()
            return summary, missing
        return None, None


class Visualizer:
    """Handles all visualization tasks"""
    def __init__(self, df):
        self.df = df

    def correlation_heatmap(self):
        """Displays correlation heatmap"""
        corr_matrix = self.df.corr(numeric_only=True)
        if corr_matrix.empty:
            st.warning("❌ Correlation matrix is empty. Please ensure the dataset has numeric columns.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,)
        st.pyplot(fig)

    def scatter_plot(self, x_col, y_col):
        """Displays scatter plot between two numeric columns"""
        if x_col not in self.df.columns or y_col not in self.df.columns:
            st.warning("Selected columns not found in dataframe.")
            return

        st.subheader(f"📉 Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart({
            'data': [{
                'x': self.df[x_col],
                'y': self.df[y_col],
                'mode': 'markers',
                'type': 'scatter'
            }],
            'layout': {'title': f"{x_col} vs {y_col}"}
        })

    def histogram(self, column):
        """Plots the distribution of a selected numeric column."""
        if column not in self.df.columns:
            st.warning("Selected column not found.")
            return

        st.subheader(f"📊 Distribution of {column}")
        fig, ax = plt.subplots()
        sns.histplot(self.df[column], kde=True, bins=30, ax=ax, color="skyblue")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    def box_plot(self, numeric_col, category_col=None):
        """Plots boxplot of a numeric column, optionally by category."""
        if numeric_col not in self.df.columns:
            st.warning("Numeric column not found.")
            return

        st.subheader("📦 Box Plot")
        fig, ax = plt.subplots()

        if category_col and category_col in self.df.columns:
            sns.boxplot(x=self.df[category_col], y=self.df[numeric_col], ax=ax, palette="Set2")
            ax.set_xlabel(category_col)
            ax.set_ylabel(numeric_col)
            ax.set_title(f"{numeric_col} by {category_col}")
        else:
            sns.boxplot(y=self.df[numeric_col], ax=ax, color="lightgreen")
            ax.set_ylabel(numeric_col)
            ax.set_title(f"Distribution of {numeric_col}")

        st.pyplot(fig)


class MLModel:
    """Handles machine learning model training and evaluation"""
    def __init__(self, df, target, problem_type, model_choice):
        self.df = df
        self.target = target
        self.problem_type = problem_type
        self.model = None
        self.model_choice = model_choice

    def prepare_data(self):
        """Prepares data by encoding categorical variables automatically."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Ensure y is numeric if possible
        if y.dtype == 'object':
            try:
                y = y.astype(float)
            except ValueError:
                # Try label encoding for string targets (if classification)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def select_model(self):
        if self.problem_type == 'Regression':
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor()
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier()
            }
        return models.get(self.model_choice)

    def run(self):
        """Train and evaluate model"""
        try:
            X_train, X_test, y_train, y_test = self.prepare_data()
            self.model = self.select_model()
            if not self.model:
                st.error("Model selection invalid")
                return
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            if self.problem_type == 'Regression':
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.subheader("🧮 Regression Results")
                st.write(f"**MSE:** {mse:.2f}")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")
            else:
                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)
                st.subheader("🧠 Classification Results")
                st.write(f"**Accuracy:** {acc:.2f}")
                st.write("Confusion Matrix:")
                st.write(cm)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")


# ======================= STREAMLIT UI =======================


st.set_page_config(page_title="The Backend", layout="wide")
st.title("🚀 The Backend — Data Analysis & ML Dashboard")

st.sidebar.header("1. Upload Data File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    handler = DataHandler(uploaded_file)
    df = handler.load_data()
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Data Summary")
    summary, missing = handler.get_summary()
    st.write(summary)
    st.write("Missing Values:")
    st.write(missing)

    st.sidebar.header("2. Preprocessing")
    if st.sidebar.checkbox("Drop Missing Values"):
        df = handler.drop_missing()
        st.success("Dropped missing values.")
    if st.sidebar.checkbox("Drop Duplicates"):
        df = handler.drop_duplicates()
        st.success("Dropped duplicate rows.")
    fill_strategy = st.sidebar.selectbox("Fill Missing Strategy", ["None", "mean", "median", "mode"])
    if fill_strategy != "None":
        df = handler.fill_missing(fill_strategy)
        st.success(f"Filled missing numeric values using {fill_strategy}.")
    if st.sidebar.checkbox("Encode Categoricals"):
        df = handler.encode_categoricals()
        st.success("Applied one-hot encoding to categorical columns.")
    if st.sidebar.checkbox("Standardize Numeric Columns"):
        df = handler.standardize_data()
        st.success("Standardized numeric columns.")

    st.sidebar.header("3. Visualize Data")
    visualizer = Visualizer(df)
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("🔍 Correlation Heatmap")
        visualizer.correlation_heatmap()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        x_col = st.sidebar.selectbox("X-axis", numeric_cols)
        y_col = st.sidebar.selectbox("Y-axis", numeric_cols, index=1)
        visualizer.scatter_plot(x_col, y_col)

    if st.sidebar.checkbox("Show Feature Distribution (Histogram)"):
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            selected_col = st.sidebar.selectbox("Select column for histogram", num_cols)
            visualizer.histogram(selected_col)
        else:
            st.warning("No numeric columns available for histogram.")

        # --- Box Plot ---
    if st.sidebar.checkbox("Show Box Plot"):
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        if len(num_cols) > 0:
            numeric_col = st.sidebar.selectbox("Numeric column", num_cols, key="box_numeric")
            category_col = None
            if len(cat_cols) > 0:
                category_col = st.sidebar.selectbox("Group by (optional)", ["None"] + list(cat_cols),
                                                    key="box_category")
                if category_col == "None":
                    category_col = None
            visualizer.box_plot(numeric_col, category_col)
        else:
            st.warning("No numeric columns available for box plot.")

    st.sidebar.header("4. Apply ML Model")
    target = st.sidebar.selectbox("Select Target Column", df.columns)
    problem_type = st.sidebar.selectbox("Problem Type", ['Regression', 'Classification'])
    if problem_type == 'Regression':
        model_choice = st.sidebar.selectbox("Choose Model", [
            "Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"
        ])
    else:
        model_choice = st.sidebar.selectbox("Choose Model", [
            "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"
        ])

    if st.sidebar.button("Run Model"):
        ml = MLModel(df, target, problem_type, model_choice)
        ml.run()
else:
    st.info("📁 Upload a CSV File from the sidebar to get started")
