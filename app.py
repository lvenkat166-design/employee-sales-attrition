import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Workforce Retail Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Workforce Retail Analytics")

st.write(
    "Employee Attrition, Retail Sales Forecasting "
    "and Inventory Management"
)

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Employee Analytics",
        "Sales Forecasting",
        "Inventory Management"
    ]
)

if page == "Home":

    st.header("Project Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Modules", "3")
    c2.metric("ML Algorithm", "Random Forest")
    c3.metric("Forecasting", "Prophet")

    st.success(
        "Welcome to Workforce Retail Analytics!"
    )

elif page == "Employee Analytics":

    st.header("Employee Attrition Analysis")

    if os.path.exists("employee_report.csv"):

        df = pd.read_csv("employee_report.csv")

        st.dataframe(df.head(20))

        st.metric(
            "Employees Analyzed",
            len(df)
        )

    else:
        st.warning(
            "Please run employee_analysis.py first."
        )

elif page == "Sales Forecasting":

    st.header("Retail Sales Forecast")

    if os.path.exists("sales_forecast.csv"):

        df = pd.read_csv("sales_forecast.csv")

        st.line_chart(
            df.set_index("ds")[["yhat"]]
        )

        st.dataframe(df.tail(7))

    else:
        st.warning(
            "Please run sales_forecasting.py first."
        )

elif page == "Inventory Management":

    st.header("Inventory Management")

    if os.path.exists("inventory_report.csv"):

        df = pd.read_csv("inventory_report.csv")

        st.dataframe(df)

        st.metric(
            "Low Stock Products",
            int((df["Status"] == "LOW STOCK").sum())
        )

    else:
        st.warning(
            "Please run inventory_analysis.py first."
        )