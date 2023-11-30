import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import os

from datetime import datetime

def main():
    st.set_page_config(page_title="Toronto Bicycle Thefts",
                       page_icon=":bar_chart:", layout="wide")

    st.title("Group 1 - Toronto Bicycle Thefts Analysis & Prediction")
    st.sidebar.title("Parameters")

    data = load()

    st.sidebar.subheader("Display Data")
    if st.sidebar.checkbox("Show (First 100 rows)", False):
        st.subheader("Bicycle Thefts dataset")
        st.write(data.head(100))

    st.sidebar.subheader("Data Analysis")
    if st.sidebar.checkbox("Data Info", False):
        st.subheader("Data Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)

    if st.sidebar.checkbox("Data Shape", False):
        st.subheader("Data Shape")
        st.write(data.shape)

    if st.sidebar.checkbox("Data Column Types", False):
        st.subheader("Data Column Types")
        st.write(data.dtypes)

    if st.sidebar.checkbox("Bike Cost Descriptive Statistics", False):
        st.subheader("Bike Cost Descriptive Statistics")
        st.write(data['BIKE_COST'].describe())

    if st.sidebar.checkbox("Bike Cost Distribution", False):
        st.subheader("Bike Cost Distribution")
        bike_costs = data["BIKE_COST"]
        min_cost = bike_costs.min()
        max_cost = bike_costs.max()

        fig = px.histogram(bike_costs, x="BIKE_COST", range_x=[
                           min_cost, max_cost], title="Bike Cost Distribution")
        
        fig.update_layout(
            title="Bike Cost Distribution",
            xaxis_title="Bike Cost",
            yaxis_title="Frequency"
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.checkbox("Bike Cost by Premises", False):
        st.subheader("Bike Cost by Premises")
        sorted_by_bike_cost = data.sort_values(by="BIKE_COST", ascending=False)
        premises = sorted_by_bike_cost["PREMISES_TYPE"]
        costs = sorted_by_bike_cost["BIKE_COST"]

        fig = go.Figure(data=[go.Bar(x=costs, y=premises, orientation='h')])

        fig.update_layout(
            title="Bike Costs by Premises",
            xaxis_title="Bike Costs",
            yaxis_title="Premises"
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.checkbox("Bike Thefts Frequency by Year", False):
        st.subheader("Bike Thefts Frequency per Year")
        subset = data.drop(data[data["OCC_YEAR"] < 2014].index)

        fig = px.histogram(subset["OCC_YEAR"])
        
        fig.update_layout(
            title="Bike Thefts Frequency by Year",
            xaxis_title="Year",
            yaxis_title="Number of Bike Thefts",
            bargap=0.2
        )
        
        st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.checkbox("Bike Status by Month", False):
        st.subheader("Bike Status by Month")
        sorted_by_occ_month = data.copy()
        sorted_by_occ_month["OCC_MONTH"] = sorted_by_occ_month["OCC_MONTH"].apply(
            replace_month_name_with_month_number)

        sorted_by_occ_month = sorted_by_occ_month.sort_values(by="OCC_MONTH")

        crosstab_data = pd.crosstab(
            sorted_by_occ_month["OCC_MONTH"], sorted_by_occ_month["STATUS"])

        st.write(crosstab_data)

        fig = px.bar(crosstab_data, x=crosstab_data.index,
                     y=crosstab_data.columns, barmode="stack")

        fig.update_layout(
            title="Bike Status by Month",
            xaxis_title="Month",
            yaxis_title="Status Frequency"
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.checkbox("Status Percentage", False):
        st.subheader("Status Percentage")
        status_counts = data['STATUS'].value_counts()

        fig = px.pie(status_counts, values=status_counts.values,
                     names=status_counts.index, title="Status Percentage")

        st.plotly_chart(fig, use_container_width=True)

@st.cache_data(persist=True)
def load():
    path = "./"
    filename = "bicycle-thefts-open-data.csv"
    filepath = os.path.join(path, filename)

    data = pd.read_csv(filepath)
    return data

def replace_month_name_with_month_number(month_name):
    return datetime.strptime(month_name, "%B").month

if __name__ == "__main__":
    main()
