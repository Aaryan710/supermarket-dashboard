import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------
# Page configuration
st.set_page_config(page_title="Supermarket Sales Dashboard", layout="wide")
st.title("🛒 Supermarket Sales Analysis Dashboard")
st.markdown("Interactive dashboard to explore sales, products, customer behavior, and profitability.")

# ------------------------------
# Load and preprocess data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    # Convert date and time
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')  # MM/DD/YYYY
    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
    # Extract features
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month_name()
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    
    def time_period(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    
    df['Time_Period'] = df['Hour'].apply(time_period)
    
    # Rename columns for clarity
    df.rename(columns={'gross income': 'Gross_Income', 'Rating': 'Rating'}, inplace=True)
    return df

data = load_data()

# ------------------------------
# Sidebar filters
st.sidebar.header("🔍 Filter Data")
cities = st.sidebar.multiselect("City", options=data['City'].unique(), default=data['City'].unique())
branches = st.sidebar.multiselect("Branch", options=data['Branch'].unique(), default=data['Branch'].unique())
cust_types = st.sidebar.multiselect("Customer Type", options=data['Customer type'].unique(), default=data['Customer type'].unique())
genders = st.sidebar.multiselect("Gender", options=data['Gender'].unique(), default=data['Gender'].unique())
product_lines = st.sidebar.multiselect("Product Line", options=data['Product line'].unique(), default=data['Product line'].unique())

# Date range filter
min_date = data['Date'].min()
max_date = data['Date'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Apply filters
filtered_df = data[
    (data['City'].isin(cities)) &
    (data['Branch'].isin(branches)) &
    (data['Customer type'].isin(cust_types)) &
    (data['Gender'].isin(genders)) &
    (data['Product line'].isin(product_lines)) &
    (data['Date'] >= pd.to_datetime(start_date)) &
    (data['Date'] <= pd.to_datetime(end_date))
]

# ------------------------------
# Key Performance Indicators (KPIs)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_sales = filtered_df['Sales'].sum()
    st.metric("💰 Total Sales", f"${total_sales:,.2f}")
with col2:
    total_profit = filtered_df['Gross_Income'].sum()
    st.metric("📈 Total Gross Income", f"${total_profit:,.2f}")
with col3:
    avg_rating = filtered_df['Rating'].mean()
    st.metric("⭐ Average Rating", f"{avg_rating:.2f}/10")
with col4:
    num_transactions = filtered_df.shape[0]
    st.metric("🧾 Transactions", f"{num_transactions:,}")
with col5:
    avg_basket = filtered_df['Sales'].mean()
    st.metric("🛍️ Avg Basket Size", f"${avg_basket:.2f}")

st.markdown("---")

# ------------------------------
# Tabs for different analysis sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sales Overview", "🏷️ Product Analysis", "👥 Customer Insights", "⏰ Time Analysis", "📈 Profit & Ratings"])

# ================= Tab 1: Sales Overview =================
with tab1:
    st.subheader("Sales Performance by Different Dimensions")
    
    # Sales by Day of Week
    day_sales = filtered_df.groupby('Day_of_Week')['Sales'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    fig1 = px.bar(day_sales, x=day_sales.index, y='Sales', title="Total Sales by Day of Week",
                  color=day_sales.values, color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        city_sales = filtered_df.groupby('City')['Sales'].sum().reset_index()
        fig2 = px.pie(city_sales, values='Sales', names='City', title="Sales Share by City", hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        payment_sales = filtered_df.groupby('Payment')['Sales'].sum().reset_index()
        fig3 = px.bar(payment_sales, x='Payment', y='Sales', title="Sales by Payment Method", color='Payment')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Sales by Branch
    branch_sales = filtered_df.groupby('Branch')['Sales'].sum().reset_index()
    fig4 = px.bar(branch_sales, x='Branch', y='Sales', title="Sales by Branch", color='Sales', color_continuous_scale='Reds')
    st.plotly_chart(fig4, use_container_width=True)

# ================= Tab 2: Product Analysis =================
with tab2:
    st.subheader("Product Performance")
    
    # Top products by revenue
    product_revenue = filtered_df.groupby('Product line')['Sales'].sum().sort_values(ascending=False).reset_index()
    fig5 = px.bar(product_revenue, x='Sales', y='Product line', orientation='h', title="Top Product Lines by Revenue",
                  color='Sales', color_continuous_scale='Viridis')
    st.plotly_chart(fig5, use_container_width=True)
    
    # Top products by quantity sold
    product_qty = filtered_df.groupby('Product line')['Quantity'].sum().sort_values(ascending=False).reset_index()
    fig6 = px.bar(product_qty, x='Quantity', y='Product line', orientation='h', title="Top Product Lines by Quantity Sold",
                  color='Quantity', color_continuous_scale='Oranges')
    st.plotly_chart(fig6, use_container_width=True)
    
    # Heatmap: Product sales by day
    product_day = filtered_df.groupby(['Product line', 'Day_of_Week'])['Sales'].sum().unstack().fillna(0)
    # Reorder days
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    product_day = product_day.reindex(columns=ordered_days)
    
    fig7 = px.imshow(product_day, text_auto='.0f', aspect="auto", color_continuous_scale='YlGnBu',
                     title="Heatmap: Sales by Product Line and Day of Week")
    st.plotly_chart(fig7, use_container_width=True)

# ================= Tab 3: Customer Insights =================
with tab3:
    st.subheader("Customer Segmentation and Behavior")
    
    col1, col2 = st.columns(2)
    with col1:
        cust_sales = filtered_df.groupby('Customer type')['Sales'].sum().reset_index()
        fig8 = px.bar(cust_sales, x='Customer type', y='Sales', title="Sales by Customer Type", color='Customer type')
        st.plotly_chart(fig8, use_container_width=True)
    with col2:
        gender_sales = filtered_df.groupby('Gender')['Sales'].sum().reset_index()
        fig9 = px.pie(gender_sales, values='Sales', names='Gender', title="Sales by Gender", hole=0.3)
        st.plotly_chart(fig9, use_container_width=True)
    
    # Average basket size by customer type
    basket_by_type = filtered_df.groupby('Customer type')['Sales'].mean().reset_index()
    fig10 = px.bar(basket_by_type, x='Customer type', y='Sales', title="Average Basket Size by Customer Type",
                   color='Customer type', text_auto='.2f')
    st.plotly_chart(fig10, use_container_width=True)
    
    # Member vs Normal spending distribution
    fig11 = px.box(filtered_df, x='Customer type', y='Sales', color='Gender', title="Sales Distribution by Customer Type and Gender")
    st.plotly_chart(fig11, use_container_width=True)

# ================= Tab 4: Time Analysis =================
with tab4:
    st.subheader("Sales Patterns Over Time")
    
    # Sales by month (assuming full year data)
    monthly_sales = filtered_df.groupby('Month')['Sales'].sum().reset_index()
    # Sort months chronologically
    month_order = ['January', 'February', 'March']  # data shows Jan-Mar only, adjust if more
    monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=month_order, ordered=True)
    monthly_sales = monthly_sales.sort_values('Month')
    fig12 = px.line(monthly_sales, x='Month', y='Sales', markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig12, use_container_width=True)
    
    # Sales by hour of day
    hourly_sales = filtered_df.groupby('Hour')['Sales'].sum().reset_index()
    fig13 = px.bar(hourly_sales, x='Hour', y='Sales', title="Sales by Hour of Day", color='Sales', color_continuous_scale='Plasma')
    st.plotly_chart(fig13, use_container_width=True)
    
    # Sales by Time Period (Morning/Afternoon/Evening/Night)
    period_sales = filtered_df.groupby('Time_Period')['Sales'].sum().reset_index()
    fig14 = px.pie(period_sales, values='Sales', names='Time_Period', title="Sales Distribution by Time Period", hole=0.3)
    st.plotly_chart(fig14, use_container_width=True)
    
    # Weekday vs Weekend
    filtered_df['Is_Weekend'] = filtered_df['Day_of_Week'].isin(['Saturday', 'Sunday'])
    weekend_sales = filtered_df.groupby('Is_Weekend')['Sales'].sum().reset_index()
    weekend_sales['Is_Weekend'] = weekend_sales['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})
    fig15 = px.bar(weekend_sales, x='Is_Weekend', y='Sales', title="Weekday vs Weekend Sales", color='Is_Weekend')
    st.plotly_chart(fig15, use_container_width=True)

# ================= Tab 5: Profit & Ratings =================
with tab5:
    st.subheader("Profitability and Customer Feedback")
    
    # Gross income by product line
    profit_by_product = filtered_df.groupby('Product line')['Gross_Income'].sum().sort_values(ascending=False).reset_index()
    fig16 = px.bar(profit_by_product, x='Gross_Income', y='Product line', orientation='h', title="Total Gross Income by Product Line",
                   color='Gross_Income', color_continuous_scale='Teal')
    st.plotly_chart(fig16, use_container_width=True)
    
    # Scatter: Sales vs Rating
    fig17 = px.scatter(filtered_df, x='Sales', y='Rating', color='Product line', size='Gross_Income',
                       title="Relationship between Sales, Rating, and Gross Income",
                       hover_data=['Customer type', 'Gender'])
    st.plotly_chart(fig17, use_container_width=True)
    
    # Rating distribution
    fig18 = px.histogram(filtered_df, x='Rating', nbins=20, color='Customer type', title="Distribution of Customer Ratings",
                         marginal='box')
    st.plotly_chart(fig18, use_container_width=True)
    
    # Average rating per product line
    rating_product = filtered_df.groupby('Product line')['Rating'].mean().sort_values(ascending=False).reset_index()
    fig19 = px.bar(rating_product, x='Rating', y='Product line', orientation='h', title="Average Rating by Product Line",
                   color='Rating', color_continuous_scale='magma')
    st.plotly_chart(fig19, use_container_width=True)

# ------------------------------
# Footer
st.markdown("---")
st.caption("Dashboard built with Streamlit, Plotly, and Pandas. Data source: Supermarket Sales Dataset.")