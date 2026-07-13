import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(

    page_title="Sales Forecast Dashboard",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded"
)

st.title("📈 Sales Forecasting & Demand Intelligence Dashboard")

st.markdown("---")

@st.cache_data
def load_data():

    df = pd.read_csv(

        "sales_cleaned.csv",

        parse_dates=["Order Date","Ship Date"]

    )

    return df

sales_df = load_data()

with st.sidebar:

    selected = option_menu(

        menu_title="Navigation",

        options=[

            "Dashboard",

            "EDA",

            "Forecast",

            "Category Forecast",

            "Region Forecast",

            "Anomaly Detection",

            "Demand Segmentation"

        ],

        icons=[

            "house",

            "bar-chart",

            "graph-up",

            "boxes",

            "geo",

            "exclamation-triangle",

            "diagram-3"

        ],

        default_index=0

    )

if selected == "Dashboard":

    st.header("Business Overview")

# ==========================================
# EDA PAGE
# ==========================================

elif selected == "EDA":

    st.header("📊 Exploratory Data Analysis")

    st.subheader("Filters")

    years = sorted(sales_df["Year"].unique())

    selected_year = st.selectbox(
        "Select Year",
        ["All"] + list(years)
    )

    regions = sorted(sales_df["Region"].unique())

    selected_region = st.selectbox(
        "Select Region",
        ["All"] + list(regions)
    )

    categories = sorted(sales_df["Category"].unique())

    selected_category = st.selectbox(
        "Select Category",
        ["All"] + list(categories)
    )
    filtered_df = sales_df.copy()

    if selected_year != "All":
        filtered_df = filtered_df[
            filtered_df["Year"] == selected_year
        ]

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == selected_category
        ]
    
    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${filtered_df['Sales'].sum():,.2f}"
    )

    col2.metric(
        "Orders",
        filtered_df["Order ID"].nunique()
    )

    col3.metric(
        "Customers",
        filtered_df["Customer ID"].nunique()
    )
    monthly = (

        filtered_df

        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]

        .sum()

        .reset_index()

    )
    fig = px.line(

        monthly,

        x="Order Date",

        y="Sales",

        markers=True,

        title="Monthly Sales Trend"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    state_sales = (

        filtered_df

        .groupby("State")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .head(15)

        .reset_index()

    )
    fig = px.bar(

        state_sales,

        x="Sales",

        y="State",

        orientation="h",

        color="Sales",

        title="Top States"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    segment = (

        filtered_df

        .groupby("Segment")["Sales"]

        .sum()

        .reset_index()

    )
    fig = px.pie(

        segment,

        names="Segment",

        values="Sales",

        title="Sales by Customer Segment"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    ship = (

        filtered_df

        .groupby("Ship Mode")["Sales"]

        .sum()

        .reset_index()

    )
    fig = px.bar(

        ship,

        x="Ship Mode",

        y="Sales",

        color="Ship Mode",

        title="Sales by Shipping Mode"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    customers = (

        filtered_df

        .groupby("Customer Name")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )
    fig = px.bar(

        customers,

        x="Sales",

        y="Customer Name",

        orientation="h",

        title="Top Customers"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    products = (

        filtered_df

        .groupby("Product Name")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )
    fig = px.bar(

        products,

        x="Sales",

        y="Product Name",

        orientation="h",

        title="Top Products"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    category_region = (

        filtered_df

        .pivot_table(

            values="Sales",

            index="Region",

            columns="Category",

            aggfunc="sum"

        )

    )
    fig = px.imshow(

        category_region,

        text_auto=True,

        aspect="auto",

        title="Category vs Region"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    corr = filtered_df[

        [

            "Sales",

            "Shipping Days",

            "Year",

            "Month",

            "Quarter"

        ]

    ].corr()
    fig = px.imshow(

        corr,

        text_auto=True,

        title="Correlation Matrix"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.histogram(

        filtered_df,

        x="Sales",

        nbins=40,

        title="Sales Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.box(

        filtered_df,

        y="Sales",

        color="Category",

        title="Sales Boxplot"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Filtered Dataset")

    st.dataframe(

        filtered_df,

        use_container_width=True,

        height=400

    )
    csv = filtered_df.to_csv(index=False)

    st.download_button(

        "Download Filtered Dataset",

        csv,

        "filtered_sales.csv",

        "text/csv"

    )

# ==========================================
# FORECAST PAGE
# ==========================================

elif selected == "Forecast":

    st.header("📈 Sales Forecasting")
    model_comparison = pd.read_csv("model_comparison.csv")

    sarima_forecast = pd.read_csv("sarima_forecast.csv")

    prophet_forecast = pd.read_csv("prophet_forecast.csv")

    xgb_forecast = pd.read_csv("xgboost_forecast.csv")

    st.subheader("Model Performance")

    st.dataframe(
        model_comparison,
        use_container_width=True
    )
    best_model = model_comparison.loc[
        model_comparison["RMSE"].idxmin(),
        "Model"
    ]
    st.success(f"🏆 Best Model : {best_model}")
    fig = px.bar(

        model_comparison,

        x="Model",

        y="RMSE",

        color="Model",

        title="RMSE Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.bar(

        model_comparison,

        x="Model",

        y="MAE",

        color="Model",

        title="MAE Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.bar(

        model_comparison,

        x="Model",

        y="MAPE",

        color="Model",

        title="MAPE Comparison"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Lowest MAE",
        f"{model_comparison['MAE'].min():.2f}"
    )

    c2.metric(
        "Lowest RMSE",
        f"{model_comparison['RMSE'].min():.2f}"
    )

    c3.metric(
        "Lowest MAPE",
        f"{model_comparison['MAPE'].min():.2f}%"
    )
    st.subheader("SARIMA Forecast")

    st.dataframe(
        sarima_forecast,
        use_container_width=True
    )
    fig = px.line(

        sarima_forecast,

        x=sarima_forecast.columns[0],

        y=sarima_forecast.columns[1],

        markers=True,

        title="SARIMA Future Forecast"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Prophet Forecast")

    st.dataframe(
        prophet_forecast,
        use_container_width=True
    )
    fig = px.line(

        prophet_forecast,

        x="ds",

        y="yhat",

        markers=True,

        title="Prophet Future Forecast"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("XGBoost Forecast")

    st.dataframe(
        xgb_forecast,
        use_container_width=True
    )
    fig = px.line(

        xgb_forecast,

        x=xgb_forecast.columns[0],

        y=xgb_forecast.columns[1],

        markers=True,

        title="XGBoost Future Forecast"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    forecast_compare = pd.read_csv(
        "forecast_comparison.csv"
    )
    st.subheader("Forecast Comparison")

    st.dataframe(
        forecast_compare,
        use_container_width=True
    )
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_compare["Month"],
            y=forecast_compare["SARIMA"],
            mode="lines+markers",
            name="SARIMA"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_compare["Month"],
            y=forecast_compare["Prophet"],
            mode="lines+markers",
            name="Prophet"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_compare["Month"],
            y=forecast_compare["XGBoost"],
            mode="lines+markers",
            name="XGBoost"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Download Results")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.download_button(

            "SARIMA Forecast",

            sarima_forecast.to_csv(index=False),

            "sarima.csv"

        )

    with c2:

        st.download_button(

            "Prophet Forecast",

            prophet_forecast.to_csv(index=False),

            "prophet.csv"

        )

    with c3:

        st.download_button(

            "XGBoost Forecast",

            xgb_forecast.to_csv(index=False),

            "xgboost.csv"

        )
    st.info(

        f"""
### Business Recommendation

After comparing all forecasting models using

• MAE

• RMSE

• MAPE

**{best_model}** achieved the lowest forecasting error.

It is therefore recommended for future demand forecasting and inventory planning.

"""
    )

# ==========================================
# CATEGORY FORECAST
# ==========================================

elif selected == "Category Forecast":

    st.header("📦 Category Forecast")
    category_forecast = pd.read_csv(
        "category_forecast.csv"
    )
    st.subheader("Next 3-Month Forecast")

    st.dataframe(
        category_forecast,
        use_container_width=True
    )
    fig = go.Figure()

    for col in category_forecast.columns:

        fig.add_trace(

            go.Scatter(

                x=[

                    "Month 1",

                    "Month 2",

                    "Month 3"

                ],

                y=category_forecast[col],

                mode="lines+markers",

                name=col

            )

        )

    fig.update_layout(

        title="Category Forecast",

        xaxis_title="Forecast Month",

        yaxis_title="Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Furniture",

        f"${category_forecast['Furniture'].sum():,.0f}"

    )

    c2.metric(

        "Office Supplies",

        f"${category_forecast['Office Supplies'].sum():,.0f}"

    )

    c3.metric(

        "Technology",

        f"${category_forecast['Technology'].sum():,.0f}"

    )
    total = category_forecast.sum()

    best = total.idxmax()

    st.success(

        f"Highest Expected Demand : {best}"

    )
    st.download_button(

        "Download Category Forecast",

        category_forecast.to_csv(index=False),

        "category_forecast.csv"

    )
# ==========================================
# REGION FORECAST
# ==========================================

elif selected == "Region Forecast":

    st.header("🌍 Region Forecast")
    region_forecast = pd.read_csv(

        "region_forecast.csv"

    )
    st.subheader("Next 3-Month Forecast")

    st.dataframe(

        region_forecast,

        use_container_width=True

    )
    fig = go.Figure()

    for col in region_forecast.columns:

        fig.add_trace(

            go.Scatter(

                x=[

                    "Month 1",

                    "Month 2",

                    "Month 3"

                ],

                y=region_forecast[col],

                mode="lines+markers",

                name=col

            )

        )

    fig.update_layout(

        title="Region Forecast",

        xaxis_title="Forecast Month",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    c1,c2,c3,c4 = st.columns(4)

    c1.metric(

        "East",

        f"${region_forecast['East'].sum():,.0f}"

    )

    c2.metric(

        "West",

        f"${region_forecast['West'].sum():,.0f}"

    )

    c3.metric(

        "South",

        f"${region_forecast['South'].sum():,.0f}"

    )

    c4.metric(

        "Central",

        f"${region_forecast['Central'].sum():,.0f}"

    )
    region_total = region_forecast.sum()

    best_region = region_total.idxmax()

    st.success(

        f"Highest Expected Sales Region : {best_region}"

    )
    pie = pd.DataFrame({

        "Region":region_total.index,

        "Sales":region_total.values

    })
    fig = px.pie(

        pie,

        names="Region",

        values="Sales",

        title="Forecast Share by Region"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    st.download_button(

        "Download Region Forecast",

        region_forecast.to_csv(index=False),

        "region_forecast.csv"

    )
    st.info("""

### Business Insights

• Technology is expected to generate the highest revenue.

• Office Supplies maintain stable demand.

• Furniture shows seasonal variation.

Inventory should be planned according to forecasted demand.

""")
    st.info("""

### Business Insights

• Focus inventory on the highest-demand region.

• Allocate warehouse capacity based on forecast.

• Optimize logistics before seasonal peaks.

""")

# ==========================================
# ANOMALY DETECTION
# ==========================================

elif selected == "Anomaly Detection":

    st.header("🚨 Anomaly Detection Dashboard")
    iso_df = pd.read_csv("isolation_forest_anomalies.csv")

    z_df = pd.read_csv("zscore_anomalies.csv")

    sales = pd.read_csv(
        "sales_cleaned.csv",
        parse_dates=["Order Date"]
    )
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Transactions",
        len(sales)
    )

    c2.metric(
        "Isolation Forest",
        len(iso_df)
    )

    c3.metric(
        "Z-Score",
        len(z_df)
    )
    fig = px.line(

        sales,

        x="Order Date",

        y="Sales",

        title="Sales Timeline"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Isolation Forest Anomalies")
    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=sales["Order Date"],

            y=sales["Sales"],

            mode="markers",

            marker=dict(color="lightblue"),

            name="Normal"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=iso_df["Order Date"],

            y=iso_df["Sales"],

            mode="markers",

            marker=dict(

                color="red",

                size=9

            ),

            name="Anomaly"

        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Z-Score Anomalies")
    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=sales["Order Date"],

            y=sales["Sales"],

            mode="markers",

            marker=dict(color="lightgreen"),

            name="Normal"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=z_df["Order Date"],

            y=z_df["Sales"],

            mode="markers",

            marker=dict(

                color="red",

                size=9

            ),

            name="Outlier"

        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.histogram(

        sales,

        x="Sales",

        nbins=40,

        title="Sales Distribution"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.box(

        sales,

        y="Sales",

        title="Sales Box Plot"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Isolation Forest Records")

    st.dataframe(

        iso_df,

        use_container_width=True

    )
    st.subheader("Z-Score Records")

    st.dataframe(

        z_df,

        use_container_width=True

    )
    c1,c2 = st.columns(2)

    with c1:

        st.download_button(

            "Download Isolation Forest",

            iso_df.to_csv(index=False),

            "IsolationForest.csv"

        )

    with c2:

        st.download_button(

            "Download ZScore",

            z_df.to_csv(index=False),

            "ZScore.csv"

        )
    st.info("""

### Business Insights

• Isolation Forest identifies anomalies using an unsupervised machine learning approach.

• Z-Score detects observations that are more than three standard deviations away from the mean.

• Most anomalies correspond to unusually large sales transactions.

• These transactions should be reviewed before inventory planning or forecasting.

""")

# ==========================================
# DEMAND SEGMENTATION
# ==========================================

elif selected == "Demand Segmentation":

    st.header("📦 Product Demand Segmentation")
    product_df = pd.read_csv("product_clusters.csv")
    high = len(product_df[product_df["Cluster"] == 0])
    medium = len(product_df[product_df["Cluster"] == 1])
    low = len(product_df[product_df["Cluster"] == 2])

    c1, c2, c3 = st.columns(3)

    c1.metric("Cluster 0", high)
    c2.metric("Cluster 1", medium)
    c3.metric("Cluster 2", low)
    fig = px.histogram(
        product_df,
        x="Cluster",
        color="Cluster",
        title="Products per Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    cluster_sales = (

        product_df

        .groupby("Cluster")["Total Sales"]

        .sum()

        .reset_index()

    )
    fig = px.bar(

        cluster_sales,

        x="Cluster",

        y="Total Sales",

        color="Cluster",

        title="Total Sales by Cluster"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    cluster_avg = (

        product_df

        .groupby("Cluster")["Average Sales"]

        .mean()

        .reset_index()

    )
    fig = px.bar(

        cluster_avg,

        x="Cluster",

        y="Average Sales",

        color="Cluster",

        title="Average Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    orders = (

        product_df

        .groupby("Cluster")["Order Count"]

        .sum()

        .reset_index()

    )
    fig = px.pie(

        orders,

        names="Cluster",

        values="Order Count",

        title="Orders by Cluster"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig = px.scatter(

        product_df,

        x="Average Sales",

        y="Total Sales",

        color="Cluster",

        size="Order Count",

        hover_name="Product Name",

        title="Demand Segmentation"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Clustered Products")

    st.dataframe(

        product_df,

        use_container_width=True,

        height=400

    )
    stats = (

        product_df

        .groupby("Cluster")[

            [

                "Total Sales",

                "Average Sales",

                "Order Count"

            ]

        ]

        .mean()

        .round(2)

    )

    st.subheader("Cluster Statistics")

    st.dataframe(
        stats,
        use_container_width=True
    )
    st.download_button(

        "Download Clustered Products",

        product_df.to_csv(index=False),

        "product_clusters.csv"

    )
    st.info("""

### Demand Segmentation Insights

🟢 Cluster 0

• High-demand products

• Maintain higher inventory

---

🟡 Cluster 1

• Medium-demand products

• Monitor demand regularly

---

🔴 Cluster 2

• Low-demand products

• Reduce inventory or offer promotions

""")
    selected_cluster = st.selectbox(

        "Select Cluster",

        sorted(product_df["Cluster"].unique())

    )

    st.dataframe(

        product_df[

            product_df["Cluster"] == selected_cluster

        ],

        use_container_width=True

    )
    st.subheader("Top Products in Selected Cluster")

    top = (

        product_df[

            product_df["Cluster"] == selected_cluster

        ]

        .sort_values(

            "Total Sales",

            ascending=False

        )

        .head(10)

    )

    fig = px.bar(

        top,

        x="Total Sales",

        y="Product Name",

        orientation="h",

        title="Top Products"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    

total_sales = sales_df["Sales"].sum()

total_orders = sales_df["Order ID"].nunique()

total_customers = sales_df["Customer ID"].nunique()

total_products = sales_df["Product ID"].nunique()

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "💰 Total Sales",

    f"${total_sales:,.2f}"

)

col2.metric(

    "📦 Orders",

    total_orders

)

col3.metric(

    "👥 Customers",

    total_customers

)

col4.metric(

    "🛒 Products",

    total_products

)

daily = sales_df.groupby(

    "Order Date"

)["Sales"].sum().reset_index()

fig = px.line(

    daily,

    x="Order Date",

    y="Sales",

    title="Daily Sales Trend"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

category = sales_df.groupby(

    "Category"

)["Sales"].sum().reset_index()

fig = px.bar(

    category,

    x="Category",

    y="Sales",

    color="Category",

    title="Sales by Category"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

region = sales_df.groupby(

    "Region"

)["Sales"].sum().reset_index()

fig = px.pie(

    region,

    names="Region",

    values="Sales",

    title="Regional Sales"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

top_products = (

    sales_df

    .groupby("Product Name")["Sales"]

    .sum()

    .nlargest(10)

    .reset_index()

)

fig = px.bar(

    top_products,

    x="Sales",

    y="Product Name",

    orientation="h",

    title="Top 10 Products"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

monthly = (

    sales_df

    .groupby(

        pd.Grouper(

            key="Order Date",

            freq="ME"

        )

    )["Sales"]

    .sum()

    .reset_index()

)

fig = px.area(

    monthly,

    x="Order Date",

    y="Sales",

    title="Monthly Sales"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

st.subheader("Dataset Preview")

st.dataframe(

    sales_df.head(20),

    use_container_width=True
)

csv = sales_df.to_csv(index=False)

st.download_button(

    label="Download Dataset",

    data=csv,

    file_name="sales.csv",

    mime="text/csv"
)

st.markdown("---")

st.caption(

    "Sales Forecasting & Demand Intelligence Dashboard"

)

# ==========================================
# EDA PAGE
# ==========================================