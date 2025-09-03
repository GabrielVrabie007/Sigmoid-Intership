import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time



def show_exploration_page():
    try:
        df = pd.read_csv("data/Wine.csv")
    except FileNotFoundError:
        st.error("Error: 'Wine.csv' not found. Please upload the file to the app's directory.")
        return

    st.title("Dataset Exploration")

    st.header("Dataset Header")
    st.write("Here is a quick look at the first 5 rows of the dataset.")
    st.dataframe(df.head())
    st.header("Column Descriptions")
    st.write("Here is a short explanation of each column in the dataset:")
    st.latex(r"""
        \begin{aligned}
        \textbf{Alcohol} & : \; \text{How strong the wine is} \\
        \textbf{Malic\_Acid} & : \; \text{The amount of malic acid in it} \\
        \textbf{Ash} & : \; \text{The ash left after burning the wine} \\
        \textbf{Ash\_Alcalinity} & : \; \text{How alkaline that ash is} \\
        \textbf{Magnesium} & : \; \text{How much magnesium is in the wine} \\
        \textbf{Total\_Phenols} & : \; \text{The total phenols (flavor \& antioxidant stuff)} \\
        \textbf{Flavanoids} & : \; \text{Flavanoids, a type of phenol} \\
        \textbf{Nonflavanoid\_Phenols} & : \; \text{Other phenols that aren’t flavanoids} \\
        \textbf{Proanthocyanins} & : \; \text{Another type of phenol in wine} \\
        \textbf{Color\_Intensity} & : \; \text{How dark or intense the color is} \\
        \textbf{Hue} & : \; \text{The color tone of the wine} \\
        \textbf{OD280} & : \; \text{A measure of certain compounds in diluted wine} \\
        \textbf{Proline} & : \; \text{Amount of the amino acid proline} \\
        \textbf{Customer\_Segment} & : \; \text{Which type or class of wine it belongs to} \\
        \end{aligned}
        """)

    st.subheader("1. Feature Correlation Heatmap")
    st.write("This heatmap shows the correlation between all numerical features. Values closer to 1 or -1 indicate a stronger relationship.")

    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Wine Features")
    st.pyplot(fig)

    st.subheader("2. Color Intensity vs. Alcohol by Customer Segment")
    st.write("This scatter plot shows the relationship between alcohol quantity and color intensity, with each point colored by its customer segment.")

    fig_scatter = px.scatter(
        df,
        x="Alcohol",
        y="Color_Intensity",
        color="Customer_Segment",
        title="Alcohol vs. Color Intensity",
        labels={"Alcohol": "Alcohol (%)", "Color_Intensity": "Color Intensity"},
        hover_name=df.index
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("3. Distribution of Proline by Customer Segment")
    st.write("The box plot below illustrates the distribution of the 'Proline' feature for each of the three customer segments.")

    fig_box = px.box(
        df,
        x="Customer_Segment",
        y="Proline",
        title="Proline Distribution by Customer Segment",
        color="Customer_Segment",
        labels={"Customer_Segment": "Customer Segment", "Proline": "Proline (mg/L)"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("4. Parallel Coordinates Plot of Key Features")
    st.write("Interesting plot to visualize how the different customer segments vary across multiple features at once. Each line represents a single wine, and you can see how the segments group together based on their characteristic values.")

    # Select key features
    features = ['Alcohol', 'Flavanoids', 'Color_Intensity', 'Proline']

    try:
        fig_parallel = px.parallel_coordinates(
            df,
            color="Customer_Segment",
            dimensions=features,
            title="Parallel Coordinates of Wine Features by Customer Segment"
        )
        st.plotly_chart(fig_parallel, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate the parallel coordinates plot. Error: {e}")


# One special effect for fun :)
def snow_effect():
    with st.spinner("Preparing your exploration..."):
        time.sleep(1.0)

        st.snow()

        progress_text = "Loading analysis..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
