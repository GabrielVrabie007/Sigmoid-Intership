import streamlit as st
from .explore_page import show_exploration_page,snow_effect


def show_homepage():
    st.title("Wine Quality Project")
    st.write(
        """
        Welcome to the analysis of the Wine Dataset. This project explores chemical properties to understand wine characteristics and predict by what customer segment is.
        """
    )
    st.subheader("The Wine Dataset")
    st.write(
        """
        The dataset contains the results of a chemical analysis of wines grown in the same region, but derived from three different types of customers. The analysis determines the quantities of 13 constituents found in each of the three types of wine.
        """
    )
    st.subheader("Our Approach")
    st.write(
        """
        To determine the quality of the wines, we'll be using a **RandomForestClassifier**. This powerful machine learning model is an ensemble learning method that builds multiple decision trees during training and outputs a class that is the mode of the classes of the individual trees. It's an excellent choice for this multi-class classification problem.
        """
    )

    st.image(
        "data/cover.png",
        use_container_width=True
    )

    if st.button("🚀 Explore the Analysis"):
        snow_effect()
        show_exploration_page()
