import streamlit as st
from pages.home_page import show_homepage
from pages.explore_page import show_exploration_page,snow_effect
from pages.model_training_page import show_model_training_page

st.set_page_config(
    page_title="Wine Quality Analysis",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
        .stButton>button {
            background-color: #0068c9;
            color: white;
            border-radius: 0.5rem;
            border: none;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #004d9c;
        }
        .st-emotion-cache-1kyx2w7 {
            color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)


page = st.sidebar.radio("Navigation", ["Home", "Exploration", "Model Training"])

if page == "Home":
    show_homepage()
elif page == "Exploration":
    snow_effect()
    show_exploration_page()
elif page == "Model Training":
    show_model_training_page()
