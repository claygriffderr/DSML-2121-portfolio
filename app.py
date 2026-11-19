import streamlit as st

st.set_page_config(
    page_title="My Streamlit Site",
    page_icon="📊",
    layout="wide"
)

st.title(" Welcome to My Streamlit Site")
st.markdown(
    """
    This is a three-page Streamlit app to introduce myself and analyze the Red Wine dataset from Kaggle.
    Pages:
    - **Bio**: Clay Griffin-Derr
    - **EDA**: Get to know the dataset
    - **Dashboard**: Run regression and identify the most important features of the dataset
    - **Future Work**: A brief description of my next steps and process

    Use the left sidebar to switch pages.
    """
)


st.caption("Built with Streamlit • Code generated with Claude Opus 5.5 on medium and high settings")
