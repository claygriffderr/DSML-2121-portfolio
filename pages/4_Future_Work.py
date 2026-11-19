import streamlit as st

st.set_page_config(page_title="Future Work", page_icon="🔭", layout="wide")

# ---------- TODO: Fill in your own text (markdown works here) ----------
NEXT_STEPS = """
- Identify a similar dataset to test model quality on unseen data 
- Data mine an online wine retailer for quality ratings and alcohol content; predict ratings from alcohol content alone.
"""

REFLECTION = """
To display the amount of information in the charts I opted for a single column layout for the analysis pages. The one drawback of this is that when toggling "Exclude outliers" the user has to scroll up and down the page, making comparisons difficult. Using previous projects made structuring the layout much simpler.
"""

# ---------- Layout ----------
st.title("🔭 Future Work")

st.header("Next Steps")
st.markdown(NEXT_STEPS)
st.divider()

st.header("Reflection")
st.markdown(REFLECTION)
