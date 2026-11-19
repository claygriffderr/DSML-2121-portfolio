import streamlit as st
from pathlib import Path

st.title("👋 My Bio")

NAME = "Clay Griffin-Derr"
PROGRAM = "Data Science and Machine Learning"
INTRO = """
I specialize in Natural Language Processing such as text cleaning and information retrieval.
I am a member of the JRaviLab in the Anschutz Department of BioMedical Informatics.

**Programming skills**
- Python: PyTorch, scikit-learn, NLTK, spaCy, Ollama, HuggingFace
- Java
- Agentic local open weight LLMs
- R: Seurat
- High Performance Computing/remote SSH with Linux
"""



PHOTO_PATH = "Profile.PNG"  # File name inside streamlit_DSML/assets/


def find_photo(filename=PHOTO_PATH):
    # Try common locations for multipage apps
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    candidates = [
        script_dir / "assets" / filename,          # pages/assets/...
        script_dir.parent / "assets" / filename,   # root/assets/... (common)
        Path("assets") / filename,                 # cwd/assets/...
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

photo_src = find_photo(PHOTO_PATH)

# ---------- Layout ----------
col1, col2 = st.columns([1, 2], vertical_alignment="center")

with col1:
    if photo_src:
        st.image(photo_src, caption=f"{NAME} - trip to Chicago", alt=f"Headshot of {NAME}", width="stretch")
    else:
        st.info(
            "📷 Place `Your_Photo.jpg` inside an `assets/` folder at the app root "
            "or update the path in `find_photo()`."
        )
        
with col2:
    st.subheader(NAME)
    st.write(f"Major: {PROGRAM} at Metropolitan State University of Denver")
    st.write(INTRO)


st.divider()

