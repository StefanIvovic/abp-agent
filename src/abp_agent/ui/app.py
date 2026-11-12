"""Entry point: the sidebar menu with the chat and the eval datasets."""

from pathlib import Path

import streamlit as st

here = Path(__file__).parent
st.navigation([
    st.Page(here / "chat.py", title="Chat", icon=":material/chat:", default=True),
    st.Page(here / "datasets.py", title="Eval datasets", icon=":material/fact_check:"),
]).run()
