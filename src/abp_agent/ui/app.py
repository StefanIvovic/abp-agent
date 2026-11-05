"""Placeholder page until the chat arrives: shows whether the platform is healthy."""

import os

import httpx
import streamlit as st

st.title("ABP Agent")
report = httpx.get(f"{os.environ['API_URL']}/health", timeout=10).json()
st.subheader(f"Platform: {report['status']}")
for name, c in report["components"].items():
    st.write(f"{'✅' if c['ok'] else '❌'} **{name}**: {c['detail']}")
