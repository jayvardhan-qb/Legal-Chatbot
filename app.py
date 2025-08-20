import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Legal AI Chatbot", layout="wide")
st.title("Legal AI Assistant")

API_URL = "http://127.0.0.1:8000"

st.sidebar.title("Features")
mode = st.sidebar.radio("Select a feature:", ["Legal Q&A", "Summarize Contract", "Generate NDA"])

if mode == "Legal Q&A":
    st.subheader("Ask a Legal Question based on Indian Laws")
    question = st.text_area("Enter your legal question:")
    if st.button("Get Answer") and question.strip():
        try:
            res = requests.post(
                f"{API_URL}/ask", 
                json={"question": question}
            )
            res.raise_for_status()
            # if res.ok:
            st.markdown("### Response")
            st.success(res.json()["response"])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching response: {str(e)}")

elif mode == "Summarize Contract":
    st.subheader("Upload a contract to get plain-language summary")
    uploaded_file = st.file_uploader("Upload PDF File", type="pdf")
    if st.button("Summarize") and uploaded_file:
        try:    
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            res = requests.post(
                f"{API_URL}/summarize",
                files = files
            )
            # if res.ok:
            res.raise_for_status()
            st.markdown("### Summary")
            st.success(res.json()["summary"])
        except requests.exceptions.RequestException as e:
            st.error(f"Error summarizing document: {str(e)}")

elif mode == "Generate NDA":
    st.subheader("Auto-Generate a Non-Disclosure Agreement (NDA)")
    col1, col2 = st.columns(2)
    with col1:
        party1 = st.text_input("Party 1 Name")
    with col2:
        party2 = st.text_input("Party 2 Name")
    purpose = st.text_input("Purpose of NDA")
    duration = st.text_input("Duration of NDA (e.g., 2 years)")
    date = st.date_input("Date of Agreement", datetime.now())

    if st.button("Generate NDA") and all([party1, party2, purpose, duration]):
        try:
            res = requests.post(
                f"{API_URL}/generate-contract", 
                json={
                    "party1": party1,
                    "party2": party2,
                    "purpose": purpose,
                    "duration": duration,
                    "date": date.strftime("%Y-%m-%d")
                }
            )
        # if res.ok:
            res.raise_for_status()
            st.markdown("### Generated NDA")
            st.text_area("NDA Content", res.json()["contract"], height=400)
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating NDA: {str(e)}")