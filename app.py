"""
This is the main application file for the AI Resume Screening Assistant.
It creates a Streamlit web interface for users to:
- Upload a job description.
- Upload a folder of resumes.
- View a ranked list of candidates with AI-generated summaries.
- Export the results to a CSV file.
"""

import os
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

# Import custom modules
from embeddings import get_documents_from_folder, create_vector_store
from ranker import rank_resumes
from summarizer import generate_summary

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- App Title and Description ---
st.title("🧠 AI-Powered Resume Screening Assistant")
st.markdown("""
Welcome to the AI Resume Screening Assistant! This tool helps you rank resumes based on a job description
and provides AI-generated summaries for the top candidates.
""")

# --- Helper Functions ---
def save_uploaded_files(uploaded_files: List, directory: str) -> None:
    """Saves uploaded files to a specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in uploaded_files:
        with open(os.path.join(directory, file.name), "wb") as f:
            f.write(file.getbuffer())

def process_and_display_results(job_description: str, resume_folder: str) -> None:
    """The main logic to process resumes and display ranked results."""
    with st.spinner("Extracting text from resumes..."):
        docs = get_documents_from_folder(resume_folder)
        if not docs:
            st.error("No resumes found or could not extract text. Please check the folder.")
            return

    with st.spinner("Creating vector store and embeddings..."):
        try:
            vector_store = create_vector_store(docs)
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return

    with st.spinner("Ranking resumes against the job description..."):
        ranked_candidates = rank_resumes(vector_store, job_description, top_k=10)
        if not ranked_candidates:
            st.warning("Could not rank any candidates.")
            return

    st.success(f"Successfully ranked {len(ranked_candidates)} candidates!")

    # --- Display Ranked Candidates ---
    st.header("Top Candidates")
    results_list = []
    for i, candidate in enumerate(ranked_candidates):
        st.subheader(f"#{i+1} - {candidate['source']} (Score: {candidate['score']:.2f})")

        with st.spinner(f"Generating AI summary for {candidate['source']}..."):
            summary = generate_summary(candidate, job_description)
            st.info(summary)

        results_list.append({
            "Rank": i + 1,
            "File": candidate['source'],
            "Similarity Score": f"{candidate['score']:.2f}",
            "AI Summary": summary
        })

    # --- Business Value & Export ---
    st.header("📊 Business Value & Export")
    # Simple metric calculation
    time_saved = len(docs) * 5  # Assuming 5 minutes saved per resume
    st.metric(label="Estimated Time Saved (minutes)", value=time_saved)

    # Convert results to DataFrame for download
    results_df = pd.DataFrame(results_list)
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ranked_resumes.csv",
        mime="text/csv",
    )

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Inputs")
    # 1. Job Description
    jd_text = st.text_area("Enter the Job Description here:", height=200)

    # 2. Resume Folder
    uploaded_resumes = st.file_uploader(
        "Upload resumes (PDF or DOCX):",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    # 3. Process Button
    process_button = st.button("Screen Resumes")

# --- Main Content Area ---
if process_button:
    if not jd_text:
        st.error("Please provide a job description.")
    elif not uploaded_resumes:
        st.error("Please upload at least one resume.")
    else:
        # Define a temporary folder to store uploaded resumes
        temp_resume_folder = "temp_resumes"
        save_uploaded_files(uploaded_resumes, temp_resume_folder)

        # Run the main processing logic
        process_and_display_results(jd_text, temp_resume_folder)
