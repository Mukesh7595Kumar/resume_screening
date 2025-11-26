"""
This module handles the generation of AI-powered summaries for top-ranked candidates.

It provides functions to:
- Connect to a Large Language Model (LLM) like Google Gemini.
- Generate a concise summary for a candidate based on their resume and the job description.
"""

import os
import logging
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Configuration ---
# Make sure to set your GOOGLE_API_KEY in a .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logging.warning("GOOGLE_API_KEY not found. Please set it in your .env file.")
    # You might want to raise an exception here in a production environment
    # raise ValueError("GOOGLE_API_KEY is not set.")

def get_llm():
    """Initializes and returns the Google Gemini LLM."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        return None

def generate_summary(candidate: Dict[str, Any], job_description: str) -> str:
    """
    Generates a summary for a candidate using an LLM.

    Args:
        candidate (Dict[str, Any]): A dictionary containing candidate information
                                    (e.g., 'source', 'score', 'content').
        job_description (str): The job description text.

    Returns:
        str: The AI-generated summary.
    """
    llm = get_llm()
    if not llm:
        return "Could not generate summary due to LLM initialization error."

    # Define a prompt template for the LLM
    prompt_template = """
    You are an expert HR assistant. Your task is to generate a concise summary of a candidate's resume,
    highlighting their suitability for a specific job.

    **Job Description:**
    {job_description}

    **Candidate's Resume Content:**
    {resume_content}

    **Instructions:**
    - Based on the job description, highlight the candidate's key skills, relevant projects, and educational background.
    - Keep the summary to 3-4 bullet points.
    - Be objective and focus on the facts presented in the resume.
    - If the resume is a poor match, state that clearly.

    **Summary:**
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description", "resume_content"]
    )

    # Create the chain to run the LLM
    chain = prompt | llm

    try:
        # Invoke the LLM to get the summary
        response = chain.invoke({
            "job_description": job_description,
            "resume_content": candidate["content"]
        })
        summary = response.content
        logging.info(f"Generated summary for {candidate['source']}.")
        return summary
    except Exception as e:
        logging.error(f"Error generating summary for {candidate['source']}: {e}")
        return "Error generating summary."

if __name__ == '__main__':
    # Example usage:
    # This requires a GOOGLE_API_KEY to be set in a .env file
    if not API_KEY:
        print("Please set your GOOGLE_API_KEY in a .env file to run this example.")
    else:
        # Dummy candidate and job description for testing
        dummy_candidate = {
            "source": "resume_python_dev.txt",
            "score": 0.85,
            "content": "John Doe | Python Developer | 5 years of experience in Django, Flask, and REST APIs. "
                       "Led a project on machine learning for sentiment analysis. "
                       "B.S. in Computer Science."
        }
        dummy_jd = "Seeking a Python Developer with experience in web frameworks and machine learning."

        # Generate the summary
        ai_summary = generate_summary(dummy_candidate, dummy_jd)

        # Print the result
        print(f"--- AI Summary for {dummy_candidate['source']} ---")
        print(ai_summary)
