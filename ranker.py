"""
This module handles the ranking of resumes based on their similarity to a job description.

It provides functions to:
- Perform a similarity search on the vector store.
- Rank the results and return a sorted list of candidates.
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rank_resumes(vector_store: FAISS, job_description: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Ranks resumes from the vector store based on similarity to the job description.

    Args:
        vector_store (FAISS): The FAISS vector store containing resume embeddings.
        job_description (str): The job description text.
        top_k (int): The number of top candidates to return.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the candidate's
                              source file, similarity score, and the matched content.
    """
    if not job_description:
        logging.warning("Job description is empty. Cannot rank resumes.")
        return []

    try:
        # Perform a similarity search with score
        results_with_scores = vector_store.similarity_search_with_score(job_description, k=top_k)

        # Process the results to create a ranked list
        ranked_candidates = []
        for doc, score in results_with_scores:
            candidate_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "score": 1 - score,  # Convert distance to similarity score
                "content": doc.page_content
            }
            ranked_candidates.append(candidate_info)

        # Sort by score in descending order
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        logging.info(f"Ranked {len(ranked_candidates)} resumes.")
        return ranked_candidates

    except Exception as e:
        logging.error(f"Error ranking resumes: {e}")
        return []

if __name__ == '__main__':
    # Example usage (requires a vector store to be created first)
    from embeddings import get_documents_from_folder, create_vector_store

    # Create a dummy resume folder and files for testing
    import os
    if not os.path.exists("data/resumes"):
        os.makedirs("data/resumes")
    with open("data/resumes/resume1.txt", "w") as f:
        f.write("Experienced Python developer with a background in machine learning and Django.")
    with open("data/resumes/resume2.txt", "w") as f:
        f.write("Data scientist skilled in NLP, PyTorch, and data visualization.")
    with open("data/resumes/resume3.txt", "w") as f:
        f.write("Java developer with 10 years of experience in enterprise applications.")

    # Create a dummy job description
    jd = "We are looking for a Python developer with machine learning skills."

    # 1. Create the vector store
    docs = get_documents_from_folder("data/resumes")
    if docs:
        vector_store = create_vector_store(docs)

        # 2. Rank the resumes
        top_candidates = rank_resumes(vector_store, jd, top_k=3)

        # 3. Print the results
        print("Top Candidates:")
        for candidate in top_candidates:
            print(f"  - Source: {candidate['source']}, Score: {candidate['score']:.2f}")
    else:
        print("No documents found to rank.")
