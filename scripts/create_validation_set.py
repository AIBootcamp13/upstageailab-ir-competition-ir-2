#!/usr/bin/env python3
# scripts/create_validation_set.py
"""
Generates a synthetic validation dataset by creating questions for existing documents.

This script reads a sample of documents from the main document collection,
uses an LLM to generate a relevant question for each, and saves the resulting
(query, relevant_document_id) pairs to a new JSONL file.

This validation set is invaluable for tuning hyperparameters (like the reranking
alpha) and testing different prompt strategies without using the official
evaluation set, preventing data leakage and providing a reliable way to
measure performance improvements.
"""
import os
import sys
import json
import random
from tqdm import tqdm
import openai
import fire

# Add the src directory to the Python path to allow for absolute imports
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

# --- FIX: Updated prompt to ensure Korean language output ---
QUESTION_GENERATION_PROMPT = """
당신은 과학 지식 검색 시스템 평가를 위한 고품질 질문을 만드는 전문가입니다.
제공된 문서 내용을 바탕으로, 명확하고 관련성 높은 한국어 질문을 하나만 생성하는 것이 당신의 임무입니다.

**지침:**
- 질문은 반드시 주어진 문서의 정보만으로 답변할 수 있어야 합니다.
- 자연스러운 사람의 질문처럼 작성해야 합니다.
- **반드시 한국어로 질문을 생성해야 합니다.**
- "생성된 질문:"과 같은 군더더기 없이, 질문 자체만 출력해야 합니다.

**문서 내용:**
---
{document_content}
---

**생성된 한국어 질문:**
"""

def generate_question_for_document(client, document_content: str, model: str = None) -> str:
    """Uses an LLM to generate a single question for a given document.

    Default model prefers a cheaper/faster option; override via `model` arg
    or OPENAI_MODEL env var. Consider using an open-source local model
    (Mistral, Llama 2) served via HuggingFace/replicate if you need much lower cost.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheaper/faster alternative to gpt-3.5-turbo (verify availability/pricing)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": QUESTION_GENERATION_PROMPT.format(document_content=document_content)}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        if getattr(response, "choices", None):
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  - Error generating question with model={model}: {e}")
    return None

def run(
    input_file: str = "data/documents.jsonl",
    output_file: str = "data/validation.jsonl",
    sample_size: int = 10
):
    """
    Creates a validation dataset from a sample of documents.

    Args:
        input_file: Path to the input documents.jsonl file.
        output_file: Path where the new validation.jsonl file will be saved.
        sample_size: The number of documents to sample for question generation.
    """
    _add_src_to_path()
    from ir_core.utils import read_jsonl

    print(f"Starting validation set creation from '{input_file}'...")
    print(f"Will generate questions for {sample_size} random documents.")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this script.")
        return

    client = openai.OpenAI()

    # Read all documents into memory to sample from them
    print("Reading all documents to create a random sample...")
    all_docs = list(read_jsonl(input_file))

    if len(all_docs) < sample_size:
        print(f"Warning: Sample size ({sample_size}) is larger than the number of documents ({len(all_docs)}). Using all documents.")
        sample_docs = all_docs
    else:
        sample_docs = random.sample(all_docs, sample_size)

    validation_set = []
    print(f"Generating questions for {len(sample_docs)} documents...")
    for doc in tqdm(sample_docs, desc="Generating Questions"):
        doc_id = doc.get("docid")
        content = doc.get("content")

        if not doc_id or not content:
            continue

        question = generate_question_for_document(client, content)

        if question:
            validation_entry = {
                "eval_id": f"val_{doc_id}",
                "msg": [{"role": "user", "content": question}],
                "ground_truth_doc_id": doc_id # Store the correct answer for evaluation
            }
            validation_set.append(validation_entry)

    print(f"\nSuccessfully generated {len(validation_set)} question-document pairs.")

    # Save the new validation set to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in validation_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Validation dataset saved to '{output_file}'.")
    print("\nYou can now use this file to test and tune your RAG pipeline's retrieval performance.")

if __name__ == "__main__":
    fire.Fire(run)

# poetry run scripts/create_validation_set.py