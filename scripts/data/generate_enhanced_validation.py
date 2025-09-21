# scripts/data/generate_enhanced_validation.py

import os
import sys
import json
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Any, Optional

import hydra
import openai
import jinja2
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from omegaconf import DictConfig


def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# --- Enhanced validation data generation functions ---

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(openai.RateLimitError)
)
async def generate_enhanced_validation_data(
    client: openai.AsyncOpenAI,
    eval_id: str,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> Optional[Dict[str, Any]]:
    """
    Generate enhanced validation data with ideal_context, ideal_answer, and hard_negative_context.

    Args:
        client: OpenAI async client
        eval_id: Evaluation ID
        query: The question/query
        retrieved_docs: List of retrieved documents from the retrieval system
        model: LLM model to use

    Returns:
        Enhanced validation data dict or None if generation fails
    """

    # Extract content from retrieved documents
    retrieved_contents = []
    for doc in retrieved_docs[:5]:  # Use top 5 retrieved documents
        hit = doc.get("hit", {})
        source = hit.get("_source", {})
        content = source.get("content", "")
        if content:
            retrieved_contents.append(content)

    if not retrieved_contents:
        print(f"Warning: No content found for eval_id {eval_id}")
        return None

    # Combine retrieved contents for context
    combined_context = "\n\n".join(retrieved_contents)

    # Prompt for generating ideal context and answer
    ideal_generation_prompt = f"""
당신은 RAG 시스템 평가를 위한 고품질 데이터셋을 생성하는 전문가입니다.

다음 질문과 검색된 문맥을 바탕으로, 질문에 답변하기 위해 반드시 필요한 핵심 내용을 추출하고, 그에 기반한 이상적인 답변을 생성해주세요.

질문: {query}

검색된 문맥:
{combined_context}

다음 형식으로 JSON을 생성해주세요:
{{
  "ideal_context": [
    "질문에 답변하기 위해 반드시 필요한 첫 번째 핵심 문장이나 단락",
    "질문에 답변하기 위해 반드시 필요한 두 번째 핵심 문장이나 단락"
  ],
  "ideal_answer": "ideal_context만을 근거로 생성된, 질문에 대한 가장 정확하고 완전한 답변"
}}

주의사항:
- ideal_context는 검색된 문맥에서 직접 추출한 내용이어야 합니다
- ideal_answer는 ideal_context에 포함된 정보만을 사용하여 생성해야 합니다
- ideal_answer는 질문에 대한 직접적이고 명확한 답변이어야 합니다
"""

    try:
        ideal_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ideal_generation_prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        ideal_result_text = ideal_response.choices[0].message.content
        if ideal_result_text is None:
            print(f"Warning: Empty response for ideal generation, eval_id {eval_id}")
            return None
        ideal_result_text = ideal_result_text.strip()

        # Remove markdown code block formatting if present
        if ideal_result_text.startswith("```json"):
            ideal_result_text = ideal_result_text[7:]
        if ideal_result_text.endswith("```"):
            ideal_result_text = ideal_result_text[:-3]
        ideal_result_text = ideal_result_text.strip()

        # Try to parse JSON response
        try:
            ideal_result = json.loads(ideal_result_text)
            ideal_context = ideal_result.get("ideal_context", [])
            ideal_answer = ideal_result.get("ideal_answer", "")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse ideal generation JSON for eval_id {eval_id}: {e}")
            print(f"Raw response: {ideal_result_text[:500]}...")
            return None

    except Exception as e:
        print(f"Error generating ideal context/answer for eval_id {eval_id}: {e}")
        return None

    # Generate hard negative context (similar topic but doesn't directly answer the question)
    hard_negative_prompt = f"""
당신은 RAG 시스템 평가를 위한 어려운 부정 예시를 생성하는 전문가입니다.

다음 질문과 그에 대한 이상적인 답변을 바탕으로, 주제는 비슷하지만 질문에 대한 직접적인 답변은 포함하지 않는 "함정" 문서를 생성해주세요.

질문: {query}
이상적인 답변: {ideal_answer}

다음 형식으로 JSON을 생성해주세요:
{{
  "hard_negative_context": [
    "주제는 비슷하지만 질문에 직접 답하지 않는 첫 번째 문장이나 단락",
    "주제는 비슷하지만 질문에 직접 답하지 않는 두 번째 문장이나 단락"
  ]
}}

주의사항:
- hard_negative_context는 질문의 주제와 관련이 있어야 하지만, ideal_answer에 포함된 구체적인 정보는 포함하지 말아야 합니다
- 예를 들어, 질문이 "나무 분류 방법"이라면, 나무에 대한 일반적인 설명은 가능하지만 구체적인 분류 기준은 피해야 합니다
- 리랭커가 이런 문서를 걸러내는 능력을 테스트하기 위한 것입니다
"""

    try:
        hard_negative_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": hard_negative_prompt}],
            temperature=0.3,
            max_tokens=1500,
        )

        hard_negative_result_text = hard_negative_response.choices[0].message.content
        if hard_negative_result_text is None:
            print(f"Warning: Empty response for hard negative generation, eval_id {eval_id}")
            hard_negative_context = []
        else:
            hard_negative_result_text = hard_negative_result_text.strip()

            # Remove markdown code block formatting if present
            if hard_negative_result_text.startswith("```json"):
                hard_negative_result_text = hard_negative_result_text[7:]
            if hard_negative_result_text.endswith("```"):
                hard_negative_result_text = hard_negative_result_text[:-3]
            hard_negative_result_text = hard_negative_result_text.strip()

            # Try to parse JSON response
            try:
                hard_negative_result = json.loads(hard_negative_result_text)
                hard_negative_context = hard_negative_result.get("hard_negative_context", [])
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse hard negative JSON for eval_id {eval_id}: {e}")
                print(f"Raw response: {hard_negative_result_text[:500]}...")
                hard_negative_context = []

    except Exception as e:
        print(f"Error generating hard negative context for eval_id {eval_id}: {e}")
        hard_negative_context = []

    # Return enhanced validation data
    return {
        "eval_id": eval_id,
        "query": query,
        "ideal_context": ideal_context,
        "ideal_answer": ideal_answer,
        "hard_negative_context": hard_negative_context
    }


async def process_single_question(
    client: openai.AsyncOpenAI,
    eval_item: Dict[str, Any],
    model: str = "gpt-4o-mini",
    semaphore: Optional[asyncio.Semaphore] = None,
    request_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Process a single evaluation question to generate enhanced validation data.

    Args:
        client: OpenAI async client
        eval_item: Evaluation item from eval.jsonl
        model: LLM model to use
        semaphore: Semaphore for limiting concurrent requests
        request_delay: Delay between API calls in seconds

    Returns:
        Enhanced validation data or None if processing fails
    """
    if semaphore:
        await semaphore.acquire()

    eval_id = eval_item.get("eval_id")
    try:
        messages = eval_item.get("msg", [])

        # Extract the query from the last user message
        query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        if not query:
            print(f"Warning: No user query found for eval_id {eval_id}")
            return None

        # Import retrieval function here to avoid circular imports
        from ir_core.retrieval.core import hybrid_retrieve

        # Retrieve relevant documents for the query
        retrieved_docs = hybrid_retrieve(
            query=query,
            bm25_k=50,  # Get more documents for better context
            rerank_k=20  # Return top 20 for processing
        )

        if not retrieved_docs:
            print(f"Warning: No documents retrieved for eval_id {eval_id}")
            return None

        # Add delay before API calls
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        # Generate enhanced validation data
        enhanced_data = await generate_enhanced_validation_data(
            client=client,
            eval_id=str(eval_id),
            query=query,
            retrieved_docs=retrieved_docs,
            model=model
        )

        return enhanced_data

    except Exception as e:
        print(f"Error processing eval_id {eval_id}: {e}")
        return None
    finally:
        if semaphore:
            semaphore.release()


@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point that runs the async function.
    """
    asyncio.run(run(cfg))


async def run(cfg: DictConfig) -> None:
    """
    Generate enhanced validation dataset from existing eval.jsonl questions.
    """
    _add_src_to_path()
    from ir_core.utils import read_jsonl

    # Configuration
    input_file = cfg.get("enhanced_validation", {}).get("input_file", "data/eval.jsonl")
    output_file = cfg.get("enhanced_validation", {}).get("output_file", "data/eval_enhanced.jsonl")
    model_name = cfg.get("enhanced_validation", {}).get("llm_model", "gpt-4")
    max_questions = cfg.get("enhanced_validation", {}).get("max_questions", None)  # None = process all
    max_concurrent = cfg.get("enhanced_validation", {}).get("max_concurrent", 2)
    request_delay = cfg.get("enhanced_validation", {}).get("request_delay", 1.0)

    print(f"Generating enhanced validation dataset from '{input_file}'...")
    print(f"Using model: {model_name}")
    print(f"Output file: {output_file}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Request delay: {request_delay}s")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        return

    # Load evaluation questions
    try:
        eval_items = list(read_jsonl(input_file))
        print(f"Loaded {len(eval_items)} evaluation questions.")
    except Exception as e:
        print(f"Error loading evaluation file: {e}")
        return

    # Limit number of questions if specified
    if max_questions:
        eval_items = eval_items[:max_questions]
        print(f"Processing first {max_questions} questions.")

    client = openai.AsyncOpenAI()

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    # Process questions asynchronously
    print("Processing questions and generating enhanced validation data...")

    tasks = [
        process_single_question(client, eval_item, model_name, semaphore, request_delay)
        for eval_item in eval_items
    ]

    enhanced_data_list = await tqdm_asyncio.gather(*tasks)

    # Filter out None results
    valid_results = [result for result in enhanced_data_list if result is not None]

    print(f"\nSuccessfully generated enhanced data for {len(valid_results)} questions.")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in valid_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Enhanced validation dataset saved to '{output_file}'")

    # Print summary statistics
    total_ideal_contexts = sum(len(item.get("ideal_context", [])) for item in valid_results)
    total_hard_negatives = sum(len(item.get("hard_negative_context", [])) for item in valid_results)

    print("\nSummary:")
    print(f"- Total questions processed: {len(valid_results)}")
    print(f"- Total ideal context snippets: {total_ideal_contexts}")
    print(f"- Total hard negative context snippets: {total_hard_negatives}")


if __name__ == "__main__":
    main()