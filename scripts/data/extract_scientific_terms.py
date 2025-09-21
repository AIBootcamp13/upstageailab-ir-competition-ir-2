#!/usr/bin/env python3
"""
Scientific Terms Extraction using Local Ollama Models

Processes keywords_per_src.json to extract Korean scientific terms
using locally available Ollama models (qwen2:7b or llama3.1:8b).

Usage:
    uv run python scripts/data/extract_scientific_terms.py --input_file outputs/reports/data_profile/latest/keywords_per_src.json --model qwen2:7b
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Set
import fire


def query_ollama(prompt: str, model: str = "qwen2:7b", max_retries: int = 3) -> str:
    """Query local Ollama model with retry logic."""
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent extraction
            "top_p": 0.9,
            "num_predict": 1000
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    return ""


def create_extraction_prompt(keywords_chunk: List[Dict[str, Any]], chunk_id: int) -> str:
    """Create a prompt for scientific term extraction."""

    # Convert keywords to simple list format for the prompt
    terms_text = []
    for src, terms in keywords_chunk:
        if isinstance(terms, list):
            for term_info in terms[:10]:  # Limit to top 10 per source
                if isinstance(term_info, dict):
                    term = term_info.get("term", "")
                    if term:
                        terms_text.append(term)

    terms_sample = "\n".join(terms_text[:50])  # Limit prompt size

    prompt = f"""당신은 한국어 과학 용어 전문가입니다. 다음 키워드 목록에서 **과학적 용어만** 추출해 주세요.

포함할 용어:
- 물리학: 원자, 분자, 에너지, 힘, 운동, 전자, 양성자, 파동 등
- 화학: 화합물, 반응, 원소, 산, 염기, 이온, 결합 등
- 생물학: 세포, 유전자, 단백질, DNA, RNA, 진화, 대사 등
- 의학: 질병, 치료, 진단, 약물, 해부학 용어 등
- 천문학: 별, 행성, 은하, 우주, 블랙홀 등
- 지질학: 암석, 광물, 지층, 화산, 지진 등
- 수학: 방정식, 함수, 적분, 확률, 통계 등

제외할 용어:
- 일반 명사 (사람, 시간, 장소 등)
- 동사 (하다, 되다, 있다 등)
- 형용사 (좋은, 나쁜, 큰 등)
- 조사/어미 (의, 을, 를 등)
- 일반적인 단어

키워드 목록 (청크 {chunk_id}):
{terms_sample}

과학 용어만 한 줄에 하나씩 나열해 주세요:"""

    return prompt


def extract_terms_from_response(response: str) -> List[str]:
    """Extract scientific terms from model response."""
    lines = response.split('\n')
    terms = []

    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, or lines that look like explanations
        if not line or line.isdigit() or '포함' in line or '제외' in line:
            continue

        # Remove common prefixes/bullets
        line = line.lstrip('- ').lstrip('• ').lstrip('* ')

        # Basic filtering
        if len(line) > 1 and len(line) < 20:  # Reasonable term length
            terms.append(line)

    return terms


def chunk_keywords(keywords_data: Dict[str, List[Dict]], chunk_size: int = 5) -> List[List]:
    """Split keywords into manageable chunks for processing."""
    items = list(keywords_data.items())
    chunks = []

    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


def extract_scientific_terms(
    input_file: str = "outputs/reports/data_profile/latest/keywords_per_src.json",
    output_file: str = "outputs/reports/data_profile/latest/scientific_terms_extracted.json",
    model: str = "qwen2:7b",
    chunk_size: int = 5,
) -> Dict[str, Any]:
    """Extract scientific terms from keywords using local Ollama model."""

    print(f"🔬 Extracting scientific terms using {model}")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")

    # Load keywords data
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {input_file}")

    with open(input_path, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)

    print(f"📊 Loaded keywords from {len(keywords_data)} sources")

    # Process in chunks
    chunks = chunk_keywords(keywords_data, chunk_size)
    all_scientific_terms = set()
    chunk_results = []

    print(f"🔄 Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        print(f"⏳ Processing chunk {i+1}/{len(chunks)}...")

        try:
            prompt = create_extraction_prompt(chunk, i+1)
            response = query_ollama(prompt, model)

            if response:
                terms = extract_terms_from_response(response)
                chunk_results.append({
                    "chunk_id": i+1,
                    "sources": [src for src, _ in chunk],
                    "extracted_terms": terms,
                    "term_count": len(terms)
                })
                all_scientific_terms.update(terms)
                print(f"✅ Extracted {len(terms)} terms from chunk {i+1}")
            else:
                print(f"⚠️ No response for chunk {i+1}")

        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            continue

        # Small delay to be nice to Ollama
        time.sleep(1)

    # Prepare results
    results = {
        "model_used": model,
        "total_chunks": len(chunks),
        "total_scientific_terms": len(all_scientific_terms),
        "scientific_terms": sorted(list(all_scientific_terms)),
        "chunk_results": chunk_results,
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Extraction complete!")
    print(f"📊 Total scientific terms extracted: {len(all_scientific_terms)}")
    print(f"💾 Results saved to: {output_file}")

    # Preview some terms
    if all_scientific_terms:
        preview = sorted(list(all_scientific_terms))[:20]
        print(f"🔍 Sample terms: {', '.join(preview)}")

    return results


if __name__ == "__main__":
    fire.Fire(extract_scientific_terms)