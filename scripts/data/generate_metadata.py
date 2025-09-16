#!/usr/bin/env python3
"""
Generate metadata (summary, keywords, hypothetical questions) for documents using OpenAI API.
Uses gpt-4o-mini model for Korean text generation with concurrent requests.
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        logger.info(f"Using OpenAI model: {model_name}")

    async def generate_metadata_batch(self, contents: List[str]) -> List[Dict[str, Any]]:
        """Generate metadata for a batch of documents concurrently."""
        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)

        async def generate_single(content: str) -> Dict[str, Any]:
            return await self.generate_metadata_async(client, content)

        tasks = [generate_single(content) for content in contents]
        results = await asyncio.gather(*tasks)
        return results

    async def generate_metadata_async(self, client, content: str) -> Dict[str, Any]:
        """Generate summary, keywords, and hypothetical questions for a document."""
        prompt = f"""다음 과학 문서를 분석하여 메타데이터를 생성하시오.

문서 내용:
{content}

다음 형식으로 응답하시오:

요약: [문서의 핵심 내용을 2-3문장으로 요약]

키워드: [쉼표로 구분된 주요 키워드와 엔티티 목록]

가상 질문: [이 문서에 답할 수 있는 3-5개의 가상 질문 목록, 각 줄에 하나씩]
"""

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7
            )
            response_text = response.choices[0].message.content
            if response_text is None:
                raise ValueError("No response content received")

            # Parse the response
            metadata = self._parse_response(response_text)
            return metadata

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured metadata."""
        lines = response.split('\n')
        summary = ""
        keywords = []
        hypothetical_questions = []

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('요약:'):
                current_section = 'summary'
                summary = line[3:].strip()
            elif line.startswith('키워드:'):
                current_section = 'keywords'
                keywords_text = line[4:].strip()
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            elif line.startswith('가상 질문:'):
                current_section = 'questions'
                continue
            elif current_section == 'summary' and not line.startswith(('키워드:', '가상 질문:')):
                summary += ' ' + line
            elif current_section == 'questions':
                if line and not line.startswith('요약:') and not line.startswith('키워드:'):
                    hypothetical_questions.append(line)

        return {
            'summary': summary,
            'keywords': keywords,
            'hypothetical_questions': hypothetical_questions
        }

async def process_documents(input_file: str, output_file: str, batch_size: int = 10, start_idx: int = 0, limit: Optional[int] = None):
    """Process documents in batches and generate metadata."""
    generator = MetadataGenerator()

    input_path = Path(input_file)
    output_path = Path(output_file)

    # Read all documents
    documents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    total_docs = len(documents)
    if limit is not None:
        total_docs = min(total_docs, start_idx + limit)
    logger.info(f"Loaded {total_docs} documents (limited to {limit} if specified)")

    # Check if output file exists and get processed count
    processed_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    processed_ids.add(doc['docid'])
        logger.info(f"Found {len(processed_ids)} already processed documents")

    # Process documents in batches starting from start_idx
    with open(output_path, 'a', encoding='utf-8') as f:
        for i in range(start_idx, total_docs, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_contents = []
            batch_indices = []

            for j, doc in enumerate(batch_docs):
                if doc['docid'] in processed_ids:
                    continue
                batch_contents.append(doc['content'])
                batch_indices.append((i + j, doc))

            if not batch_contents:
                continue

            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_contents)} documents")

            try:
                metadata_list = await generator.generate_metadata_batch(batch_contents)

                for (idx, doc), metadata in zip(batch_indices, metadata_list):
                    enhanced_doc = {
                        **doc,
                        'summary': metadata['summary'],
                        'keywords': metadata['keywords'],
                        'hypothetical_questions': metadata['hypothetical_questions']
                    }
                    f.write(json.dumps(enhanced_doc, ensure_ascii=False) + '\n')
                    f.flush()
                    logger.info(f"Processed document {doc['docid']}")

            except Exception as e:
                logger.error(f"Error processing batch starting at {i}: {e}")
                # Write original documents if batch fails
                for idx, doc in batch_indices:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    f.flush()

    logger.info("Metadata generation completed")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    asyncio.run(process_documents(args.input_file, args.output_file, args.batch_size, args.start_idx, args.limit))