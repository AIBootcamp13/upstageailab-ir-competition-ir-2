#!/usr/bin/env python3
"""
Generate metadata (summary, keywords, hypothetical questions) for documents using local LLM.
Uses EleutherAI/polyglot-ko-12.8b with 4-bit quantization for Korean text generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataGenerator:
    def __init__(self, model_name: str = "EleutherAI/polyglot-ko-12.8b", max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("Model loaded successfully")

    def generate_metadata(self, content: str) -> Dict[str, Any]:
        """Generate summary, keywords, and hypothetical questions for a document."""
        prompt = f"""다음 과학 문서를 분석하여 메타데이터를 생성하시오.

문서 내용:
{content}

다음 형식으로 응답하시오:

요약: [문서의 핵심 내용을 2-3문장으로 요약]

키워드: [쉼표로 구분된 주요 키워드와 엔티티 목록]

가상 질문: [이 문서에 답할 수 있는 3-5개의 가상 질문 목록, 각 줄에 하나씩]
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove the prompt from response

        # Parse the response
        metadata = self._parse_response(response)
        return metadata

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

def process_documents(input_file: str, output_file: str, batch_size: int = 10, start_idx: int = 0):
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
    logger.info(f"Loaded {total_docs} documents")

    # Check if output file exists and get processed count
    processed_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    processed_ids.add(doc['docid'])
        logger.info(f"Found {len(processed_ids)} already processed documents")

    # Process documents starting from start_idx
    with open(output_path, 'a', encoding='utf-8') as f:
        for i in range(start_idx, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")

            for doc in batch:
                if doc['docid'] in processed_ids:
                    logger.info(f"Skipping already processed document {doc['docid']}")
                    continue

                try:
                    metadata = generator.generate_metadata(doc['content'])
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
                    logger.error(f"Error processing document {doc['docid']}: {e}")
                    # Write the original document if metadata generation fails
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    f.flush()

    logger.info("Metadata generation completed")

if __name__ == "__main__":
    fire.Fire(process_documents)