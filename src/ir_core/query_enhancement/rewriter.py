# src/ir_core/query_enhancement/rewriter.py

from typing import Optional
import openai
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type


class QueryRewriter:
    """
    Query Rewriting and Expansion using OpenAI.

    This class enhances queries by expanding them with relevant synonyms,
    related terms, and making them more specific for better document retrieval.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        openai_client: Optional[openai.OpenAI] = None,  # For backward compatibility
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the Query Rewriter.

        Args:
            llm_client: Pre-configured LLM client. If None, creates one based on model_name.
            openai_client: Pre-configured OpenAI client (for backward compatibility).
            model_name: OpenAI model to use. Defaults to settings value.
            max_tokens: Maximum tokens for response. Defaults to settings value.
            temperature: Temperature for generation. Defaults to settings value.
        """
        # Handle backward compatibility
        if llm_client is None:
            if openai_client:
                from .llm_client import OpenAIClient
                self.llm_client = OpenAIClient(openai_client)
            else:
                # Auto-detect based on model name
                if model_name is None:
                    model_name = getattr(settings, 'query_enhancement', {}).get('openai_model', 'gpt-3.5-turbo')
                # Ensure model_name is not None for type checker
                assert model_name is not None
                client_type = detect_client_type(model_name)
                self.llm_client = create_llm_client(client_type, model_name=model_name)
        else:
            self.llm_client = llm_client

        self.model_name = model_name or getattr(settings, 'query_enhancement', {}).get('openai_model', 'gpt-3.5-turbo')
        self.max_tokens = max_tokens or getattr(settings, 'query_enhancement', {}).get('max_tokens', 500)
        self.temperature = temperature or getattr(settings, 'query_enhancement', {}).get('temperature', 0.3)

        # Load rewriter-specific configuration
        rewriter_config = getattr(settings, 'query_enhancement', {}).get('rewriter', {})
        self.korean_char_start = rewriter_config.get('korean_char_range_start', '\uac00')
        self.korean_char_end = rewriter_config.get('korean_char_range_end', '\ud7a3')
        self.default_expansion_factor = rewriter_config.get('default_expansion_factor', 2)
        self.min_rewrite_length_ratio = rewriter_config.get('min_rewrite_length_ratio', 0.5)
        self.expansion_levels = rewriter_config.get('expansion_levels', {
            1: "Add 2-3 closely related terms",
            2: "Add 4-6 related terms and synonyms",
            3: "Add comprehensive related terms, synonyms, and technical variations"
        })

        # Load template paths
        self.korean_template_path = rewriter_config.get('korean_template', 'prompts/rewriter/rewriter_korean_v1.jinja2')
        self.english_template_path = rewriter_config.get('english_template', 'prompts/rewriter/rewriter_english_v1.jinja2')
        self.expansion_template_path = rewriter_config.get('expansion_template', 'prompts/rewriter/rewriter_expansion_v1.jinja2')

        # Load templates
        self._load_templates()

    def _load_templates(self):
        """Load Jinja2 templates for rewriting and expansion."""
        import jinja2
        from pathlib import Path

        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent

        # Initialize template loader
        template_dir = project_root / "prompts"
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Load templates
        try:
            self.korean_template = self.template_env.get_template(self.korean_template_path.replace('prompts/', ''))
            self.english_template = self.template_env.get_template(self.english_template_path.replace('prompts/', ''))
            self.expansion_template = self.template_env.get_template(self.expansion_template_path.replace('prompts/', ''))
        except jinja2.TemplateNotFound as e:
            print(f"Warning: Template not found: {e}")
            # Fallback to basic templates
            self.korean_template = None
            self.english_template = None
            self.expansion_template = None

    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite and expand the query for better retrieval.

        Args:
            original_query: The original user query

        Returns:
            Enhanced query string optimized for retrieval
        """
        # Detect if the query is in Korean
        is_korean = any(self.korean_char_start <= char <= self.korean_char_end for char in original_query)

        if is_korean:
            if self.korean_template:
                prompt = self.korean_template.render(query=original_query)
            else:
                prompt = f"""
                이 쿼리를 검색에 최적화된 형태로 재작성하세요.

                원본 쿼리: {original_query}

                지침:
                - 핵심 개념과 주요 용어 유지
                - 관련 동의어와 기술 용어 추가 가능
                - 더 자연스럽고 구체적인 표현 사용
                - 의미 100% 유지
                - 한국어로 출력

                재작성된 쿼리만 한 줄로 출력하세요. 설명이나 추가 텍스트 없이 쿼리만 제공하세요.
                """
        else:
            if self.english_template:
                prompt = self.english_template.render(query=original_query)
            else:
                prompt = f"""
                Rewrite this query in an optimized form for search.

                Original query: {original_query}

                Guidelines:
                - Keep core concepts and key terms
                - Add relevant synonyms and technical terms if helpful
                - Use more natural and specific phrasing
                - Preserve 100% of original meaning

                Output only the rewritten query on one line. No explanations or additional text.
                """

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not response['success']:
                print(f"Query rewriting failed: {response.get('error', 'Unknown error')}")
                return original_query

            content = response.get('content')
            if not content:
                return original_query

            rewritten_query = content.strip()

            # Fallback if response is empty or too short
            if not rewritten_query or len(rewritten_query) < len(original_query) * self.min_rewrite_length_ratio:
                return original_query

            return rewritten_query

        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return original_query  # Return original query on failure

    def expand_query(self, query: str, expansion_factor: int = 2) -> str:
        """
        Expand query with additional related terms.

        Args:
            query: Original query
            expansion_factor: How much to expand (1-3, where 3 is most expansive)

        Returns:
            Expanded query with additional terms
        """
        # Use configured default if expansion_factor is not provided
        if expansion_factor is None:
            expansion_factor = self.default_expansion_factor

        level_desc = self.expansion_levels.get(expansion_factor, self.expansion_levels.get(self.default_expansion_factor, "Add 4-6 related terms and synonyms"))

        if self.expansion_template:
            prompt = self.expansion_template.render(
                query=query,
                expansion_description=level_desc
            )
        else:
            prompt = f"""
            Expand this query by adding relevant terms for better search results:

            Original query: {query}

            {level_desc} to improve document retrieval.
            Maintain the original query's intent and meaning.

            Provide only the expanded query, no explanation.
            """

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not response['success']:
                print(f"Query expansion failed: {response.get('error', 'Unknown error')}")
                return query

            content = response.get('content')
            if not content:
                return query

            expanded_query = content.strip()

            if not expanded_query or len(expanded_query) < len(query):
                return query

            return expanded_query

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query