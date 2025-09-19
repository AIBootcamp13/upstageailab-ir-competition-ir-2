# src/ir_core/generation/huggingface.py
from typing import List, Optional
import os
import torch
import jinja2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import BaseGenerator


class HuggingFaceGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        prompt_template_path: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        HuggingFace의 사전 학습된 언어 모델을 사용하여 텍스트 생성을 수행하는 생성기 클래스입니다.

        Args:
            model_name (str): HuggingFace 모델 허브에서 사용할 모델의 이름 또는 경로.
            prompt_template_path (str): 프롬프트 템플릿 파일의 경로.
            max_tokens (int): 생성할 최대 토큰 수. 기본값은 512입니다.
            temperature (float): 생성의 창의성을 제어하는 온도 매개변수. 기본값은 0.1입니다.
            device (Optional[str]): 사용할 디바이스. None이면 자동으로 GPU/CPU 선택.
        """
        self.model_name = model_name
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Set device (GPU if available)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Setup Jinja2 environment for template loading
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try quantized loading only when both bitsandbytes and accelerate are available.
        # Using device_map or torch.set_default_device requires accelerate to be installed; if
        # it's not available, fall back to standard loading to avoid noisy warnings.
        use_quantization = False
        try:
            try:
                import bitsandbytes  # type: ignore

                has_bnb = True
            except Exception:
                has_bnb = False

            try:
                import accelerate  # type: ignore

                has_accelerate = True
            except Exception:
                has_accelerate = False

            use_quantization = has_bnb and has_accelerate
        except Exception:
            use_quantization = False

        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                # Use the newer `dtype` argument instead of deprecated `torch_dtype`.
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=("auto" if self.device.type == "cuda" else None),
                    dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                )
                self.is_causal = True
            except Exception as e:
                # If anything goes wrong with quantized path, log and fallback to standard loading
                print(f"Warning: Could not load with quantization ({e}), falling back to standard loading")
                use_quantization = False

        if not use_quantization:
            try:
                # Standard loading path. Use `dtype` to avoid deprecation warning.
                tmp = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                )
                # Do not call .to() here; the pipeline will place the model on the
                # appropriate device using the `device` argument. Calling .to(...) here
                # was causing static analysis/type-checker warnings.
                self.model = tmp
                self.is_causal = True
            except Exception:
                # Fallback for models that don't support causal LM
                from transformers import AutoModelForSeq2SeqLM

                tmp = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                )
                # See note above: leave placement to the pipeline.
                self.model = tmp
                self.is_causal = False

        # Create text generation pipeline
        device_id = 0 if self.device.type == "cuda" else -1
        if self.is_causal:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
            )

        # Load default template
        self.default_template_path = prompt_template_path

    def generate(
        self,
        query: str,
        context_docs: List[str],
        prompt_template_path: Optional[str] = None,
    ) -> str:
        """
        주어진 질문과 컨텍스트를 기반으로 텍스트를 생성합니다.

        Args:
            query (str): 사용자로부터 입력받은 질문.
            context_docs (List[str]): 검색된 문서들의 리스트.
            prompt_template_path (Optional[str]): 사용할 프롬프트 템플릿 경로.

        Returns:
            str: 생성된 텍스트 응답.
        """
        # Use provided template or default one
        template_path = prompt_template_path or self.default_template_path

        try:
            template = self.jinja_env.get_template(template_path)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(
                f"'{template_path}'에서 프롬프트 템플릿을 찾을 수 없습니다. "
                f"프로젝트 루트 기준 경로가 올바른지 확인하세요."
            )

        # Prepare context
        context = "\n\n".join(context_docs)

        # Format prompt
        prompt = template.render(query=query, context=context)

        # Generate response
        if self.is_causal:
            outputs = self.generator(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Extract generated text (remove the input prompt)
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt) :].strip()
        else:
            # For seq2seq models
            outputs = self.generator(
                prompt,
                max_length=self.max_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
            )
            response = outputs[0]["generated_text"].strip()

        return response
