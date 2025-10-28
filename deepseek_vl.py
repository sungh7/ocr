"""
DeepSeek Vision-Language Model for Local Inference

This module provides a wrapper for the DeepSeek VL model for table extraction
from images using local inference instead of API calls.
"""

import os
import json
import torch
from typing import Optional, Dict, Any
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekVLModel:
    """
    Wrapper for DeepSeek Vision-Language model for local inference.

    Supports multiple model sizes:
    - deepseek-ai/deepseek-vl-1.3b-chat (smaller, faster)
    - deepseek-ai/deepseek-vl-7b-chat (larger, more accurate)
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl-1.3b-chat",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the DeepSeek VL model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('auto', 'cuda', 'cpu')
            load_in_8bit: Load model in 8-bit precision (saves memory)
            load_in_4bit: Load model in 4-bit precision (saves more memory)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        logger.info(f"Initializing DeepSeek VL model: {model_name}")

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            logger.info(f"Loading model from {self.model_name}...")

            # Prepare loading kwargs
            load_kwargs = {
                "trust_remote_code": True,
            }

            # Add quantization settings if specified
            if self.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit mode")
            elif self.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
                logger.info("Loading model in 4-bit mode")
            else:
                # Use float16 for better performance on GPU
                if torch.cuda.is_available():
                    load_kwargs["torch_dtype"] = torch.float16

            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    load_kwargs["device_map"] = "auto"
                    logger.info("Using CUDA device")
                else:
                    logger.info("Using CPU device")
            else:
                load_kwargs["device_map"] = self.device

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Set to evaluation mode
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_table_from_image(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract table data from an image using the DeepSeek VL model.

        Args:
            image: PIL Image object
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary containing extracted table data
        """
        if self.model is None:
            self.load_model()

        try:
            # Prepare the prompt
            prompt = """Analyze this image and extract any tables present.

For each table found:
1. Identify the table structure (rows and columns)
2. Extract all text content from each cell
3. Preserve the table layout and relationships

Return the result in JSON format with this structure:
{
    "tables": [
        {
            "table_number": 1,
            "rows": [
                ["cell1", "cell2", "cell3"],
                ["cell4", "cell5", "cell6"]
            ],
            "headers": ["Header1", "Header2", "Header3"],
            "description": "Brief description of what this table contains"
        }
    ],
    "total_tables": 1
}

If no tables are found, return:
{
    "tables": [],
    "total_tables": 0,
    "message": "No tables detected in the image"
}"""

            # Prepare conversation format
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # Prepare inputs for the model
            pil_images = [image]
            prepare_inputs = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )

            # Move inputs to device
            if torch.cuda.is_available() and not self.load_in_8bit and not self.load_in_4bit:
                prepare_inputs = {k: v.to(self.model.device) for k, v in prepare_inputs.items()}

            # Prepare images
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs, images=pil_images)

            # Generate response
            logger.info("Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs['attention_mask'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    use_cache=True
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0].cpu().tolist(),
                skip_special_tokens=True
            )

            logger.info("Response generated successfully")

            # Extract the assistant's response
            # The response includes the full conversation, so we need to extract just the answer
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

            # Try to parse JSON from the response
            result = self._parse_json_response(response)

            return result

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {
                "tables": [],
                "total_tables": 0,
                "error": str(e),
                "message": f"Error during table extraction: {str(e)}"
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from model response.

        Args:
            response: Raw response text from model

        Returns:
            Parsed dictionary or error response
        """
        try:
            # Sometimes the model wraps JSON in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            # Try to parse JSON
            result = json.loads(response)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return raw response if JSON parsing fails
            return {
                "tables": [],
                "total_tables": 0,
                "raw_response": response,
                "message": "Unable to parse structured table data. Raw response included."
            }

    def unload_model(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")


# Singleton instance for model reuse
_model_instance: Optional[DeepSeekVLModel] = None


def get_model_instance(
    model_name: str = "deepseek-ai/deepseek-vl-1.3b-chat",
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> DeepSeekVLModel:
    """
    Get or create a singleton model instance.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        load_in_8bit: Load model in 8-bit precision
        load_in_4bit: Load model in 4-bit precision

    Returns:
        DeepSeekVLModel instance
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = DeepSeekVLModel(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        _model_instance.load_model()

    return _model_instance
