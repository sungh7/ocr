"""
Model Factory for selecting and initializing different OCR models.

This module provides a unified interface for working with multiple
vision-language models for table extraction.
"""

import logging
from typing import Optional, Dict, Any
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating and managing different OCR models.

    Supported models:
    - DeepSeek VL (deepseek-vl-1.3b, deepseek-vl-7b)
    - MiniCPM-o 2.6
    """

    SUPPORTED_MODELS = {
        "deepseek-vl-1.3b": {
            "module": "deepseek_vl",
            "class": "DeepSeekVLModel",
            "hf_name": "deepseek-ai/deepseek-vl-1.3b-chat",
            "description": "DeepSeek VL 1.3B - Fast and efficient",
            "vram": "4GB"
        },
        "deepseek-vl-7b": {
            "module": "deepseek_vl",
            "class": "DeepSeekVLModel",
            "hf_name": "deepseek-ai/deepseek-vl-7b-chat",
            "description": "DeepSeek VL 7B - More accurate",
            "vram": "14GB"
        },
        "minicpm-o-2.6": {
            "module": "minicpm_model",
            "class": "MiniCPMOModel",
            "hf_name": "openbmb/MiniCPM-o-2_6",
            "description": "MiniCPM-o 2.6 - Efficient multimodal model",
            "vram": "8GB"
        },
        "paddleocr": {
            "module": "paddleocr_model",
            "class": "PaddleOCRModel",
            "hf_name": "PaddlePaddle/PaddleOCR",
            "description": "PaddleOCR - Specialized table recognition",
            "vram": "2GB"
        }
    }

    @staticmethod
    def create_model(
        model_type: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Create and initialize a model instance.

        Args:
            model_type: Model type identifier (e.g., 'deepseek-vl-1.3b', 'minicpm-o-2.6')
            device: Device to load model on ('auto', 'cuda', 'cpu')
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in ModelFactory.SUPPORTED_MODELS:
            available = ", ".join(ModelFactory.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available models: {available}"
            )

        model_config = ModelFactory.SUPPORTED_MODELS[model_type]
        module_name = model_config["module"]
        class_name = model_config["class"]
        hf_name = model_config["hf_name"]

        logger.info(f"Creating model: {model_type} ({model_config['description']})")

        # Dynamically import the module
        if module_name == "deepseek_vl":
            from deepseek_vl import get_model_instance
            return get_model_instance(
                model_name=hf_name,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
        elif module_name == "minicpm_model":
            from minicpm_model import get_model_instance
            return get_model_instance(
                model_name=hf_name,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
        elif module_name == "paddleocr_model":
            from paddleocr_model import get_model_instance
            # PaddleOCR uses GPU flag instead of device string
            use_gpu = (device in ["auto", "cuda"])
            return get_model_instance(
                use_gpu=use_gpu,
                lang="en"
            )
        else:
            raise ValueError(f"Unknown module: {module_name}")

    @staticmethod
    def list_models() -> Dict[str, Dict[str, str]]:
        """
        List all supported models with their descriptions.

        Returns:
            Dictionary of model information
        """
        return ModelFactory.SUPPORTED_MODELS

    @staticmethod
    def get_hf_name(model_type: str) -> str:
        """
        Get HuggingFace model name for a given model type.

        Args:
            model_type: Model type identifier

        Returns:
            HuggingFace model name
        """
        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        return ModelFactory.SUPPORTED_MODELS[model_type]["hf_name"]


class UnifiedModelWrapper:
    """
    Unified wrapper providing a consistent interface across different models.
    """

    def __init__(
        self,
        model_type: str = "deepseek-vl-1.3b",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize unified model wrapper.

        Args:
            model_type: Type of model to use
            device: Device to load model on
            load_in_8bit: Load in 8-bit precision
            load_in_4bit: Load in 4-bit precision
        """
        self.model_type = model_type
        self.model = ModelFactory.create_model(
            model_type=model_type,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

    def extract_table_from_image(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract table from image using the loaded model.

        Args:
            image: PIL Image object
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with extracted table data
        """
        return self.model.extract_table_from_image(
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def unload_model(self):
        """Unload the model from memory."""
        if hasattr(self.model, 'unload_model'):
            self.model.unload_model()


# Global singleton instance
_unified_model: Optional[UnifiedModelWrapper] = None


def get_unified_model(
    model_type: str = "deepseek-vl-1.3b",
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> UnifiedModelWrapper:
    """
    Get or create a unified model instance.

    Args:
        model_type: Type of model to use
        device: Device to load model on
        load_in_8bit: Load in 8-bit precision
        load_in_4bit: Load in 4-bit precision

    Returns:
        UnifiedModelWrapper instance
    """
    global _unified_model

    if _unified_model is None:
        _unified_model = UnifiedModelWrapper(
            model_type=model_type,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

    return _unified_model
