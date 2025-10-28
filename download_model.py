#!/usr/bin/env python3
"""
Download and cache vision-language models for offline use.

This script downloads models from HuggingFace and caches them locally.
Run this script before deploying the application to ensure models are available.

Usage:
    python download_model.py [--model-type MODEL_TYPE] [--cache-dir CACHE_DIR]

Examples:
    # Download default model (deepseek-vl-1.3b)
    python download_model.py

    # Download MiniCPM-o 2.6
    python download_model.py --model-type minicpm-o-2.6

    # Download larger DeepSeek model
    python download_model.py --model-type deepseek-vl-7b

    # Specify custom cache directory
    python download_model.py --cache-dir /path/to/models

    # Download all models
    python download_model.py --all
"""

import argparse
import os
import sys
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch

# Import model factory for supported models
from model_factory import ModelFactory


def download_model(model_type: str, cache_dir: str = None):
    """
    Download and cache the model and tokenizer.

    Args:
        model_type: Model type identifier (e.g., 'deepseek-vl-1.3b', 'minicpm-o-2.6')
        cache_dir: Optional custom cache directory
    """
    # Get model configuration
    if model_type not in ModelFactory.SUPPORTED_MODELS:
        available = ", ".join(ModelFactory.SUPPORTED_MODELS.keys())
        print(f"✗ Error: Unknown model type '{model_type}'")
        print(f"Available models: {available}")
        sys.exit(1)

    model_config = ModelFactory.SUPPORTED_MODELS[model_type]
    model_name = model_config["hf_name"]

    print("=" * 70)
    print(f"Downloading Model: {model_type}")
    print("=" * 70)
    print(f"HuggingFace Name: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"VRAM Required: {model_config['vram']}")
    print()

    # Special handling for PaddleOCR
    if model_type == "paddleocr":
        print("PaddleOCR models are automatically downloaded on first use.")
        print("No pre-download necessary!")
        print()
        print("=" * 70)
        print("✓ PaddleOCR is ready to use!")
        print("=" * 70)
        print()
        print("You can now run the application with:")
        print("  python app_local.py")
        print()
        print("To use this model, set in .env:")
        print(f"  MODEL_TYPE={model_type}")
        print()
        print("Note: Models will be downloaded automatically when you first")
        print("      process an image. This may take a few minutes.")
        print()
        return

    print("This may take several minutes depending on your internet connection...")
    print()

    try:
        # Set cache directory if specified
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            print(f"Using cache directory: {cache_dir}")

        # Check available device
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ CUDA not available. Model will be downloaded for CPU use.")

        print()

        # Download tokenizer
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
        print()

        # Download model
        print("[2/2] Downloading model...")
        print("Note: This is a large file and may take significant time.")

        # Determine which AutoModel class to use
        if "deepseek" in model_type.lower():
            ModelClass = AutoModelForCausalLM
        else:
            ModelClass = AutoModel

        model = ModelClass.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded successfully")
        print()

        # Get model info
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print("=" * 70)
        print("Model Information:")
        print("=" * 70)
        print(f"  Parameters: {param_count:.2f}B")
        print(f"  Model Type: {model_type}")
        print(f"  HuggingFace Name: {model_name}")
        if cache_dir:
            print(f"  Cache Location: {cache_dir}")
        else:
            from transformers.utils import TRANSFORMERS_CACHE
            print(f"  Cache Location: {TRANSFORMERS_CACHE}")

        print()
        print("=" * 70)
        print("✓ Model download completed successfully!")
        print("=" * 70)
        print()
        print("You can now run the application with:")
        print("  python app_local.py")
        print()
        print("To use this model, set in .env:")
        print(f"  MODEL_TYPE={model_type}")
        print()

    except Exception as e:
        print()
        print("=" * 70)
        print("✗ Error downloading model")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have enough disk space (models are several GB)")
        print("  3. Try running with sudo if you have permission issues")
        print("  4. Check if the model name is correct")
        print()
        sys.exit(1)


def download_all_models(cache_dir: str = None):
    """
    Download all supported models.

    Args:
        cache_dir: Optional custom cache directory
    """
    models = list(ModelFactory.SUPPORTED_MODELS.keys())
    total = len(models)

    print("=" * 70)
    print(f"Downloading ALL models ({total} total)")
    print("=" * 70)
    print()

    for i, model_type in enumerate(models, 1):
        print(f"\n[{i}/{total}] Starting download for: {model_type}")
        print("-" * 70)
        try:
            download_model(model_type, cache_dir)
        except SystemExit:
            print(f"✗ Failed to download {model_type}, continuing with next model...")
            continue

    print()
    print("=" * 70)
    print("✓ All models downloaded!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download vision-language models for local inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Models:
{chr(10).join(f"  - {key}: {value['description']} (VRAM: {value['vram']})"
              for key, value in ModelFactory.SUPPORTED_MODELS.items())}

Examples:
  python download_model.py
  python download_model.py --model-type minicpm-o-2.6
  python download_model.py --model-type deepseek-vl-7b
  python download_model.py --cache-dir /data/models
  python download_model.py --all
        """
    )

    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="deepseek-vl-1.3b",
        help="Model type identifier (default: deepseek-vl-1.3b)"
    )

    parser.add_argument(
        "--cache-dir",
        "-c",
        type=str,
        default=None,
        help="Custom cache directory for model storage"
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all supported models"
    )

    args = parser.parse_args()

    print()

    if args.all:
        download_all_models(args.cache_dir)
    else:
        download_model(args.model_type, args.cache_dir)


if __name__ == "__main__":
    main()
