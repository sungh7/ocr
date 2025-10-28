#!/usr/bin/env python3
"""
Download and cache DeepSeek VL model for offline use.

This script downloads the model from HuggingFace and caches it locally.
Run this script before deploying the application to ensure the model is available.

Usage:
    python download_model.py [--model MODEL_NAME] [--cache-dir CACHE_DIR]

Examples:
    # Download default model (1.3B)
    python download_model.py

    # Download larger model (7B)
    python download_model.py --model deepseek-ai/deepseek-vl-7b-chat

    # Specify custom cache directory
    python download_model.py --cache-dir /path/to/models
"""

import argparse
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def download_model(model_name: str, cache_dir: str = None):
    """
    Download and cache the model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        cache_dir: Optional custom cache directory
    """
    print(f"Downloading DeepSeek VL model: {model_name}")
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
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA not available. Model will be downloaded for CPU use.")

        print()

        # Download tokenizer
        print("1/2 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
        print()

        # Download model
        print("2/2 Downloading model...")
        print("Note: This is a large file and may take significant time.")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        print("✓ Model downloaded successfully")
        print()

        # Get model info
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Model information:")
        print(f"  - Parameters: {param_count:.2f}B")
        print(f"  - Model name: {model_name}")
        if cache_dir:
            print(f"  - Cache location: {cache_dir}")
        else:
            from transformers.utils import TRANSFORMERS_CACHE
            print(f"  - Cache location: {TRANSFORMERS_CACHE}")

        print()
        print("=" * 60)
        print("✓ Model download completed successfully!")
        print("=" * 60)
        print()
        print("You can now run the application with:")
        print("  python app.py")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ Error downloading model")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have enough disk space (models are several GB)")
        print("  3. Try running with sudo if you have permission issues")
        print("  4. Check if the model name is correct")
        print()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepSeek VL model for local inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  - deepseek-ai/deepseek-vl-1.3b-chat  (smaller, faster, ~3GB)
  - deepseek-ai/deepseek-vl-7b-chat   (larger, more accurate, ~15GB)

Examples:
  python download_model.py
  python download_model.py --model deepseek-ai/deepseek-vl-7b-chat
  python download_model.py --cache-dir /data/models
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-vl-1.3b-chat",
        help="HuggingFace model identifier (default: deepseek-ai/deepseek-vl-1.3b-chat)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for model storage"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("DeepSeek VL Model Downloader")
    print("=" * 60)
    print()

    download_model(args.model, args.cache_dir)


if __name__ == "__main__":
    main()
