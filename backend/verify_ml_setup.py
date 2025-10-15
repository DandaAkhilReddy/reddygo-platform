"""
ML/AI Setup Verification Script
Run this before starting Day 1 to ensure all dependencies are working
"""

import sys
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"✗ Python {version.major}.{version.minor} (Need 3.11+)"


def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation"""
    try:
        import torch
        import torchvision
        cuda_available = torch.cuda.is_available()
        device = "CUDA" if cuda_available else "CPU"
        return True, f"✓ PyTorch {torch.__version__} ({device})"
    except ImportError as e:
        return False, f"✗ PyTorch not installed: {str(e)}"


def check_tensorflow() -> Tuple[bool, str]:
    """Check TensorFlow installation"""
    try:
        import tensorflow as tf
        gpus = len(tf.config.list_physical_devices('GPU'))
        device = f"{gpus} GPU(s)" if gpus > 0 else "CPU"
        return True, f"✓ TensorFlow {tf.__version__} ({device})"
    except ImportError as e:
        return False, f"✗ TensorFlow not installed: {str(e)}"


def check_langchain() -> Tuple[bool, str]:
    """Check LangChain installation"""
    try:
        import langchain
        from langchain_community import __version__ as lc_community_version
        from langchain_openai import __version__ as lc_openai_version
        return True, f"✓ LangChain {langchain.__version__}"
    except ImportError as e:
        return False, f"✗ LangChain not installed: {str(e)}"


def check_transformers() -> Tuple[bool, str]:
    """Check Hugging Face Transformers"""
    try:
        import transformers
        return True, f"✓ Transformers {transformers.__version__}"
    except ImportError as e:
        return False, f"✗ Transformers not installed: {str(e)}"


def check_sentence_transformers() -> Tuple[bool, str]:
    """Check Sentence Transformers"""
    try:
        import sentence_transformers
        return True, f"✓ Sentence Transformers {sentence_transformers.__version__}"
    except ImportError as e:
        return False, f"✗ Sentence Transformers not installed: {str(e)}"


def check_qdrant() -> Tuple[bool, str]:
    """Check Qdrant client"""
    try:
        from qdrant_client import QdrantClient
        # Try connecting to local Qdrant (won't work until Docker is running)
        try:
            client = QdrantClient(host="localhost", port=6333, timeout=2)
            collections = client.get_collections()
            return True, f"✓ Qdrant client installed + server running ({len(collections.collections)} collections)"
        except Exception:
            return True, f"✓ Qdrant client installed (server not running yet)"
    except ImportError as e:
        return False, f"✗ Qdrant client not installed: {str(e)}"


def check_ollama() -> Tuple[bool, str]:
    """Check Ollama installation"""
    try:
        import ollama
        # Try connecting to Ollama server
        try:
            client = ollama.Client()
            models = client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            llama_installed = any('llama' in name.lower() for name in model_names)
            if llama_installed:
                return True, f"✓ Ollama installed + LLaMA downloaded ({len(model_names)} models)"
            return True, f"✓ Ollama installed (LLaMA not downloaded yet)"
        except Exception:
            return True, f"✓ Ollama client installed (server not running yet)"
    except ImportError as e:
        return False, f"✗ Ollama not installed: {str(e)}"


def check_scikit_learn() -> Tuple[bool, str]:
    """Check scikit-learn"""
    try:
        import sklearn
        return True, f"✓ scikit-learn {sklearn.__version__}"
    except ImportError as e:
        return False, f"✗ scikit-learn not installed: {str(e)}"


def check_pandas() -> Tuple[bool, str]:
    """Check pandas"""
    try:
        import pandas as pd
        return True, f"✓ pandas {pd.__version__}"
    except ImportError as e:
        return False, f"✗ pandas not installed: {str(e)}"


def check_numpy() -> Tuple[bool, str]:
    """Check NumPy"""
    try:
        import numpy as np
        return True, f"✓ NumPy {np.__version__}"
    except ImportError as e:
        return False, f"✗ NumPy not installed: {str(e)}"


def check_mlflow() -> Tuple[bool, str]:
    """Check MLflow"""
    try:
        import mlflow
        return True, f"✓ MLflow {mlflow.__version__}"
    except ImportError as e:
        return False, f"✗ MLflow not installed: {str(e)}"


def check_openai() -> Tuple[bool, str]:
    """Check OpenAI SDK"""
    try:
        import openai
        return True, f"✓ OpenAI SDK {openai.__version__}"
    except ImportError as e:
        return False, f"✗ OpenAI SDK not installed: {str(e)}"


def check_mem0() -> Tuple[bool, str]:
    """Check Mem0 AI"""
    try:
        import mem0
        return True, f"✓ Mem0 AI {mem0.__version__}"
    except ImportError as e:
        return False, f"✗ Mem0 AI not installed: {str(e)}"


def main():
    """Run all verification checks"""
    print("=" * 70)
    print("ReddyGo ML/AI Setup Verification")
    print("=" * 70)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("TensorFlow", check_tensorflow),
        ("LangChain", check_langchain),
        ("Transformers", check_transformers),
        ("Sentence Transformers", check_sentence_transformers),
        ("Qdrant Client", check_qdrant),
        ("Ollama", check_ollama),
        ("scikit-learn", check_scikit_learn),
        ("pandas", check_pandas),
        ("NumPy", check_numpy),
        ("MLflow", check_mlflow),
        ("OpenAI SDK", check_openai),
        ("Mem0 AI", check_mem0),
    ]

    results: List[Tuple[str, bool, str]] = []

    for name, check_func in checks:
        try:
            success, message = check_func()
            results.append((name, success, message))
            print(f"{message}")
        except Exception as e:
            results.append((name, False, f"✗ {name}: Unexpected error: {str(e)}"))
            print(f"✗ {name}: Unexpected error: {str(e)}")

    print()
    print("=" * 70)

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"Summary: {passed}/{total} checks passed")
    print()

    # Action items
    failed_checks = [name for name, success, _ in results if not success]

    if failed_checks:
        print("🔧 Action Required:")
        print()
        if any("PyTorch" in name or "TensorFlow" in name or "LangChain" in name for name in failed_checks):
            print("   1. Install Python dependencies:")
            print("      cd backend")
            print("      pip install -r requirements.txt")
            print()

        if "Qdrant Client" in failed_checks:
            print("   2. Start Qdrant vector database:")
            print("      docker run -d -p 6333:6333 qdrant/qdrant")
            print()

        if "Ollama" in failed_checks:
            print("   3. Install Ollama:")
            print("      Visit: https://ollama.ai/download")
            print("      Then run: ollama pull llama3.1:8b")
            print()
    else:
        print("✅ All checks passed! You're ready to start Day 1.")
        print()
        print("📚 Next Steps:")
        print("   1. Read ML_AI_LEARNING_GUIDE.md")
        print("   2. Start Day 1: Build Transformer from Scratch")
        print("   3. Follow backend/ml/custom_llm/ code")

    print("=" * 70)

    return 0 if not failed_checks else 1


if __name__ == "__main__":
    sys.exit(main())
