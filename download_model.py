#!/usr/bin/env python3
"""
Script to download and cache the embedding model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.embeddings import load_model

def main():
    print("Loading embedding model...")
    model = load_model()
    print(f"Model loaded successfully: {model}")
    print("Model is now cached and ready to use.")

if __name__ == "__main__":
    main()