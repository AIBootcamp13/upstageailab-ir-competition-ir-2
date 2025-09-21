#!/usr/bin/env python3
"""
Test script to demonstrate confidence scores in enhancement results and rich logging toggle.

Usage:
    # Rich logging (default)
    python test_enhancement_display.py

    # Simple LM-friendly logging
    RAG_SIMPLE_LOGGING=1 python test_enhancement_display.py
"""

import os

def test_enhancement_display():
    """Test the enhancement result display with confidence scores."""

    # Mock enhancement result
    enhancement_result = {
        'enhanced_query': 'Who is Dmitri Ivanovsky and what did he discover?',
        'technique_used': 'rewriting',
        'confidence': 0.85
    }

    query = "Dmitri Ivanovsky가 누구야?"

    print("Testing Enhancement Result Display")
    print("=" * 50)

    # Check the environment variable toggle
    USE_RICH_LOGGING = os.getenv('RAG_SIMPLE_LOGGING', '0') != '1'

    print(f"Rich logging enabled: {USE_RICH_LOGGING}")
    print(f"RAG_SIMPLE_LOGGING env var: {os.getenv('RAG_SIMPLE_LOGGING', 'not set')}")
    print()

    # Simulate the enhancement result display
    if USE_RICH_LOGGING:
        # Rich display (simplified for demo)
        print("╭────── Original Query ──────╮")
        print(f"│ {query} │")
        print("╰────────────────────────────╯")

        print("╭──────────── Enhancement Result ────────────╮")
        print(f"│ Enhanced Query: {enhancement_result['enhanced_query']} │")
        print(f"│ Technique Used: {enhancement_result['technique_used']} │")

        # Add confidence score with color coding
        confidence = enhancement_result.get('confidence')
        if confidence is not None:
            if confidence >= 0.8:
                confidence_display = f"[GREEN]{confidence:.2f}[/GREEN]"
            elif confidence >= 0.5:
                confidence_display = f"[YELLOW]{confidence:.2f}[/YELLOW]"
            else:
                confidence_display = f"[RED]{confidence:.2f}[/RED]"
            print(f"│ Confidence Score: {confidence_display} │")

        print("╰────────────────────────────────────────────╯")
    else:
        # Simple LM-friendly logging
        print(f"Original Query: {query}")
        print(f"Enhanced Query: {enhancement_result['enhanced_query']}")
        print(f"Technique Used: {enhancement_result['technique_used']}")
        confidence = enhancement_result.get('confidence')
        if confidence is not None:
            print(f"Confidence Score: {confidence:.2f}")

if __name__ == "__main__":
    test_enhancement_display()