#!/usr/bin/env python3
"""
Simple launcher for emotion detection
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main
from main import main

if __name__ == "__main__":
    main()