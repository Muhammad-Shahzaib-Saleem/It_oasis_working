#!/usr/bin/env python3
"""
Startup script for the Email Analysis System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has GROQ_API_KEY"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your GROQ_API_KEY:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your GROQ_API_KEY")
        print("\nExample:")
        print("GROQ_API_KEY=your_actual_api_key_here")
        return False
    
    # Check if GROQ_API_KEY is set
    with open(env_file, 'r') as f:
        content = f.read()
        if 'GROQ_API_KEY=' not in content or 'your_groq_api_key_here' in content:
            print("❌ GROQ_API_KEY not properly set in .env file!")
            print("Please edit .env and add your actual GROQ API key")
            return False
    
    print("✅ Environment configuration found")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import chromadb
        import sentence_transformers
        import groq
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Email Analysis System...")
    print("📧 Access the application at: http://localhost:8501")
    print("🔗 Or use the provided runtime URL")
    print("\n" + "="*50)
    
    # Run streamlit with proper configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "12000",
        "--server.address", "0.0.0.0",
        "--server.allowRunOnSave", "true",
        "--server.enableCORS", "true",
        "--server.enableXsrfProtection", "false"
    ]
    
    subprocess.run(cmd)

def main():
    """Main startup function"""
    print("🔧 Email Analysis System - Startup Check")
    print("="*50)
    
    # Check environment
    if not check_env_file():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()