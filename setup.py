#!/usr/bin/env python3
"""
Personal Knowledge Assistant - Setup Script
Run this script to set up the project quickly.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("=" * 60)
    print("🧠 Personal Knowledge Assistant - Setup")
    print("=" * 60)
    print()
    
    # Get project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Step 1: Check Python version
    print("Step 1: Checking Python version...")
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ is required. You have Python {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    print()
    
    # Step 2: Install dependencies
    print("Step 2: Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Try running: pip install -r requirements.txt")
        sys.exit(1)
    print()
    
    # Step 3: Create .env file if it doesn't exist
    print("Step 3: Setting up environment...")
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            # Copy example to .env
            with open(env_example, "r") as f:
                content = f.read()
            with open(env_file, "w") as f:
                f.write(content)
            print("✅ Created .env file from .env.example")
            print()
            print("⚠️  IMPORTANT: You need to add your OpenAI API key!")
            print("   1. Open the .env file")
            print("   2. Replace 'your-openai-api-key-here' with your actual API key")
            print("   3. Get an API key at: https://platform.openai.com/api-keys")
        else:
            # Create basic .env file
            content = "OPENAI_API_KEY=your-openai-api-key-here\n"
            with open(env_file, "w") as f:
                f.write(content)
            print("✅ Created .env file")
            print()
            print("⚠️  IMPORTANT: You need to add your OpenAI API key!")
            print("   Edit .env and add your key")
    else:
        print("✅ .env file already exists")
        # Check if API key is set
        with open(env_file, "r") as f:
            content = f.read()
        if "your-openai-api-key-here" in content or "your-api-key-here" in content:
            print()
            print("⚠️  WARNING: Your API key is not set!")
            print("   Edit .env and add your OpenAI API key")
    print()
    
    # Step 4: Create storage directories
    print("Step 4: Creating storage directories...")
    (project_dir / "storage" / "uploads").mkdir(parents=True, exist_ok=True)
    (project_dir / "storage" / "vectors").mkdir(parents=True, exist_ok=True)
    print("✅ Storage directories created")
    print()
    
    # Done!
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("To start the application, run:")
    print()
    print("    python app.py")
    print()
    print("Then open your browser at: http://localhost:7860")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
