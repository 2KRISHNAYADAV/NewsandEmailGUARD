#!/usr/bin/env python3
"""
Setup script for NewsGuardian.AI
This script helps users set up the environment and install dependencies.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    nltk_script = """
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", nltk_script], 
                              capture_output=True, text=True, check=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = ['data', 'models', 'app', 'notebooks']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"ℹ️  Directory already exists: {directory}")
    
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    
    required_files = ['True.csv', 'Fake.csv']
    required_dirs = ['spam', 'easy_ham']
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_files.append(directory)
        else:
            print(f"✅ Found: {directory}")
    
    if missing_files:
        print(f"\n⚠️  Missing data files/directories: {', '.join(missing_files)}")
        print("Please download the required datasets:")
        print("1. Fake News Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
        print("2. SpamAssassin Dataset: https://spamassassin.apache.org/old/publiccorpus/")
        return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    packages = [
        'pandas', 'numpy', 'sklearn', 'nltk', 'streamlit', 
        'joblib', 'matplotlib', 'seaborn', 'wordcloud', 'plotly'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 NewsGuardian.AI Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        print("\n⚠️  Warning: Could not download NLTK data")
        print("You may need to download it manually later")
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed: Package import test failed")
        return False
    
    # Check data files
    data_ok = check_data_files()
    
    print("\n" + "=" * 50)
    if data_ok:
        print("✅ Setup completed successfully!")
        print("\n🎉 You're ready to start!")
        print("\nNext steps:")
        print("1. Train the models: python train_models.py")
        print("2. Test the models: python test_models.py")
        print("3. Launch the web app: streamlit run app/app.py")
    else:
        print("⚠️  Setup completed with warnings!")
        print("\nPlease download the required datasets before proceeding.")
        print("Then run: python train_models.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 