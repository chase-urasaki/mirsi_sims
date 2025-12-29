#%%
#!/usr/bin/env python3
"""
Package Installer for MIRSI Simulations
Checks for required packages and installs missing ones.
"""

import sys
import subprocess
import importlib.util

# Define required packages
# Format: (import_name, pip_install_name)
REQUIRED_PACKAGES = [
    ('numpy', 'numpy'),
    ('astropy', 'astropy'),
    ('matplotlib', 'matplotlib'),
    ('photutils', 'photutils'),
    ('scipy', 'scipy'),  # Often used with scientific Python packages
]

def check_package(import_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            package_name
        ])
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False

def main():
    """Check and install all required packages."""
    print("=" * 60)
    print("MIRSI Simulations - Package Dependency Checker")
    print("=" * 60)
    print()
    
    missing_packages = []
    installed_packages = []
    
    # Check which packages are missing
    print("Checking for required packages...")
    for import_name, pip_name in REQUIRED_PACKAGES:
        if check_package(import_name):
            print(f"✓ {import_name} is already installed")
            installed_packages.append(import_name)
        else:
            print(f"✗ {import_name} is NOT installed")
            missing_packages.append((import_name, pip_name))
    
    print()
    
    # Install missing packages
    if missing_packages:
        print(f"Found {len(missing_packages)} missing package(s).")
        response = input("Do you want to install them now? [Y/n]: ").strip().lower()
        
        if response in ('', 'y', 'yes'):
            print()
            print("Installing missing packages...")
            print("-" * 60)
            
            failed = []
            for import_name, pip_name in missing_packages:
                if not install_package(pip_name):
                    failed.append(pip_name)
            
            print("-" * 60)
            print()
            
            if failed:
                print(f"✗ Failed to install {len(failed)} package(s): {', '.join(failed)}")
                print("Please try installing them manually:")
                for pkg in failed:
                    print(f"  pip install {pkg}")
                return 1
            else:
                print(f"✓ All packages installed successfully!")
        else:
            print("Installation cancelled. Please install manually:")
            for _, pip_name in missing_packages:
                print(f"  pip install {pip_name}")
            return 1
    else:
        print("✓ All required packages are already installed!")
    
    print()
    print("=" * 60)
    print("Setup complete! You can now run the MIRSI simulations.")
    print("=" * 60)
    return 0

#%%
if __name__ == "__main__":
    sys.exit(main())

#%%