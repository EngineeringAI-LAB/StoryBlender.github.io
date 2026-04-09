#!/usr/bin/env python3
"""
Script to download Python wheels for all packages in requirements_blender.txt
and update blender_manifest.toml with the wheel paths.
"""

import subprocess
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm

# Platforms to download for all packages (ensures compatibility)
PLATFORMS = [
    'macosx_11_0_arm64',
    'manylinux_2_28_x86_64',
    'win_amd64'
]

PYTHON_VERSION = '3.11'


def parse_requirements(requirements_file):
    """Parse the requirements_blender.txt file and extract package names."""
    packages = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse conda list format: package_name version build channel
            parts = line.split()
            if len(parts) >= 2:
                package_name = parts[0]
                # Skip conda packages (not available on PyPI or built-in)
                skip_packages = {
                    'appnope', 'asttokens', 'bzip2', 'ca-certificates',
                    'comm', 'debugpy', 'decorator', 'executing', 'expat',
                    'ipykernel', 'ipython', 'jedi', 'jupyter_client', 'jupyter_core',
                    'krb5', 'libcxx', 'libedit', 'libffi', 'libsodium',
                    'matplotlib-inline', 'ncurses', 'nest-asyncio', 'openssl',
                    'parso', 'pexpect', 'pickleshare', 'pip', 'platformdirs',
                    'prompt-toolkit', 'psutil', 'ptyprocess', 'pure_eval',
                    'pygments', 'pyzmq', 'readline', 'setuptools', 'sqlite',
                    'stack_data', 'tk', 'tornado', 'traitlets', 'wheel',
                    'xz', 'zeromq', 'zlib', 'bpy'  # bpy is Blender-specific
                }
                
                if package_name not in skip_packages:
                    # Convert package name to PyPI format
                    pypi_name = package_name.replace('_', '-')
                    packages.append(pypi_name)
    
    return sorted(set(packages))


def check_package_exists(package, wheels_dir):
    """Check if wheel files for a package already exist."""
    # Normalize package name (replace - with _ and vice versa for matching)
    package_patterns = [
        f"{package.replace('-', '_')}-*.whl",
        f"{package.replace('_', '-')}-*.whl",
    ]
    
    for pattern in package_patterns:
        if list(Path(wheels_dir).glob(pattern)):
            return True
    return False


def download_package_wheels(package, wheels_dir, pbar=None):
    """
    Download wheels for a package across all platforms.
    This ensures we have both pure Python and platform-specific wheels as needed.
    """
    if pbar:
        pbar.set_description(f"Checking {package}")
    
    # Check if wheels already exist
    if check_package_exists(package, wheels_dir):
        if pbar:
            pbar.write(f"  ✓ {package} wheels already exist, skipping")
            pbar.update(1)
        return True
    
    if pbar:
        pbar.set_description(f"Downloading {package}")
    
    success = False
    
    # Try downloading for each platform
    # If it's pure Python, pip will download the same wheel multiple times (we'll dedupe later)
    # If it's platform-specific, we'll get the correct wheels for each platform
    for platform in PLATFORMS:
        cmd = [
            'pip', 'download', package,
            '--dest', str(wheels_dir),
            '--only-binary=:all:',
            f'--python-version={PYTHON_VERSION}',
            f'--platform={platform}'
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            success = True
        except subprocess.CalledProcessError as e:
            # If binary download fails for a platform, try pure Python wheel
            if platform == PLATFORMS[0]:  # Only try once
                if pbar:
                    pbar.write(f"  Trying pure Python wheel for {package}...")
                cmd_pure = ['pip', 'wheel', package, '-w', str(wheels_dir), '--no-deps']
                try:
                    subprocess.run(cmd_pure, check=True, capture_output=True)
                    success = True
                    break
                except subprocess.CalledProcessError:
                    if pbar:
                        pbar.write(f"  Warning: Could not download {package}")
    
    if pbar:
        pbar.update(1)
    
    return success


def get_wheel_files(wheels_dir):
    """Get all wheel files in the wheels directory."""
    wheels = list(Path(wheels_dir).glob('*.whl'))
    return sorted([f'./wheels/{w.name}' for w in wheels])


def update_manifest(manifest_file, wheel_paths):
    """Update the blender_manifest.toml file with the wheel paths."""
    print(f"\nUpdating {manifest_file}...")
    
    with open(manifest_file, 'r') as f:
        content = f.read()
    
    # Find the wheels section and replace it
    # Match the entire wheels array
    wheels_pattern = r'wheels\s*=\s*\[[\s\S]*?\]'
    
    # Format the wheels list
    wheels_str = 'wheels = [\n'
    for wheel in wheel_paths:
        wheels_str += f'  "{wheel}",\n'
    wheels_str += ']'
    
    # Replace the wheels section
    new_content = re.sub(wheels_pattern, wheels_str, content)
    
    with open(manifest_file, 'w') as f:
        f.write(new_content)
    
    print(f"Updated manifest with {len(wheel_paths)} wheels.")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    requirements_file = script_dir / 'requirements_blender.txt'
    wheels_dir = project_root / 'wheels'
    manifest_file = project_root / 'blender_manifest.toml'
    
    # Create wheels directory if it doesn't exist
    wheels_dir.mkdir(exist_ok=True)
    
    # Parse requirements
    print("Parsing requirements...")
    packages = parse_requirements(requirements_file)
    print(f"Found {len(packages)} packages to download.\n")
    
    # Download wheels for each package with progress bar
    with tqdm(total=len(packages), desc="Downloading packages", unit="pkg") as pbar:
        for package in packages:
            download_package_wheels(package, wheels_dir, pbar)
    
    # Download Windows-only conditional dependencies that pip may miss
    # when running on macOS/Linux (they are platform-specific extras).
    win_only_deps = ['win32-setctime', 'colorama']
    for dep in win_only_deps:
        if not check_package_exists(dep, wheels_dir):
            print(f"Downloading Windows-only dependency: {dep}")
            try:
                subprocess.run(
                    ['pip', 'download', dep, '--dest', str(wheels_dir),
                     '--only-binary=:all:', '--no-deps'],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                print(f"  Warning: Could not download {dep}")

    # Get all wheel files
    wheel_paths = get_wheel_files(wheels_dir)
    print(f"\nTotal wheels downloaded: {len(wheel_paths)}")
    
    # Update manifest
    update_manifest(manifest_file, wheel_paths)
    
    print("\nDone!")


if __name__ == '__main__':
    main()