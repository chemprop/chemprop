from html.parser import HTMLParser
import os
import re
import subprocess
import sys

import rdkit
import torch

# requests is needed for this install but is not part of a typical chemprop install
try:
    import requests
except ImportError:
    print(
        "requests is not installed. Please install it with `conda install 'conda-forge::requests'`"
    )
    exit(1)

# Check if the system is Linux, MacOS or Windows
system = sys.platform
print(f"System: {system}")

if system == "linux":
    print("Currently on Linux.")
elif system == "darwin":
    print("Currently on MacOS. cuik-molmaker is currently support only on Linux.")
    exit(1)
elif system == "win32":
    print("Currently on Windows. cuik-molmaker is currently support only on Linux.")
    exit(1)
else:
    print(f"Unknown system: {system}")
    exit(1)

# Check that conda is installed
try:
    conda_version = subprocess.check_output(["conda", "--version"], text=True).strip()
    print(f"Conda is installed and accessible. Conda version: {conda_version}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Conda is not installed or not in your system's PATH.")
    exit(1)

conda_prefix = os.getenv("CONDA_PREFIX")

rdkit_so_file = os.path.join(conda_prefix, "lib", "libRDKitGraphMol.so")
print(f"RDKit shared object file: {rdkit_so_file}")
if not os.path.exists(rdkit_so_file):
    print(f"Could not find RDKit shared object file: {rdkit_so_file}")
    print(
        "This probably means that RDKit was not installed using conda. Currently, cuik-molmaker is only supported with RDKit installed using conda. Please install RDKit/Chemprop from conda and try again."
        "For installation instructions, please refer to the Chemprop documentation: https://chemprop.readthedocs.io/en/latest/installation.html#option-3-installing-from-source-using-environment-yml"
    )
    exit(1)

libtorch_so_file = os.path.join(conda_prefix, "lib", "libtorch_cpu.so")
if not os.path.exists(libtorch_so_file):
    print(f"Could not find PyTorch shared object file: {libtorch_so_file}")
    print(
        "This probably means that PyTorch was not installed using conda. Currently, cuik-molmaker is only supported with PyTorch installed using conda. Please install PyTorch/Chemprop from conda and try again."
        "For installation instructions, please refer to the Chemprop documentation: https://chemprop.readthedocs.io/en/latest/installation.html#option-3-installing-from-source-using-environment-yml"
    )
    exit(1)

# Check if rdkit_so_file was compiled with CXX11 ABI by looking for mangled symbols
try:
    # Run strings command and pipe through grep to check for cxx11 symbols
    strings_output = subprocess.check_output(["strings", rdkit_so_file], text=True)
    grep_output = subprocess.run(
        ["grep", "-i", "GLIBCXX"],
        input=strings_output,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    cxx11_check = subprocess.run(
        ["grep", "-q", "cxx11"],
        input=grep_output.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rdkit_compiled_with_cxx_abi = cxx11_check.returncode == 0

except subprocess.CalledProcessError:
    rdkit_compiled_with_cxx_abi = False

if not rdkit_compiled_with_cxx_abi:
    print("RDKit was likely compiled with _GLIBCXX_USE_CXX11_ABI=0 (old ABI)")
    print(
        "cuik-molmaker will only work with RDKit compiled with _GLIBCXX_USE_CXX11_ABI=1 (C++11 ABI)"
    )
    exit(1)
else:
    print("RDKit was likely compiled with _GLIBCXX_USE_CXX11_ABI=1 (C++11 ABI)")

# Check libtorch ABI
try:
    strings_output = subprocess.check_output(["strings", libtorch_so_file], text=True)
    grep_output = subprocess.run(
        ["grep", "-i", "GLIBCXX"],
        input=strings_output,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    cxx11_check = subprocess.run(
        ["grep", "-q", "cxx11"],
        input=grep_output.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    libtorch_compiled_with_cxx_abi = cxx11_check.returncode == 0

except subprocess.CalledProcessError:
    libtorch_compiled_with_cxx_abi = False

if not libtorch_compiled_with_cxx_abi:
    print("LibTorch was likely compiled with _GLIBCXX_USE_CXX11_ABI=0 (old ABI)")
    print(
        "cuik-molmaker will only work with LibTorch compiled with _GLIBCXX_USE_CXX11_ABI=1 (C++11 ABI)"
    )
    exit(1)
else:
    print("LibTorch was likely compiled with _GLIBCXX_USE_CXX11_ABI=1 (C++11 ABI)")


rdkit_version = rdkit.__version__
print(f"RDKit version: {rdkit_version}")

torch_version = torch.__version__
print(f"Torch version: {torch_version}")

wheel_url = f"https://pypi.nvidia.com/rdkit-{rdkit_version}_torch-{torch_version}/"
print(f"Installing cuik-molmaker from: {wheel_url}")

# Check if URL exists
response = requests.head(wheel_url)
if response.status_code != 200:
    print(
        f"URL {wheel_url} does not exist for the version of RDKit ({rdkit_version}) and PyTorch ({torch_version}) you have installed."
    )
    print("Here are your options for installing cuik-molmaker:")
    print(
        "1. Install from source. Follow instructions at https://github.com/NVIDIA-Digital-Bio/cuik-molmaker"
    )
    print("2. Reach out to cuik-molmaker developers at cuik_molmaker_dev@nvidia.com")
    print("3. Use conda to install one of these combinations of RDKit and PyTorch:")

    class LinkExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, value in attrs:
                    if name == "href":
                        self.links.append(value)

    url = "https://pypi.nvidia.com/"
    response = requests.get(url)
    response.raise_for_status()

    parser = LinkExtractor()
    parser.feed(response.text)
    pattern = re.compile(r"rdkit-(\d{4}\.\d{2}\.\d+)_torch-(\d+\.\d+\.\d+)")
    for link in parser.links:
        match = pattern.search(link)
        if match:
            rdkit_version, torch_version = match.groups()
            print(f"   - RDKit: {rdkit_version}, Torch: {torch_version}")
    exit(1)

# Install cuik-molmaker from correct wheel

try:
    subprocess.run(
        ["pip", "install", "--no-deps", "--extra-index-url", wheel_url, "cuik_molmaker==0.1"],
        check=True,
    )
except subprocess.CalledProcessError as e:
    print(f"Failed to install cuik-molmaker: {e}")
    print(
        "1. Install from source. Follow instructions at https://github.com/NVIDIA-Digital-Bio/cuik-molmaker"
    )
    print("2. Reach out to cuik-molmaker developers at cuik_molmaker_dev@nvidia.com")
    exit(1)

print("cuik-molmaker installed successfully.")
print("You can now use cuik-molmaker with Chemprop using --use-cuikmolmaker-featurization flag.")
