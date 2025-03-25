import pkg_resources
import subprocess
import sys

REQUIREMENTS = ['scikit-learn==1.0.2', 'pandas==1.3.5']

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for requirement in REQUIREMENTS:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        install(requirement)
