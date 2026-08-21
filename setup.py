from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="ambrs",  # package name
    version="0.1.0",
    author="AMBRS Project",
    description="Aerosol Model Benchmarking Repository and Standards",
    packages=find_packages(),  # automatically finds the ambrs/ folder
    install_requires=requirements,
    python_requires=">=3.11",  # adjust if needed
)
