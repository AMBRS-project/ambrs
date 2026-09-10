from setuptools import setup, find_packages
from pathlib import Path
import re

# Read requirements from requirements.txt. Treat a # preceded by whitespace as
# the start of a comment so URL fragments such as "https://...#sha256=..." are
# preserved.
ROOT = Path(__file__).resolve().parent
requirements = [
    requirement
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if not line.lstrip().startswith("#")
    if (requirement := re.sub(r"\s+#.*$", "", line).strip())
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
