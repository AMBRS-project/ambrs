from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name="ambrs",  # package name
    version="0.1.0",
    author="AMBRS Project",
    description="Aerosol Model Benchmarking Repository and Standards",
    packages=find_packages(),  # automatically finds the ambrs/ folder
    install_requires=requirements,
    python_requires=">=3.11",  # adjust if needed
)