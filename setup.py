from setuptools import find_packages, setup

# Read runtime dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="triples-sigfast",
    version="1.0.1",  # Bump to 1.0.1 since we're updating the pipeline!
    author="TripleS Studio",
    description="An enterprise-grade, JIT-compiled time-series engine stress-tested on 100M+ row datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamdaniSayam/triples-sigfast",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=required,  # <-- This now reads from the file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
    ],
)
