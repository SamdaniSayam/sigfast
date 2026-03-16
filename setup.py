from setuptools import setup, find_packages

setup(
    name="triples-sigfast",  # The official PyPI name
    version="0.3.1", # Reset back to v1.0 for the new brand
    author="TripleS Studio", # Flex the agency name instead of your personal name!
    author_email="golamsamdani301416@gmail.com",
    description="High-performance, JIT-compiled time-series and signal processing core.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TripleS-Studio/sigfast", # Use your agency GitHub!
    packages=["sigfast"], # Change this to explicitly point to the new folder
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.55.0",
        "pandas>=1.3.0"  # <-- Added Pandas!
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry", # Makes it look B2B
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.8',
)
