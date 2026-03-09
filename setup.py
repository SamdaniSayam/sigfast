from setuptools import setup, find_packages

setup(
    name="turbosignal-sayam",
    version="0.1.0",
    author="Golam Samdani Sayam",
    author_email="golamsamdani301416@gmail.com",
    description="A blazing fast, Numba-JIT compiled time-series processor.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamdaniSayam/turbosignal",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.55.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)