"""Setup the package."""

from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "torch>=1.10.0",
    "torchdrug",
    "torch-scatter>=2.0.8",
    "pandas<=1.3.5",
    "tqdm",
    "scikit-learn",
    "class-resolver>=0.2.1",
    "tabulate",
    "pystow",
    "pytdc",
    "more-itertools",
]


setup_requires = ["pytest-runner"]

tests_require = ["pytest", "pytest-cov"]

extras_require = {
    "tests": tests_require,
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-click",
        "sphinx-autodoc-typehints",
        "sphinx_automodapi",
        "nbsphinx_link",
        "jupyter-sphinx",
    ],
}

keywords = [
    "drug",
    "deep-learning",
    "deep-chemistry",
    "deep-ai",
    "torch-drug",
    "synergy-prediction",
    "synergy",
    "drug-combination",
    "deep-synergy",
    "drug-interaction",
    "chemistry",
    "pharma",
]


setup(
    name="chemicalx",
    packages=find_packages(),
    version="0.1.0",
    license="Apache License, Version 2.0",
    description="A Deep Learning Library for Drug Pair Scoring.",
    author="Benedek Rozemberczki and Charles Hoyt",
    author_email="benedek.rozemberczki@gmail.com",
    url="https://github.com/AstraZeneca/chemicalx",
    download_url="https://github.com/AstraZeneca/chemicalx/archive/v0.1.0.tar.gz",
    keywords=keywords,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
