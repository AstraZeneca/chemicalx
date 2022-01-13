from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "torch>=1.10.0",
    "torchdrug",
    "torch-scatter>=2.0.8",
    "pandas<=1.3.5",
    "scipy",
    "tqdm",
    "six",
    "scikit-learn",
    "class-resolver",
]


setup_requires = ["pytest-runner"]

tests_require = ["pytest", "pytest-cov", "mock", "unittest"]

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
    version="0.0.7",
    license="Apache License, Version 2.0",
    description="A Deep Learning Library for Drug Pair Scoring.",
    author="Benedek Rozemberczki and Charles Hoyt",
    author_email="",
    url="https://github.com/AstraZeneca/chemicalx",
    download_url="https://github.com/AstraZeneca/chemicalx/archive/v0.0.7.tar.gz",
    keywords=keywords,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
)
