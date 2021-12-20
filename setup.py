from setuptools import find_packages, setup

install_requires = []


setup_requires = ["pytest-runner"]

tests_require = ["pytest", "pytest-cov", "mock", "unittest"]

keywords = [
    "bla",
]


setup(
    name="chemicalx",
    packages=find_packages(),
    version="0.0.2",
    license="Apache License, Version 2.0",
    description="A Deep Learning Library for Drug Pair Scoring.",
    author="Benedek Rozemberczki and Charles Hoyt",
    author_email="",
    url="https://github.com/AstraZeneca/chemicalx",
    download_url="https://github.com/AstraZeneca/chemicalx/archive/v_00002.tar.gz",
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
