#!/bin/bash
set -e

pip install pre-commit 
pre-commit install 

pip install -e .
