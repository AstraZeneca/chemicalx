"""Base classes for models and utilities."""

__all__ = [
    "Model",
]


class Model:
    """The base class for ChemicalX models."""

    def __init__(self, x: int):
        self.x = x
