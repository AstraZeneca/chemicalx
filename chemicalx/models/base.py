"""Base classes for models and utilities."""

__all__ = [
    "UnimplementedModel",
]


class UnimplementedModel:
    """The base class for ChemicalX models."""

    def __init__(self, x: int):
        """Instantiate a base model."""
        self.x = x
