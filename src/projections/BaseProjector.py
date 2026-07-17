"""
BaseProjector.py

Classe base para todos os algoritmos de projeção do espaço vetorial.

Todos os projetores devem transformar uma matriz

    (N, D)

em

    (N, k)

onde normalmente k = 2 ou 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseProjector(ABC):
    """
    Interface comum para todos os algoritmos de projeção.

    Exemplos
    --------

    PCAProjector
    UMAPProjector
    TSNEProjector
    """

    def __init__(
        self,
        dimensions: int = 2,
        random_state: int | None = 42,
    ):
        if dimensions < 2:
            raise ValueError("dimensions deve ser >= 2")

        self.dimensions = dimensions
        self.random_state = random_state

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Projector", "").lower()

    @abstractmethod
    def fit_transform(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Recebe

            (N,D)

        Retorna

            (N,dimensions)
        """

    def validate_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> None:

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings deve ser numpy.ndarray")

        if embeddings.ndim != 2:
            raise ValueError(
                "Esperada matriz (N,D)."
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "Nenhum embedding encontrado."
            )

        if embeddings.shape[1] < self.dimensions:
            raise ValueError(
                "Número de dimensões do embedding menor que a projeção."
            )

    def metadata(self) -> dict[str, Any]:
        """
        Informações úteis para cache.
        """

        return {
            "algorithm": self.name,
            "dimensions": self.dimensions,
            "random_state": self.random_state,
        }

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"dimensions={self.dimensions}, "
            f"random_state={self.random_state})"
        )