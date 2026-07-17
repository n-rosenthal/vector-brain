"""
PCAProjector.py

Implementação baseada em sklearn.decomposition.PCA.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .BaseProjector import BaseProjector


class PCAProjector(BaseProjector):
    """
    Projetor utilizando Principal Component Analysis.
    """

    def __init__(
        self,
        dimensions: int = 2,
        whiten: bool = False,
        random_state: int | None = 42,
    ):
        super().__init__(
            dimensions=dimensions,
            random_state=random_state,
        )

        self.whiten = whiten

    def fit_transform(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:

        self.validate_embeddings(embeddings)

        projector = PCA(
            n_components=self.dimensions,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        projection = projector.fit_transform(
            embeddings
        )

        return projection.astype(np.float32)

    def explained_variance(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna a variância explicada por cada componente.
        """

        self.validate_embeddings(embeddings)

        pca = PCA(
            n_components=self.dimensions,
            whiten=self.whiten,
            random_state=self.random_state,
        )

        pca.fit(embeddings)

        return pca.explained_variance_ratio_

    def metadata(self):

        meta = super().metadata()

        meta.update(
            {
                "whiten": self.whiten,
            }
        )

        return meta