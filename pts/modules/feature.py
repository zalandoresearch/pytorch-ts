import torch
import torch.nn as nn


class FeatureEmbedder(nn.Module):
    def __init__(
            self,
            cardinalities: List[int],
            embedding_dims: List[int],
    ) -> None:
        super().__init__()

        self.__num_features = len(cardinalities)

        def create_embedding(c: int, d: int) -> nn.Embedding:
            embedding = nn.Embedding(c, d)
            return embedding

        self.__embedders = nn.ModuleList([
            create_embedding(c, d)
            for c, d in zip(cardinalities, embedding_dims)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length
            # self.__num_features with shape (N,T) or (N)
            cat_feature_slices = torch.chunk(features,
                                             self.__num_features,
                                             dim=-1)
        else:
            cat_feature_slices = [features]

        return torch.cat([
            embed(cat_feature_slice.squeeze(-1)) for embed, cat_feature_slice
            in zip(self.__embedders, cat_feature_slices)
        ], dim=-1)

class FeatureAssembler(nn.Module):
    pass