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
    def __init__(T: int,
                 use_static_cat: bool = False,
                 use_static_real: bool = False,
                 use_dynamic_cat: bool = False,
                 use_dynamic_real: bool = False,
                 embed_static: Optional[FeatureEmbedder] = None,
                 embed_dynamic: Optional[FeatureEmbedder] = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()

        self.T = T
        self.use_static_cat = use_static_cat
        self.use_static_real = use_static_real
        self.use_dynamic_cat = use_dynamic_cat
        self.use_dynamic_real = use_dynamic_real

        self.embed_static: Callable[[torch.Tensor], torch.
                                    Tensor] = embed_static or (lambda x: x)
        self.embed_dynamic: Callable[[torch.Tensor], torch.
                                     Tensor] = embed_dynamic or (lambda x: x)

    def forward(
            self,
            feat_static_cat: torch.Tensor,
            feat_static_real: torch.Tensor,
            feat_dynamic_cat: torch.Tensor,
            feat_dynamic_real: torch.Tensor,
    ) -> torch.Tensor:
        processed_features = [
            self.process_static_cat(feat_static_cat),
            self.process_static_real(feat_static_real),
            self.process_dynamic_cat(feat_dynamic_cat),
            self.process_dynamic_real(feat_dynamic_real),
        ]

        return torch.cat(processed_features, dim=-1)

    def process_static_cat(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.embed_static(feature.to(self.dtype))
        return feature.unsqueeze(1).expand(-1, self.T, -1)

    def process_dynamic_cat(self, feature: torch.Tensor) -> torch.Tensor:
        return self.embed_dynamic(feature.to(self.dtype))

    def process_static_real(self, feature: torch.Tensor) -> torch.Tensor:
        return feature.unsqueeze(1).expand(-1, self.T, -1)

    def process_dynamic_real(self, feature: torch.Tensor) -> torch.Tensor:
        return feature
