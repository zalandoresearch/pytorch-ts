import pytest
from itertools import chain, combinations


import torch
import torch.nn as nn

from pts.modules import FeatureEmbedder, FeatureAssembler

@pytest.mark.parametrize(
    "config",
    (
        lambda N, T: [
            # single static feature
            dict(
                shape=(N, 1),
                kwargs=dict(cardinalities=[50], embedding_dims=[10]),
            ),
            # single dynamic feature
            dict(
                shape=(N, T, 1),
                kwargs=dict(cardinalities=[2], embedding_dims=[10]),
            ),
            # multiple static features
            dict(
                shape=(N, 4),
                kwargs=dict(
                    cardinalities=[50, 50, 50, 50],
                    embedding_dims=[10, 20, 30, 40],
                ),
            ),
            # multiple dynamic features
            dict(
                shape=(N, T, 3),
                kwargs=dict(
                    cardinalities=[30, 30, 30], embedding_dims=[10, 20, 30]
                ),
            ),
        ]
    )(10, 20),
)
def test_feature_embedder(config):
    out_shape = config["shape"][:-1] + (
        sum(config["kwargs"]["embedding_dims"]),
    )
    embed_feature = FeatureEmbedder(
        **config["kwargs"]
    )
    for embed in embed_feature._FeatureEmbedder__embedders:
        nn.init.constant_(embed.weight, 1.0)

    def test_parameters_length():
        exp_params_len = len([p for p in embed_feature.parameters()])
        act_params_len = len(config["kwargs"]["embedding_dims"])
        assert exp_params_len == act_params_len
    
    def test_forward_pass():
        act_output = embed_feature(torch.ones(config["shape"]).to(torch.long))
        exp_output = torch.ones(out_shape)
        
        assert act_output.shape == exp_output.shape
        import pdb; pdb.set_trace()
        assert torch.abs(torch.sum(act_output - exp_output)) < 1e-20

    test_parameters_length()
    test_forward_pass()
