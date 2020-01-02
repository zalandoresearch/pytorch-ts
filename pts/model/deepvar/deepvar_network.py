from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Distribution

import numpy as np

from pts.modules import DistributionOutput, MeanScaler, NOPScaler, FeatureEmbedder


class DeepVARTrainingNetwork(nn.Module):
    pass


class DeepVARPredictionNetwork(DeepVARTrainingNetwork):
    pass
