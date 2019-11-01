import torch
import torch.nn as nn

from pts.modules import DistributionOutput


class DeepARNetwork(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_cells: int,
            cell_type: str,
            history_length: int,
            context_length: int,
            prediction_length: int,
            distr_output: DistributionOutput,
            dropout_rate: float,
            cardinality: List[int],
            embedding_dimension: List[int],
            lags_seq: List[int],
            scaling: bool = True,
            dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype

        self.lags_seq = lags_seq

        self.distr_output = distr_output
        self.rnn = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU}[
            self.cell_type
        ]

        # TODO
        # self.target_shape = distr_output.event_shape

        self.proj_distr_args = distr_output.get_args_proj()




class DeepARTrainingNetwork(DeepARNetwork):
    pass