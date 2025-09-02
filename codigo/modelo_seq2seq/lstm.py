import torch
from seq2seq import Attention, DecoderBase, layer_init
from torch import nn


class Encoder(nn.Module):
    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            enc_feature_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, inputs):
        # inputs: (tamaño de lote, longitud de secuencia de entrada, número de características del codificador)
        return self.lstm(inputs)

class DecoderWithAttention(DecoderBase):
    def __init__(
        self,
        dec_feature_size,
        dec_target_size,
        hidden_size,
        num_gru_layers,
        target_indices,
        dropout,
        device,
    ):
        super().__init__(
            device,
            dec_target_size,
            target_indices,
        )
        self.attention_model = Attention(hidden_size, num_gru_layers)
        #LSTM toma el objetivo del paso anterior y la suma ponderada de los estados ocultos del codificador (el resultado de la atención)
        self.lstm = nn.LSTM(
            dec_feature_size + hidden_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )
        # La capa de salida toma la salida del estado oculto del decodificador, la suma ponderada y la entrada del decodificador
        # NOTA: Alimentar la entrada del decodificador a la capa de salida actúa esencialmente como una conexión residual
        self.out = layer_init(
            nn.Linear(
                hidden_size + hidden_size + dec_feature_size,
                dec_target_size,
            )
        )

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        actual_hidden, cell_state = hidden
        weightings = self.attention_model(actual_hidden[-1], enc_outputs)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)
        output, (actual_hidden, cell_state) = self.lstm(torch.cat((inputs, weighted_sum), dim=2), (actual_hidden, cell_state))
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size)
        return output, (actual_hidden, cell_state)

