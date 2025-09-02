import random

import torch
from torch import nn


def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0.0)
    return layer


class Encoder(nn.Module):
    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            enc_feature_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, inputs):
        # inputs: (tamaño de lote, longitud de secuencia de entrada, número de características del codificador)
        output, hidden = self.gru(inputs)

        # output: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
        # hidden: (número de capas GRU, tamaño de lote, tamaño oculto)
        return output, hidden


# Superclase del decodificador cuyo método forward es llamado por Seq2Seq, pero otros métodos son implementados por subclases
class DecoderBase(nn.Module):
    def __init__(self, device, dec_target_size, target_indices):
        super().__init__()
        self.device = device
        self.target_indices = target_indices
        self.target_size = dec_target_size

    # Debe ejecutarse un paso a la vez, a diferencia del codificador, ya que a veces no se aplica teacher forcing
    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        raise NotImplementedError()

    def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
        # inputs: (tamaño de lote, longitud de secuencia de salida, número de características del decodificador)
        # hidden: (número de capas GRU, tamaño de lote, dimensión oculta), es decir, el último estado oculto
        # enc_outputs: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)

        batch_size, dec_output_seq_length, _ = inputs.shape

        # Almacenar las salidas del decodificador
        # outputs: (tamaño de lote, longitud de secuencia de salida, número de objetivos)
        outputs = torch.zeros(
            batch_size,
            dec_output_seq_length,
            self.target_size,
            dtype=torch.float,
        ).to(self.device)

        # curr_input: (tamaño de lote, 1, número de características del decodificador)
        curr_input = inputs[:, 0:1, :]

        for t in range(dec_output_seq_length):
            # dec_output: (tamaño de lote, 1, número de objetivos)
            # hidden: (número de capas GRU, tamaño de lote, dimensión oculta)
            # run_single_recurrent_step es un método que deben implementar las subclases
            dec_output, hidden = self.run_single_recurrent_step(
                curr_input, hidden, enc_outputs
            )
            # Guardar la predicción
            outputs[:, t : t + 1, :] = dec_output

            # Si se aplica teacher forcing, usar el objetivo de este paso como la siguiente entrada; de lo contrario, usar la predicción
            teacher_force = (
                random.random() < teacher_force_prob
                if teacher_force_prob is not None
                else False
            )

            curr_input = inputs[:, t : t + 1, :].clone()
            if not teacher_force:
                curr_input[:, :, self.target_indices] = dec_output
        return outputs


class Attention(nn.Module):
    def __init__(self, hidden_size, num_gru_layers):
        super().__init__()
        # NOTA: el tamaño oculto para la salida de attn (y entrada de v) puede ser cualquier número
        # Además, usar dos capas permite tener una función de activación no lineal entre ellas
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        # decoder_hidden_final_layer: (tamaño de lote, tamaño oculto)
        # encoder_outputs: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)

        # Repetir el estado oculto del decodificador tantas veces como la longitud de la secuencia de entrada
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(
            1, encoder_outputs.shape[1], 1
        )

        # Comparar el estado oculto del decodificador con cada salida del codificador usando una capa tanh entrenable
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Luego comprimir en valores únicos para cada comparación (energía)
        attention = self.v(energy).squeeze(2)

        # Después aplicar softmax para que los pesos sumen 1
        weightings = torch.nn.functional.softmax(attention, dim=1)

        # weightings: (tamaño de lote, longitud de secuencia de entrada)
        return weightings


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
        # GRU toma el objetivo del paso anterior y la suma ponderada de los estados ocultos del codificador (el resultado de la atención)
        self.gru = nn.GRU(
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
        # inputs: (tamaño de lote, 1, número de características del decodificador)
        # hidden: (número de capas GRU, tamaño de lote, tamaño oculto)
        # enc_outputs: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)

        # Obtener los pesos de atención
        # weightings: (tamaño de lote, longitud de secuencia de entrada)
        weightings = self.attention_model(hidden[-1], enc_outputs)

        # Luego calcular la suma ponderada
        # weighted_sum: (tamaño de lote, 1, tamaño oculto)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)

        # Después introducir en la GRU
        # entradas de GRU: (tamaño de lote, 1, número de características del decodificador + tamaño oculto)
        # output: (tamaño de lote, 1, tamaño oculto)
        output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)

        # Obtener predicción
        # entrada de out: (tamaño de lote, 1, tamaño oculto + tamaño oculto + número de objetivos)
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size)

        # output: (tamaño de lote, 1, número de objetivos)
        # hidden: (número de capas GRU, tamaño de lote, tamaño oculto)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr, grad_clip):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.opt = torch.optim.Adam(self.parameters(), lr)
        self.loss_func = Seq2Seq.compute_smape
        self.grad_clip = grad_clip

    @staticmethod
    def compute_smape(prediction, target):
        return (
            torch.mean(
                torch.abs(prediction - target)
                / ((torch.abs(target) + torch.abs(prediction)) / 2.0 + 1e-8)
            )
            * 100.0
        )

    def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
        # enc_inputs: (tamaño de lote, longitud de secuencia de entrada, número de características del codificador)
        # dec_inputs: (tamaño de lote, longitud de secuencia de salida, número de características del decodificador)

        # enc_outputs: (tamaño de lote, longitud de secuencia de entrada, tamaño oculto)
        # hidden: (número de capas GRU, tamaño de lote, dimensión oculta), es decir, el último estado oculto
        enc_outputs, hidden = self.encoder(enc_inputs)

        # outputs: (tamaño de lote, longitud de secuencia de salida, número de objetivos)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, teacher_force_prob)

        return outputs

    def compute_loss(self, prediction, target):
        # prediction: (tamaño de lote, longitud de secuencia del decodificador, número de objetivos)
        # target: (tamaño de lote, longitud de secuencia del decodificador, número de objetivos)
        loss = self.loss_func(prediction, target)
        return loss if self.training else loss.item()

    def optimize(self, prediction, target):
        # prediction & target: (tamaño de lote, longitud de secuencia, dimensión de salida)
        self.opt.zero_grad()
        loss = self.compute_loss(prediction, target)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
        return loss.item()
