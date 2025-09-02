import matplotlib.pyplot as plt
import torch


def plot(
    index,
    model,
    val_data,
    max_month: int,
    colors=["#61bbb6", "#c3f2f0", "#ad56cd", "#4a3b85"],
):
    # Example
    encoder_input = val_data[0][index].squeeze().tolist()
    decoder_input = val_data[1][index].squeeze().tolist()
    real_decoder_output = val_data[2][index].squeeze().tolist()
    computed_decoder_output = (
        model(
            torch.Tensor(encoder_input).unsqueeze(0),
            torch.Tensor(decoder_input).unsqueeze(0),
        )
        .squeeze()
        .tolist()
    )  # Without teacher forcing only covariables are used from decoder input

    real = [*encoder_input, *real_decoder_output]
    predicted = [*encoder_input, *computed_decoder_output]

    x = [0, 6, 12, 18, 24, 30]
    first_month = float(encoder_input[0][0])
    first_month_int = int(first_month * max_month)
    x = [xx + first_month_int for xx in x]
    real_ys = [[real_point[f] for real_point in real] for f in range(4)]
    predicted_ys = [
        [predicted_point[f] for predicted_point in predicted] for f in range(4)
    ]
    for real_y, predicted_y, color, i in zip(
        real_ys, predicted_ys, colors, range(1, 5)
    ):
        plt.plot(x, real_y, color=color, label=f"Real updrs_{i}")
        plt.plot(x, predicted_y, "-.", color=color, label=f"Predicted updrs_{i}")
    plt.legend()
    plt.gcf().set_size_inches((18, 6))
