import math
import random
from copy import deepcopy
from time import time

import torch


def batch_generator(data, batch_size):
    encoder_inputs, decoder_inputs, decoder_targets = data
    indices = torch.randperm(encoder_inputs.shape[0])
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        yield encoder_inputs[batch_indices], decoder_inputs[
            batch_indices
        ], decoder_targets[batch_indices], None


def train(model, data, batch_size, teacher_force_prob):
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(
        data, batch_size
    ):
        output = model(batch_enc_inputs, batch_dec_inputs, teacher_force_prob)
        loss = model.optimize(output, batch_dec_targets)

        epoch_loss += loss
        num_batches += 1
    return epoch_loss / num_batches


def evaluate(model, val_data, batch_size):
    model.eval()

    epoch_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(
            val_data, batch_size
        ):
            output = model(batch_enc_inputs, batch_dec_inputs)
            loss = model.compute_loss(output, batch_dec_targets)

            epoch_loss += loss
            num_batches += 1

    return epoch_loss / num_batches


def calc_teacher_force_prob(decay, indx):
    return decay / (decay + math.exp(indx / decay))


def get_best_model(
    model, train_data, val_data, batch_size, num_epochs, decay, no_best_break=None
):
    best_val, best_model = float("inf"), None
    no_best_count = 0
    for epoch in range(num_epochs):
        start_t = time()
        teacher_force_prob = calc_teacher_force_prob(decay, epoch)
        train_loss = train(model, train_data, batch_size, teacher_force_prob)
        val_loss = evaluate(model, val_data, batch_size)

        new_best_val = False
        if val_loss < best_val:
            new_best_val = True
            best_val = val_loss
            best_model = deepcopy(model)
            no_best_count = 0
        else:
            no_best_count += 1
            if no_best_break is not None and no_best_count >= no_best_break:
                break
        print(
            f"Epoch {epoch+1} => Train loss: {train_loss:.5f},",
            f"Val: {val_loss:.5f},",
            f"Teach: {teacher_force_prob:.2f},",
            f'Took {(time() - start_t):.1f} s{"      (NEW BEST)" if new_best_val else ""}',
        )

    return best_model


def train_val_split(data, p=0.8):
    n = data[0].shape[0]
    indices = random.sample(list(range(n)), k=int(n * p))
    remaining = [i for i in range(n) if i not in indices]
    return tuple(map(lambda arr: torch.Tensor(arr[indices]), data)), tuple(
        map(lambda arr: torch.Tensor(arr[remaining]), data)
    )
