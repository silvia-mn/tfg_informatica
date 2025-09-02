from copy import copy
from functools import partial

import numpy as np
import pandas as pd


def partitioned_series_to_sequences(
    df: pd.DataFrame, seq_len: int, features: list[str], order_key: str
):
    result = []
    buf = []
    for i, row in df.reset_index().sort_values(by=order_key).iterrows():
        if i < seq_len:
            buf.append(list(row[features]))
            continue
        result.append(copy(buf))
        buf.pop(0)
        buf.append(list(row[features]))
    if not result:
        return None
    return np.array(result)


def create_seqs(
    df: pd.DataFrame,
    partition_key: str,
    order_key: str,
    features: list[str],
    seq_len: int,
) -> np.array:
    return np.concat(
        list(
            df.groupby(partition_key)
            .apply(
                partial(
                    partitioned_series_to_sequences,
                    seq_len=seq_len,
                    features=features,
                    order_key=order_key,
                ),
                include_groups=False,
            )
            .dropna()
        )
    )


def format_data(
    df: pd.DataFrame,
    partition_key,
    order_key,
    encoder_input_features,
    decoder_input_features,
    output_features,
    input_seq_length,
    output_seq_length,
) -> tuple[tuple[np.array, np.array, np.array], list[int]]:
    features = list(
        set(encoder_input_features + decoder_input_features + output_features)
    )
    feature_index_map = {feature: i for i, feature in enumerate(features)}
    encoder_input_features_idx = [
        feature_index_map[feature] for feature in encoder_input_features
    ]
    decoder_input_features_idx = [
        feature_index_map[feature] for feature in decoder_input_features
    ]
    output_features_idx = [feature_index_map[feature] for feature in output_features]
    seqs = create_seqs(
        df,
        partition_key,
        order_key,
        features=features,
        seq_len=input_seq_length + output_seq_length,
    )
    encoder_inputs = seqs[:, :input_seq_length, encoder_input_features_idx]
    decoder_inputs = seqs[
        :,
        input_seq_length - 1 : input_seq_length + output_seq_length - 1,
        decoder_input_features_idx,
    ]
    decoder_outputs = seqs[
        :, input_seq_length : input_seq_length + output_seq_length, output_features_idx
    ]
    feature_index_map = {feature: i for i, feature in enumerate(decoder_input_features)}
    return (encoder_inputs, decoder_inputs, decoder_outputs), [
        feature_index_map[feature] for feature in output_features
    ]

