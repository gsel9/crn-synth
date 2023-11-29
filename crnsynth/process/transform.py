import pandas as pd


def hybridize_data(data_synth, data_real, target_col, target_val):
    """Transfer a part of the real data over to the syntehtic data.

    Args:
        data_real: The real dataset.
        data_synth: The synthetic dataset.
        target_col: Name for the column in real the data to transfer.
        target_val: Value used to mask the rows to transfer.

    Returns:
        The combined data.
    """

    partial_real = data_real[data_real[target_col] == target_val]
    return pd.concat([data_synth, partial_real], axis=0, ignore_index=True)


def concat_data(datasets):
    concated = pd.concat(datasets, axis=0, ignore_index=True)

    # sanity checks
    assert concated.shape[0] == sum([data.shape[0] for data in datasets])
    assert all([concated.shape[1] == data.shape[1] for data in datasets])

    return concated
