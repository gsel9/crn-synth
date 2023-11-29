"""Uility functions for generator handling."""


def set_generator_attrs(synth_alg, generator_kwargs, verbose=1):
    """Update the attributes of a generator algorithm.

    Args:
        synth_alg: An instance of a generator algorithm.
        generator_kwargs: key-value pairs of generator attributes.
        verbose: Verbosity level.

    """
    obj = synth_alg
    # some generators objects are wrappers
    if hasattr(obj, "model"):
        obj = synth_alg.model

    for key, value in vars(obj).items():
        if key in generator_kwargs.keys():
            if verbose > 0:
                print(f"{key}: old value: {value}; new value: {generator_kwargs[key]}")

            setattr(obj, key, generator_kwargs[key])
