def update_config_from_kwargs(
    run_config, base_kwargs=None, data_kwargs=None, generator_kwargs=None
):
    """Modify the config key: value pairs in place based on optional keyword
    arguments.

    Args:
        run_config:
        base_kwargs:
        data_kwargs:
        generator_kwargs:

    """

    if base_kwargs is not None:
        _update_config(run_config, "base", base_kwargs)

    if data_kwargs is not None:
        _update_config(run_config, "data", data_kwargs)

    if generator_kwargs is not None:
        _update_config(run_config, "generator", generator_kwargs)


# TODO: handle nested dict updates by recursive func call
def _update_config(config, mode_key, kwargs):
    if not kwargs:
        return

    for value_key, value in kwargs.items():
        # nested values
        if isinstance(value, dict):
            for subject_key, subject_value in value.items():
                config[mode_key][value_key][subject_key] = subject_value

        else:
            config[mode_key][value_key] = value
