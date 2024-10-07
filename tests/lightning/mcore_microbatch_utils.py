import contextlib


# @akoumparouli: use a context manager that saves/restores gbs/mbs when using
# reconfigure_num_microbatches_calculator to avoid interference between tests.
@contextlib.contextmanager
def reconfigure_num_microbatches_calculator_manager(*args, **kwargs):
    import megatron.core.num_microbatches_calculator as mb_calc

    # Store current mbs, gbs values
    if not mb_calc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        _mbs = mb_calc.get_micro_batch_size()
        _gbs = mb_calc.get_current_global_batch_size()

        # use user's settings
        mb_calc.reconfigure_num_microbatches_calculator(*args, **kwargs)
    else:
        _mbs, _gbs = 1, 1

    try:
        # run user's code
        yield
        # @akoumparouli: no catch
    finally:
        # restore old mbs, gbs
        if not mb_calc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
            mb_calc.reconfigure_num_microbatches_calculator(0, None, _gbs, _mbs, data_parallel_size=1)
