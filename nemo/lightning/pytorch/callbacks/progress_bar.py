from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n


class MegatronProgressBar(TQDMProgressBar):
    """
    Add MegatronProgressBar to remove 's/it' and display progress per step instead of per microbatch
    for megatron models.
    """

    def init_train_tqdm(self):
        """
        Override bar_format to not have 's/it'.
        """
        self.bar = super().init_train_tqdm()
        self.bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        return self.bar

    def on_train_epoch_start(self, trainer, *_):
        if trainer.max_steps > 0:  # and (trainer.ckpt_path is not None):
            # while resuming from a ckpt use trainer.max_steps as the total for progress bar as trainer.num_training_batches
            # is truncated to max_steps - step being resumed at
            num_training_batches = trainer.max_steps
        else:
            num_training_batches = trainer.num_training_batches

        self.train_progress_bar.reset(num_training_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        """
        Override parent class on_train_batch_end to update progress bar per global batch instead of per microbatch.
        """
        n = trainer.strategy.current_epoch_step
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module), refresh=False)


def calculate_data_parallel_groups() -> int:
    from nemo.utils import AppState

    app_state = AppState()

    pipeline_model_parallel_size = app_state.pipeline_model_parallel_size
    tensor_model_parallel_size = app_state.tensor_model_parallel_size

    world_size = app_state.world_size
    data_parallel_group_len = world_size // (pipeline_model_parallel_size * tensor_model_parallel_size)

    return world_size // data_parallel_group_len
