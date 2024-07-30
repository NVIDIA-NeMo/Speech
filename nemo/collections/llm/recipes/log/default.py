from nemo import lightning as nl
from nemo.collections.llm.utils import factory


@factory
def default_log() -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        save_on_train_epoch_end=True,
    )

    return nl.NeMoLogger(ckpt=ckpt)
