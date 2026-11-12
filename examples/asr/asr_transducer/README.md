# ASR with Transducer Models

This directory contains example scripts to train ASR models using Transducer Loss (often termed RNNT Loss). 

Currently supported models are - 

* Character based RNNT model
* Subword based RNNT model
* Experimental [enrollment-conditioned target-speaker RNNT](README_target_speaker_asr.md)

# Model execution overview

The training scripts in this directory execute in the following order. When preparing your own training-from-scratch / fine-tuning scripts, please follow this order for correct training/inference.

```mermaid

graph TD
    A[Hydra Overrides + Yaml Config] --> B{Config}
    B --> |Init| C[Trainer]
    C --> D[ExpManager]
    B --> D[ExpManager]
    C --> E[Model]
    B --> |Init| E[Model]
    E --> |Constructor| F1(Change Vocabulary)
    F1 --> F2(Setup Adapters if available)
    F2 --> G(Setup Train + Validation + Test Data loaders)
    G --> H1(Setup Optimization)
    H1 --> H2(Change Transducer Decoding Strategy)
    H2 --> I[Maybe init from pretrained]
    I --> J["trainer.fit(model)"]
```

During restoration of the model, you may pass the Trainer to the restore_from / from_pretrained call, or set it after the model has been initialized by using `model.set_trainer(Trainer)`.
