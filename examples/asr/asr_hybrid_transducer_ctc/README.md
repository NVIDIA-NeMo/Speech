# ASR with Hybrid Transducer/CTC Models

This directory contains example scripts to train ASR models with two decoders of Transducer and CTC Loss. 

Currently supported models are - 

* Character based Hybrid RNNT/CTC model
* Subword based Hybrid RNNT/CTC model

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