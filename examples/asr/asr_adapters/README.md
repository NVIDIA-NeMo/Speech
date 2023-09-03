# ASR Adapters support

This examples directory contains scripts to enable Adapters support for supported ASR models in NeMo.

For further discussion of what are adapters, how they are trained and how are they used, please refer to the ASR tutorials.

# Train one-or-more adapters to a pre-trained model.

Using the `train_asr_adapter.py` script, you can provide the path to a pre-trained model, a config to define and add an adapter module to this pre-trained model, some information to setup datasets for training / validation - and then easily add any number of adapter modules to this network.

**Note**: In order to train multiple adapters on a single model, provide the `model.nemo_model` (in the config) to be a previously adapted model! Ensure that you use a new unique `model.adapter.adapter_name` in the config.

## Training execution flow diagram

```mermaid

graph TD
    A[Hydra Overrides + Yaml Config] --> Bo{Config}
    Bo --> B[Update Config for Adapter Supported Modules]
    B --> |Init| C[Trainer]
    C --> D[ExpManager]
    B --> D[ExpManager]
    C --> E[Pretrained Model Restore]
    B --> |Init| E[Pretrained Model Restore]
    E --> |Constructor| F1(Change Vocabulary)
    F1 --> G(Setup Train + Validation + Test Data loaders)
    G --> H1(Setup Optimization)
    H1 --> H2(Setup Older Adapters)
    H2 --> H3[Add New Adapters]
    H3 --> H4[Disable all adapters, Enable newest adapter]
    H4 --> H5[Freeze all model parameters, Unfreeze newest adapter parameters]
    H5 --> I["trainer.fit(model)"]
```

# Evaluate adapted models

In order to easily evaluate adapted models, you can use the `eval_asr_adapter.py` script, which takes in the path / name of an adapted model, and then selects one of the any number of adapter names to evaluate over.

## Evaluation execution flow diagram

```mermaid

graph TD
    A[Hydra Overrides + Yaml Config] --> Bo{Config}
    Bo --> B[Update Config for Adapter Supported Modules]
    B --> |Init| C[Trainer]
    C --> E[Pretrained Adapted Model Restore]
    E --> |Constructor| F1(Change Vocabulary)
    F1 --> G(Setup Test Data loaders)
    G --> H1(Setup Optimization)
    H1 --> H2(Setup Older Adapters)
    H2 --> H4{Adapter Name Provided}
    H4 --> |Yes| H5[Disable all Adapters, Enable selected Adapter]
    H4 --> |No| H6[Disable all Adapters] 
    H5 --> Ho[Freeze Weights]
    H6 --> Ho[Freeze Weights]
    Ho --> I["trainer.test(model)"]
```

**Note**: If you wish to evaluate the base model (with all adapters disabled), simply pass `model.adapter.adapter_name=null` to the config of this script to disable all adapters and evaluate just the base model.

# Scoring and Analysis of Results

The script `scoring_and_analysis.py` can be used to calculate the scoring metric for selecting hyperparameters for constrained and unconstrained adaptation experiments as outlined in [Damage Control During Domain Adaptation for Transducer Based Automatic Speech Recognition](https://arxiv.org/abs/2210.03255).

The script takes in as input a csv file containing all the hyperparameters and their corresponding WERs. Currently, it shows how it can be used to perform analysis on the [Crowdsourced high-quality UK and Ireland English Dialect speech data set](http://www.openslr.org/83/). To use it for other experiments, please updated the global variables outlined in the beginning of the script accordingly. These global variables correspond to the column names within the input csv file.
