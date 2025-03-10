This example can be used to train a small LLaMA model using NeMo2.0 and understand how some of the resiliency features like the ones listed below work.

- in-job restart
- straggler detection
- asynchronous checkpointing

Official documentation for these features can be found in the [NeMo user guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).


See [resiliency-in-pretraining-demo.ipynb](resiliency-in-pretraining-demo.ipynb) for a demo of these features.
[crash_simulator.py](crash_simulator.py) is a crash simulator that can be used to simulate a fatal crash at a specific step and thus see the capabilities of the in-job restart resiliency feature.
[preemption_simulator.py](preemption_simulator.py) is a preemption simulator that can be used to simulate `signal.SIGTERM` and thus see the capabilities of the preemption feature.

