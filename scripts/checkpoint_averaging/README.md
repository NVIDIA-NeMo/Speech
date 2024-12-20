Checkpoint Averaging
====================

Overview
--------
The checkpoint averaging script is used to compute the average of multiple distributed checkpoints. This can be useful for improving model performance by combining multiple training states.

When executed, the script processes checkpoints stored in a specified directory, averages their weights, and generates new checkpoint containing the averaged weights.

Average Zarr Distributed Checkpoints
------------------------------------
Use the following command to run the checkpoint averaging script for zarr distributed checkpoints:

```shell
python scripts/checkpoint_averaging/zarr_distributed_checkpoint_averaging.py \
    --name_prefix <output checkpoint name> \
    --checkpoint_dir <folder with zarr distriubted checkpoints> \
    --steps <optionally a list of checkpoint steps to average, if not provided, it will average all the checkpoints>
```
**Arguments**:
- `--name_prefix`: Specifies the prefix for the generated averaged checkpoint.
- `--checkpoint_dir`: Specifies the folder containing zarr distributed checkpoints.
- `--steps`: (Optional) A comma-separated list of checkpoint steps to average (e.g., 1000, 2000, 3000). If not provided, the script will average all the checkpoints in the directory.

After execution, the script generates averaged checkpoint in `<checkpoint_dir>` named `<name_prefix>-averaged`.
