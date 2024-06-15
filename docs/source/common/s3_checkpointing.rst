****************
S3 Checkpointing
****************

S3CheckpointIO
==============

This checkpoint_io is used for saving and loading files to and from S3. 
Initializing this checkpoint_io requires the dirpath be an S3 dirpath. 

**Example Usage:**

.. code-block:: bash

    async_checkpointing = self.cfg.s3_checkpointing.get('enable_async_checkpointing', False)
    chunk_size_MB = self.cfg.s3_checkpointing.get('chunk_size_MB')
    max_read_concurrency = self.cfg.s3_checkpointing.get('max_read_concurrency')
    max_write_concurrency = self.cfg.s3_checkpointing.get('max_write_concurrency')
    dirpath = self.cfg.exp_manager.checkpoint_callback_params.get('dirpath')

    s3_checkpoint_io = S3CheckpointIO(dirpath=dirpath, chunk_size_MB=chunk_size_MB, max_read_concurrency=max_read_concurrency, max_write_concurrency=max_write_concurrency, async_checkpointing=async_checkpointing)

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        checkpoint_io=s3_checkpoint_io,
        gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
        nccl_communicator_config_path=self.cfg.model.get('nccl_communicator_config_path', None),
        sharp=self.cfg.model.get('sharp', False),
    )


**Config changes:**

.. code-block:: bash
    
    checkpoint_callback_params:
    dirpath: s3://mstar-eks-dev-us-east-2/alxzhang/nemo123/1n/checkpoints
    
    ...

    s3_checkpointing:
        # write_concurrency * tp * pp * 1.15 (buffer) should be within 3500 S3 TPS limit per partition
        max_write_concurrency: 10
        # read_concurrency * tp * pp * 1.15 (buffer) should be within 5500 S3 TPS limit per partition
        max_read_concurrency: 15
        chunk_size_MB: 64
        # enables asynchronous checkpoint writing to S3
        enable_async_checkpointing: False

**Asynchronous**
By default, the S3CheckpointIO class acts synchronously. 
The async feature currently does not check if the previous async save is completed, so it is possible
that an old checkpoint is removed even when the current save fails. 
To prevent this, this feature is meant to be used in conjunction with saving top k checkpoints. 


S3Utils and Dependencies
========================

This utility class is used by the S3CheckpoinIO and the exp_manager to do S3-related operations. 
It has dependencies on 

1. boto3[crt]

2. s3fs==0.4.2

3. tenacity

If any of these are missing, this class can't be used. 



s3_dirpath_utils
================

Used to operate on strings by checking if they are S3 dirpaths, or convert a bucket and key into an s3 dirpath. 
This has no reliance on the S3Utils utility class, and can be used without any new dependencies. 


S3 Demands and ExpManager Details When Running at Scale
=======================================================

Typically, in the ExpManager, every rank looks for the checkpoint file to  load from. At large scale, there can be thousands of ranks querying S3 for dirpaths which can cause slowdown or throttling errors. 

To avoid overloading S3 when resuming from a checkpoint only rank 0 needs to identify the checkpoint path and find the correct resumption file. Rank 0 will broadcast the checkpoint path to the other ranks. 

.. code-block:: bash

    trainer._checkpoint_connector = NeMoCheckpointConnector(trainer)

The NeMoModelCheckpoint setup() method will automatically broadcast the checkpoint path. 

The NeMoCheckpointConnector is defined in the exp_manager.py file, and uses the broadcasted checkpoint path founds by rank 0 on all ranks when resuming training from an existing checkpoint. 

The setting of the trainer._checkpoint_connector needs to happen before the ExpManager call as the ExpManager updates the trainer's checkpoint connector. 
