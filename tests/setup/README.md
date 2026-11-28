# Functional-test inputs

## ASR adapter checkpoint

The test action pre-stages `stt_en_conformer_ctc_small` version `1.6.0` on the
selected runner before starting the adapter training container. Both variants in
`L2_ASR_Adapters.sh` restore the prepared `.nemo` file, so training does not fetch
this checkpoint from NGC. Other tests keep their existing setup.

The preparation uses the same test-data mount as training. It does not depend on
`L0_Setup_Test_Data_And_Models`, which can run in parallel on another runner.
A valid checkpoint is reused on later runs. A new download has a 15-second
connection timeout, a 120-second limit per attempt and at most two retries.
The helper checks the archive for the model configuration and weights before
atomically publishing it; failed transfers do not publish partial checkpoints.
A preparation failure stops the job before training.

To prepare the checkpoint manually:

1. Mount the runner's test-data directory at `/home/TestData` in the CI image.
2. Run `bash tests/setup/models/prestage_asr_adapter.sh` from the repository root.
3. Run `bash tests/functional_tests/L2_ASR_Adapters.sh` in the GPU test environment.

For a different staging destination, pass the output filename as the helper's
first argument. The adapter smoke test uses the default path,
`/home/TestData/asr/stt_en_conformer_ctc_small-1.6.0.nemo`.

The archive check is structural, not a cryptographic checksum. If a cached file
is damaged, remove it and rerun preparation. The helper replaces empty or
structurally invalid files automatically.
