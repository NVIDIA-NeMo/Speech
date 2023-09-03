# NMT gRPC Server Getting Started

## Starting the NMT server

Start the server by specifying multiple models (.nemo files) via the `--model` argument:

```
python server.py --model models/en-es.nemo --model models/en-de.nemo --model models/en-fr.nemo
```

If working with the outputs of a speech recognition system without punctuation and capitalization, you can provide the path to a .nemo model file that performs punctuation and capitalization ex: https://ngc.nvidia.com/catalog/models/nvidia:nemo:punctuation_en_bert via the `--punctuation_model` flag.

NOTE: The server will throw an error if NMT models do not have src_language and tgt_language attributes.

## Notes

Port can be overridden with `--port` flag. Default is 50052. Beam decoder parameters can also be set at server start time. See `--help` for more details.

## Example Text Client

```
python client.py --target_language de --source_language en --text Hello
```

# ASR with Riva + Translation with NeMo cascade

Below, we'll describe how to use Riva's ASR models and NeMo's NMT models to do speech translation via a cascade pipeline.

## Installing Riva and python APIs

Follow instructions in https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html to setup and install Riva along with the python whl.

For latest setup instructions follow the link above, since instructions below may not be up-to-date.

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart:1.4.0-beta
```

```bash
cd riva_quickstart_v1.4.0-beta
bash riva_init.sh
bash riva_start.sh

pip install riva_api-1.4.0b0-py3-none-any.whl
```

This will start a Riva Speech Recognition service and `nvidia-smi` should show `tritonserver` running on GPU0.

## ASR + NMT

Start the NeMo translation server using instructions in the previous section (with or without a punctuation and capitalization model).

Run the cascade client using a single channel audio wav file specifying the target language to translate into. By default, Riva ASR is in English and so we specify only the target language to translate into.

```bash
python asr_nmt_client.py --audio-file recording.mono.wav --asr_punctuation --target_language de
```

To view ASR outputs only

```bash
python asr_nmt_client.py --audio-file recording.mono.wav --asr_punctuation --target_language de --asr_only
```
