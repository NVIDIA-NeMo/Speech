## EasyMagpie TTS — vLLM-Omni + Triton service

Streaming TTS server for **EasyMagpieTTS** (NeMo model
`nemo.collections.tts.models.easy_magpietts.EasyMagpieTTSModel` /
`EasyMagpieTTSInferenceModel`, Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec).

The vLLM-Omni model definition (talker that runs the backbone + local
transformer as a single CUDA graph during uniform-batch decoding, piecewise
during prefill/mixed) lives in
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni). A Triton
ensemble wraps it together with a TensorRT codec decoder to serve gRPC
streaming requests.

### Pipeline

1. **Convert the NeMo checkpoint to a vLLM-Omni model directory** — bakes the
   text embedding + CAS lookup, dumps `config.json`, `model.safetensors`, the
   text tokenizer, and optional reference speaker embeddings.

   ```bash
   python examples/tts/easymagpie_vllm_omni/easy_magpietts_convert_to_vllm.py \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --outdir examples/tts/easymagpie_vllm_omni/easymp_vllm_model \
       --context_audio english_sample.wav --speaker_name eng \
       --phoneme_tokenizer_path <ckpt>/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh_ko-KR_pt-BR_ar.json
   ```

   Checkpoints: <https://huggingface.co/nvidia/easymagpietts_NEXT/tree/main/2605_NemotronTTS_V0.2/v2>.

2. **Export the codec decoder to ONNX** — wraps `AudioCodecModel` so a single
   `(B, T, C*S)` int tensor of stacked model codes decodes to a 22.05 kHz
   waveform (clamp specials → unstack → FSQ index-convert → decode baked in).

   ```bash
   python examples/tts/easymagpie_vllm_omni/export_codec_decoder_onnx.py \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --onnx-path examples/tts/easymagpie_vllm_omni/codec.onnx \
       --frames 15 --device cuda
   ```

3. **Build the serving container** (Triton 26.02 + vLLM 0.21.0 +
   vllm-omni 0.21.0rc1 + this plugin).

   ```bash
   docker build --network=host -t easymp-vllm-omni examples/tts/easymagpie_vllm_omni/
   ```

4. **Launch the container** with the workspace and a GPU mounted.

   ```bash
   docker run --rm -it --gpus all --network host --shm-size=8g \
       -v "$PWD":/workspace -w /workspace \
       easymp-vllm-omni bash
   ```

5. **Build the TensorRT engine from the ONNX** (inside the container) and drop
   it into the Triton repo as `model.plan`. For now fp32 seems to be mandatory.

   ```bash
   python examples/tts/easymagpie_vllm_omni/export_codec_decoder_trt.py \
       --onnx-path examples/tts/easymagpie_vllm_omni/codec.onnx \
       --trt-path  examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan \
       --batch-profile 1 8 32 --frames-profile 15 15 15 --fp32
   ```

6. **Start the Triton inference server** against
   [`model_repository/`](model_repository) (two models: `easymp` python
   backend + `codec` TRT plan).

   ```bash
   tritonserver --model-repository=examples/tts/easymagpie_vllm_omni/model_repository
   ```

7. **Send a request.** End-to-end gRPC streaming example in
   [`run_server_request.ipynb`](run_service_request.ipynb) — sends `text`,
   receives streamed `audio` chunks at 22.05 kHz.
