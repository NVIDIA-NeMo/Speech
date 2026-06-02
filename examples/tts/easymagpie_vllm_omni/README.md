WIP model definition of EasyMP for vllm-omni. Follows footsteps of qwen3tts:
backbone and LT are compiled into a single cuda graph during uniform batch decoding,
piecewise during mixed/prefill.

Install:
```
pip install -e ".[all]"
pip install ninja mamba_ssm causal_conv1d --no-build-isolation
# install vllm
pip install vllm==0.21.0 vllm_omni==0.21.0rc1
# register vllm models
pip install -e examples/tts/easymagpie_vllm_omni/
```

Conver the checkpoint from
https://huggingface.co/nvidia/easymagpietts_NEXT/tree/main/2605_NemotronTTS_V0.2/v2
```
python examples/tts/easymagpie_vllm_omni/easy_magpietts_convert_to_vllm.py \
  --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
  --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
  --outdir examples/tts/easymagpie_vllm_omni/easymp_vllm_model \
  --context_audio english_sample.wav --speaker_name eng \
  --phoneme_tokenizer_path <ckpt>/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh_ko-KR_pt-BR_ar.json
```

Finally run notebook `examples/tts/easymagpie_vllm_omni/easymagpie_inference_demo.ipynb`
to predict acoustic tokens