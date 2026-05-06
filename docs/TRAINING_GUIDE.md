# Unified Language-ID ASR Training 

This guide covers how to train cache-aware streaming ASR models with unified language-ID.

- NeMo container: `nvcr.io/nvidia/nemo:25.11.01`
- NeMo branch: https://github.com/ealbasiri/NeMo/tree/unified_archetecture_langid

## Overview

There are two model variants:

| Variant | Training Script | Model Class | Description |
|---------|----------------|-------------|-------------|
| **Hybrid RNNT+CTC** | `speech_to_text_hybrid_rnnt_ctc_bpe_prompt.py` | `EncDecHybridRNNTCTCBPEModelWithPrompt` | Trains with both RNNT and CTC losses (CTC weight = 0.1 recommended). |
| **RNNT-only** | `speech_to_text_rnnt_bpe_prompt.py` | `EncDecRNNTBPEModelWithPrompt` | Trains with RNNT loss only. |

Both variants use the same config, tokenizer, and data format.

## Data Format

### Manifest files must include target_lang or lang/lang field 

```json
{"audio_filepath": "/data/audio/sample.wav", "duration": 5.2, "text": "The transcript of the audio.", "target_lang": "en-US"}
```

`target_lang` Language ID used as the prompt (e.g., `en-US`, `ar-AR`, `de-DE`). Mapped to a prompt index via the `prompt_dictionary` in the config. 

Multi-language example:

```json
{"audio_filepath": "/data/en/audio_001.wav", "duration": 5.7, "text": "No, I don't think I need any further assistance.", "target_lang": "en-US"}
{"audio_filepath": "/data/ko/audio_002.wav", "duration": 5.9, "text": "선생님들 모두 어 우리 민서가 수학에 대한 흥미는", "target_lang": "ko-KR"}
{"audio_filepath": "/data/ar/audio_003.wav", "duration": 9.3, "text": "وللإشارة فإن هذا الاتفاق لا يعد اتفاقا منفصلا", "target_lang": "ar-AR"}
{"audio_filepath": "/data/fr/audio_004.wav", "duration": 4.1, "text": "Bonjour, comment allez-vous aujourd'hui?", "target_lang": "fr-FR"}
```


## Config File

The config is located in the repo at:

```
examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming_prompt.yaml
```

### Key config sections

**Prompt dictionary** -- maps language tags to prompt indices. This is defined in `model.model_defaults.prompt_dictionary`:

```yaml
model:
  model_defaults:
    num_prompts: 128
    prompt_dictionary: {
      'en-US': 0, 'en': 0, 'en-GB': 1, 'es-ES': 2, 'es': 3,
      'zh-CN': 4, 'hi-IN': 6, 'ar-AR': 7, 'fr-FR': 8, 'de-DE': 9,
      'ja-JP': 10, 'ru-RU': 11, 'pt-BR': 12, 'ko-KR': 14, 'it-IT': 15,
      # ... up to 128 language/locale slots
      'auto': 101
    }
```


## Training

### Option 1: Hybrid RNNT+CTC 

This trains with both RNNT and auxiliary CTC loss.

```bash
python3 /code/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_prompt.py \
    --config-path=/code/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming \
    --config-name=${CONFIG_NAME}
```

### Option 2: RNNT-only

Same as above but uses the RNNT-only training script:

```bash
python3 /code/examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py \
    --config-path=/code/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming \
    --config-name=${CONFIG_NAME}
```



## Supported Languages

The prompt dictionary supports 40+ languages. The full mapping is in the config YAML. Add new lang to the dictionary as needed. 

| Language | Tag | Prompt ID |
|----------|-----|-----------|
| English (US) | `en-US` | 0 |
| English (GB) | `en-GB` | 1 |
| Spanish (ES) | `es-ES` | 2 |
| Spanish (US) | `es` | 3 |
| Chinese | `zh-CN` | 4 |
| Hindi | `hi-IN` | 6 |
| Arabic | `ar-AR` | 7 |
| French | `fr-FR` | 8 |
| German | `de-DE` | 9 |
| Japanese | `ja-JP` | 10 |
| Russian | `ru-RU` | 11 |
| Portuguese (BR) | `pt-BR` | 12 |
| Korean | `ko-KR` | 14 |
| Italian | `it-IT` | 15 |
| Auto-detect | `auto` | 101 |

See the full list in the config YAML under `model.model_defaults.prompt_dictionary`.


