# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batched MagpieTTS synthesis of a text jsonl into 16 kHz FLAC + a NeMo manifest (run in the NeMo env).

Each text gets one (speaker, language-mode) voice, assigned round-robin from --voices, e.g. "en:0,en:1,hi:3".
"hi" speaks the English text through the Hindi tokenizer path (an accent experiment); use only voices that pass the
round-trip intelligibility check. Texts that need more than one chunk fall back to the single-utterance path.
"""
import argparse
import json
import os
import random

import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True, help="jsonl with a 'text' field")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--voices", default="en:0,en:1,en:2,en:3,en:4")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default="0/1", help="i/n: process every n-th text starting at i")
    p.add_argument("--apply-tn", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    import librosa
    from nemo.collections.tts.models import MagpieTTSModel
    from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
    from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference, get_tokenizer_for_language

    si, sn = map(int, a.shard.split("/"))
    rows = [json.loads(line) for line in open(a.texts)][si::sn]
    if a.limit:
        rows = rows[: a.limit]
    voices = [(v.split(":")[0], int(v.split(":")[1])) for v in a.voices.split(",")]
    rng = random.Random(a.seed + si)
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    m = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m").cuda().eval()
    sr_in = m.output_sample_rate
    use_lt = m.local_transformer_type != LocalTransformerType.NO_LT
    toks = {}
    for lang, _ in voices:
        toks[lang] = get_tokenizer_for_language(
            lang,
            list(m.tokenizer.tokenizers.keys()),
            language_tokenizer_map=m.cfg.get("language_to_tokenizer_mapping"),
        )
    items = []
    for i, r in enumerate(rows):
        lang, spk = voices[rng.randrange(len(voices))]
        src = r.get("spoken") or r["text"]  # TTS reads the spoken form; the label stays `text`
        text = m._get_normalized_text(transcript=src, language="en") if a.apply_tn else src
        ct, cl, _ = chunk_text_for_inference(
            text=text, language=lang, tokenizer_name=toks[lang], text_tokenizer=m.tokenizer, eos_token_id=m.eos_id
        )
        items.append(dict(row=r, idx=i, lang=lang, spk=spk, chunks=(ct, cl)))
    single = [x for x in items if len(x["chunks"][0]) == 1]
    multi = [x for x in items if len(x["chunks"][0]) != 1]
    single.sort(key=lambda x: int(x["chunks"][1][0]))
    out_rows = []

    def save(x, audio):
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=16000) if sr_in != 16000 else audio
        uid = f"{os.path.basename(a.out_dir)}_{si}_{x['idx']:06d}"
        path = os.path.join(a.out_dir, uid + ".flac")
        sf.write(path, audio, 16000)
        out_rows.append(
            {
                **x["row"],
                "id": uid,
                "audio_filepath": path,
                "duration": round(len(audio) / 16000, 3),
                "voice": f"magpie:{x['lang']}:{x['spk']}",
            }
        )

    with torch.inference_mode():
        for b in range(0, len(single), a.batch_size):
            batch = single[b : b + a.batch_size]
            lens = [int(x["chunks"][1][0]) for x in batch]
            text = torch.zeros(len(batch), max(lens), dtype=torch.long)
            for j, x in enumerate(batch):
                text[j, : lens[j]] = x["chunks"][0][0][: lens[j]]
            inp = {
                "text": text.to(m.device),
                "text_lens": torch.tensor(lens, device=m.device),
                "speaker_indices": [x["spk"] for x in batch],
            }
            st = m.create_chunk_state(batch_size=len(batch))
            o = m.generate_speech(
                inp,
                chunk_state=st,
                end_of_text=[True] * len(batch),
                beginning_of_text=True,
                use_cfg=True,
                use_local_transformer_for_inference=use_lt,
            )
            aud, alen, _ = m._codec_helper.codes_to_audio(o.predicted_codes, o.predicted_codes_lens)
            for j, x in enumerate(batch):
                if int(alen[j]) > 0:
                    save(x, aud[j, : int(alen[j])].float().cpu().numpy())
            print(f"[tts] {min(b + a.batch_size, len(single))}/{len(single)} batched", flush=True)
        for x in multi:
            aud, alen = m.do_tts(
                x["row"].get("spoken") or x["row"]["text"],
                language=x["lang"],
                apply_TN=a.apply_tn,
                speaker_index=x["spk"],
            )
            if int(alen[0]) > 0:
                save(x, aud[0, : int(alen[0])].float().cpu().numpy())
    with open(a.manifest, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[tts] wrote {a.manifest}: {len(out_rows)} rows, {sum(r['duration'] for r in out_rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
