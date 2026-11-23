# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run reference-text alignment and create a portable, word-audition HTML report.

Manifest entries contain ``id``, ``audio_filepath``, ``text``, and ``language``.
Audio paths are resolved relative to the manifest. For example::

    python examples/tts/inspect_nemotron_timestamp_aligner.py \
        --manifest reference.jsonl --model-path /path/to/hf/snapshot \
        --output-json alignments.json --output-html inspection.html

Rebuild the viewer without loading a model::

    python examples/tts/inspect_nemotron_timestamp_aligner.py \
        --alignments-json alignments.json --output-html inspection.html

The HTML contains one PCM WAV per sample and has no external dependencies. Word
buttons schedule exactly the annotated interval with Web Audio; no padding or
HTMLAudioElement time-update polling is used. A bad interval is reported and
clamped to the recording, never silently played beyond its bounds.

Inference preserves the teacher checkpoint's mixed precision by default. A run
with any failed recording writes its diagnostic outputs and exits with status 1.
Render-only rebuilds preserve failed records without failing the command.
"""

import argparse
import base64
import io
import json
import math
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any


def build_inspection_html(
    records: list[dict[str, Any]],
    output_html: str | Path,
    metadata: dict[str, Any] | None = None,
    title: str = 'Nemotron reference-word timestamp inspection',
) -> Path:
    """Write a self-contained viewer and return its path.

    Each record contains ``id``, ``language``, ``text``, and ``words`` (a list of
    ``{word, start, end, confidence?}``). Supply ``audio_filepath`` or an embedded
    ``audio_wav_base64`` string. ``error``, ``diagnostics``, ``timings``, and
    language confidence/probabilities are optional. ``word_timestamps`` and
    ``start_time``/``end_time`` aliases are accepted for existing exports.
    """
    prepared = [_prepare_record(record, index) for index, record in enumerate(records)]
    payload = {'title': title, 'metadata': metadata or {}, 'records': prepared}
    # JSON is data inside a script element: escape HTML delimiters and JS line separators.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
    for character, escaped in (
        ('&', '\\u0026'),
        ('<', '\\u003c'),
        ('>', '\\u003e'),
        ('\u2028', '\\u2028'),
        ('\u2029', '\\u2029'),
    ):
        serialized = serialized.replace(character, escaped)
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_HTML_TEMPLATE.replace('__REPORT_DATA__', serialized), encoding='utf-8')
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='Reference audio/text JSONL manifest.')
    source.add_argument('--alignments-json', type=Path, help='Existing JSON/JSONL records; skip model inference.')
    parser.add_argument('--model-path', type=Path, help='Local Nemotron ASR Hugging Face snapshot directory.')
    parser.add_argument('--teacher-config', type=Path, help='Additional JSON teacher configuration overrides.')
    parser.add_argument('--output-html', required=True, type=Path)
    parser.add_argument('--output-json', type=Path, help='Save records with embedded audio for portable rebuilding.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--encoder-dtype', choices=['checkpoint', 'float32', 'bfloat16'], default='checkpoint')
    parser.add_argument(
        '--samples-per-language', type=int, default=0, help='Select the first N per language; 0 keeps all.'
    )
    parser.add_argument('--title', default='Nemotron reference-word timestamp inspection')
    args = parser.parse_args()
    if args.batch_size < 1 or args.sample_rate < 1 or args.samples_per_language < 0:
        parser.error('batch-size/sample-rate must be positive and samples-per-language nonnegative')
    if args.manifest and not args.model_path:
        parser.error('--model-path is required with --manifest')

    started = time.perf_counter()
    input_path = args.alignments_json or args.manifest
    records, metadata = _load_records(input_path)
    if args.samples_per_language:
        counts = Counter()
        selected = []
        for record in records:
            language = str(record.get('language', 'unknown'))
            if counts[language] < args.samples_per_language:
                selected.append(record)
                counts[language] += 1
        records = selected
    for index, record in enumerate(records):
        record.setdefault('id', str(index + 1))
        if record.get('audio_filepath'):
            path = Path(record['audio_filepath']).expanduser()
            record['audio_filepath'] = str(path if path.is_absolute() else input_path.parent / path)
    if args.manifest:
        records, run_metadata = _align_manifest(records, args)
        metadata.update(run_metadata)
    metadata['report_preparation_seconds'] = round(time.perf_counter() - started, 4)
    # Embed once before writing both outputs, so the JSON also travels independently.
    records = [_prepare_record(record, index) for index, record in enumerate(records)]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({'metadata': metadata, 'records': records}, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
    output = build_inspection_html(records, args.output_html, metadata=metadata, title=args.title)
    print(
        json.dumps({'output_html': str(output.resolve()), 'samples': len(records), 'metadata': metadata}, default=str)
    )
    failures = sum(bool(record.get('error')) for record in records)
    if args.manifest and failures:
        print(
            f'Alignment failed for {failures}/{len(records)} recordings. Diagnostic report: {output.resolve()}',
            file=sys.stderr,
        )
        return 1
    return 0


def _load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content = path.read_text(encoding='utf-8')
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(value, dict):
        metadata = value.get('metadata', {})
        records = value.get('records', value.get('alignments'))
        if records is None and 'audio_filepath' in value:
            records = [value]
    else:
        metadata, records = {}, value
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f'{path} must contain a list of records, a records object, or JSONL records')
    return records, metadata


def _read_audio(path: str | Path, sample_rate: int | None = None):
    import numpy as np
    import soundfile as sf

    audio, source_rate = sf.read(str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('Audio is empty or contains nonfinite samples')
    if sample_rate is not None and source_rate != sample_rate:
        import torch
        import torchaudio.functional as audio_functional

        audio = audio_functional.resample(torch.from_numpy(audio), source_rate, sample_rate).numpy()
        source_rate = sample_rate
    return audio, source_rate


def _encode_wav(audio, sample_rate: int) -> str:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _align_manifest(records: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from nemo.collections.tts.modules.nemotron_tts.v2.nemotron_asr_aligner import NemotronASRTimestampAligner

    config = {'snapshot_path': str(args.model_path), 'measure_timings': True, 'encoder_dtype': args.encoder_dtype}
    if args.teacher_config:
        config.update(json.loads(args.teacher_config.read_text(encoding='utf-8')))
    initialized = time.perf_counter()
    teacher = NemotronASRTimestampAligner(config).to(args.device).eval()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    metadata = {
        'model_path': str(args.model_path),
        'teacher_config': config,
        'device': args.device,
        'batch_size': args.batch_size,
        'sample_rate': args.sample_rate,
        'model_load_seconds': round(time.perf_counter() - initialized, 4),
        'batch_timings': [],
    }
    inference_started = time.perf_counter()
    for first in range(0, len(records), args.batch_size):
        batch_records, waveforms = [], []
        for record in records[first : first + args.batch_size]:
            try:
                audio, sample_rate = _read_audio(record['audio_filepath'], args.sample_rate)
                record['audio_wav_base64'] = _encode_wav(audio, sample_rate)
                record['duration'] = len(audio) / sample_rate
                batch_records.append(record)
                waveforms.append(torch.from_numpy(audio))
            except Exception as error:
                record.update(words=[], error=f'{type(error).__name__}: {error}')
        if not waveforms:
            continue
        lengths = torch.tensor([len(audio) for audio in waveforms], dtype=torch.long, device=args.device)
        audio_batch = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).to(args.device)
        try:
            with torch.inference_mode():
                results = teacher(
                    audio_batch,
                    lengths,
                    args.sample_rate,
                    [record['text'] for record in batch_records],
                    [record.get('language') for record in batch_records],
                )
            if len(results) != len(batch_records):
                raise ValueError(f'Teacher returned {len(results)} results for {len(batch_records)} recordings')
            for record, result in zip(batch_records, results):
                record.update(result)
                if not record.get('words') and not record.get('error'):
                    record['error'] = 'Teacher returned an empty alignment'
            metadata['batch_timings'].append(
                {
                    'ids': [record['id'] for record in batch_records],
                    'audio_seconds': sum(record['duration'] for record in batch_records),
                    'seconds': getattr(teacher, 'last_timings', {}),
                }
            )
        except Exception as error:
            for record in batch_records:
                record.update(words=[], error=f'{type(error).__name__}: {error}')
        print(f'Processed {min(first + args.batch_size, len(records))}/{len(records)}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    metadata['inference_seconds'] = round(time.perf_counter() - inference_started, 4)
    metadata['audio_seconds'] = round(sum(record.get('duration', 0) for record in records), 4)
    metadata['errors'] = sum(bool(record.get('error')) for record in records)
    return records, metadata


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prepare_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    prepared = _json_safe(dict(record))
    prepared['id'] = str(record.get('id', index + 1))
    prepared['language'] = str(record.get('language', 'unknown'))
    prepared['text'] = str(record.get('text', ''))
    warnings = list(record.get('viewer_warnings') or [])
    duration = _finite_number(record.get('duration'))
    try:
        encoded = record.get('audio_wav_base64', record.get('audio_base64'))
        if encoded:
            if encoded.startswith('data:'):
                encoded = encoded.split(',', 1)[1]
            with wave.open(io.BytesIO(base64.b64decode(encoded)), 'rb') as audio:
                duration = audio.getnframes() / audio.getframerate()
            prepared['audio_wav_base64'] = encoded
        elif record.get('audio_filepath'):
            waveform, sample_rate = _read_audio(record['audio_filepath'])
            prepared['audio_wav_base64'] = _encode_wav(waveform, sample_rate)
            duration = len(waveform) / sample_rate
        else:
            warnings.append('No embedded audio or audio_filepath; playback unavailable.')
    except Exception as error:
        warnings.append(f'Audio unavailable: {type(error).__name__}: {error}')
        prepared.pop('audio_wav_base64', None)
    duration = max(0.0, duration or 0.0)
    prepared['duration'] = duration
    normalized_words, previous_end = [], None
    for word_index, word in enumerate(record.get('words', record.get('word_timestamps', [])) or []):
        word = dict(word)
        start = _finite_number(word.get('start', word.get('start_time')))
        end = _finite_number(word.get('end', word.get('end_time')))
        issue = word.get('interval_issue')
        if start is None or end is None:
            issue = 'Nonfinite or missing interval'
            start = end = 0.0
        elif end < start:
            issue = 'End precedes start'
            end = start
        raw_start, raw_end = start, end
        start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
        if (start, end) != (raw_start, raw_end):
            issue = f'{issue}; clamped to recording' if issue else 'Clamped to recording'
        if end <= start and not (issue and 'empty interval' in issue.lower()):
            issue = f'{issue}; empty interval' if issue else 'Empty interval'
        if issue:
            warnings.append(f'Word {word_index + 1}: {issue}.')
        normalized = _json_safe(word)
        normalized.update(
            word=str(word.get('word', word.get('text', word.get('token', '')))),
            start=start,
            end=end,
            duration=end - start,
            gap_before=None if previous_end is None else start - previous_end,
            interval_issue=issue,
        )
        normalized_words.append(normalized)
        previous_end = end
    prepared['words'] = normalized_words
    prepared['viewer_warnings'] = list(dict.fromkeys(warnings))
    return prepared


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference-word timestamp inspection</title>
<style>
:root{color-scheme:light;--ink:#172435;--muted:#506377;--border:#cfd9e2;--accent:#006c67}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1240px;margin:auto;padding:28px 24px 70px}h1{font-size:1.8rem;margin:0 0 8px}
h2{font-size:1.1rem;margin:0}p{margin:8px 0 16px}.muted{color:var(--muted)}
button,select{font:inherit}button{border:1px solid var(--border);border-radius:7px;background:white;color:var(--ink);
padding:7px 12px;cursor:pointer}button:hover{border-color:var(--accent);background:#edf8f5}
button:focus-visible,summary:focus-visible{outline:3px solid #0078c6;outline-offset:3px}
button:disabled{opacity:.45;cursor:not-allowed}
.playing{background:#c7f0e4!important;border-color:var(--accent)!important}
.toolbar,.heading,.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.toolbar{margin:20px 0}
.filters button[aria-pressed="true"]{background:var(--accent);color:white;border-color:var(--accent)}
.sample{background:white;border:1px solid var(--border);border-radius:12px;padding:20px;margin:18px 0}
.heading{justify-content:space-between}.tag{font-size:.8rem;background:#e8edf2;padding:3px 9px;border-radius:20px}
.transcript{font-size:1.05rem;margin:14px 0}.words{display:flex;flex-wrap:wrap;gap:7px;margin:15px 0}
.word{display:inline-grid;text-align:center;min-width:44px}
.word small{font-size:.68rem;color:var(--muted);font-variant-numeric:tabular-nums}
.error,.warning{border-left:4px solid #ad3d21;padding:8px 12px;background:#fff2ed;margin:12px 0;white-space:pre-wrap}
.warning{border-color:#b67b00;background:#fff8e5;font-size:.88rem}
details{margin-top:12px}summary{cursor:pointer;font-weight:600}
.table-wrap{overflow:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{text-align:right;border-bottom:1px solid #e4eaf0;padding:6px 10px;white-space:nowrap}
th:nth-child(2),td:nth-child(2){text-align:left}
td button{padding:2px 8px}tr.bad{background:#fff2ed}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}
.stats{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--muted);margin-top:9px}
#status{min-height:1.5em;color:var(--accent)}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
@media(max-width:600px){main{padding:20px 12px}.sample{padding:14px}h1{font-size:1.4rem}.word{padding:5px 8px}}
</style></head><body><main>
<h1 id="title"></h1><p class="muted">Reference audio and reference-text alignments. Select a word to hear only its
annotated interval. Use Play recording for context. No interval padding is added; Escape stops playback.</p>
<div id="summary" class="muted"></div>
<div class="toolbar"><div id="filters" class="controls filters" aria-label="Filter language"></div>
<button id="stop" type="button">Stop playback</button></div>
<div id="status" role="status" aria-live="polite">Ready. Audio starts only when you select a playback button.</div>
<details id="run-details"><summary>Run metadata and timings</summary><pre id="metadata"></pre></details>
<div id="samples"></div>
</main><script id="report-data" type="application/json">__REPORT_DATA__</script><script>
'use strict';
const report=JSON.parse(document.getElementById('report-data').textContent);
const records=report.records, samples=document.getElementById('samples'), status=document.getElementById('status');
document.title=report.title;document.getElementById('title').textContent=report.title;
document.getElementById('metadata').textContent=JSON.stringify(report.metadata,null,2);
const languages=[...new Set(records.map(r=>r.language))];
const count=records.filter(r=>r.error).length;
document.getElementById('summary').textContent=
 `${records.length} recordings · ${records.length-count} without alignment errors · ${count} errors · `+
 languages.map(l=>`${l}: ${records.filter(r=>r.language===l).length}`).join(' · ');
let context=null, activeSource=null, activeButton=null, playRequest=0;
const buffers=new Map();
const sec=x=>Number.isFinite(x)?x.toFixed(3):'—';
const label=x=>x===null||x===undefined?'—':typeof x==='object'?JSON.stringify(x):String(x);
function node(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;
 if(className)e.className=className;return e;}
function button(text,action,className){const b=node('button',text,className);
 b.type='button';b.addEventListener('click',action);return b;}
function stop(update=true){++playRequest;if(activeSource){try{activeSource.stop();}catch(e){}activeSource=null;}
 if(activeButton){activeButton.classList.remove('playing');
 activeButton.removeAttribute('aria-current');activeButton=null;}
 if(update)status.textContent='Playback stopped.';}
document.getElementById('stop').addEventListener('click',()=>stop());
document.addEventListener('keydown',e=>{if(e.key==='Escape')stop();});
async function decoded(index){if(buffers.has(index))return buffers.get(index);
 const raw=atob(records[index].audio_wav_base64), bytes=new Uint8Array(raw.length);
 for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 const pending=context.decodeAudioData(bytes.buffer);buffers.set(index,pending);
 try{return await pending;}catch(e){buffers.delete(index);throw e;}}
async function play(index,start,end,b,word){stop(false);const request=playRequest;
 try{if(!context)context=new (window.AudioContext||window.webkitAudioContext)();await context.resume();
 const audio=await decoded(index);if(request!==playRequest)return;
 const offset=Math.max(0,Math.min(start,audio.duration));const duration=Math.min(end,audio.duration)-offset;
 if(!Number.isFinite(offset)||!Number.isFinite(duration)||duration<=0)throw Error('Empty or invalid audio interval');
 const source=context.createBufferSource();source.buffer=audio;source.connect(context.destination);
 activeSource=source;activeButton=b;
 b.classList.add('playing');b.setAttribute('aria-current','true');
 status.textContent=`Playing ${records[index].id}: ${word} [${sec(offset)}–${sec(offset+duration)} s]`;
 source.onended=()=>{if(activeSource===source){activeSource=null;activeButton=null;b.classList.remove('playing');
 b.removeAttribute('aria-current');status.textContent='Playback complete.';}};
 source.start(0,offset,duration);
 }catch(e){if(request===playRequest)status.textContent='Playback error: '+e.message;}}
function wordButton(record,index,word,compact=false){let b;
 b=button(compact?word.word:'',()=>play(index,word.start,word.end,b,word.word),'word');
 if(!compact){b.append(node('span',word.word),node('small',`${sec(word.start)}–${sec(word.end)}`));}
 b.disabled=!record.audio_wav_base64||word.end<=word.start;
 b.title=`${word.word}: ${sec(word.start)} to ${sec(word.end)} seconds`+
 (word.interval_issue?'; '+word.interval_issue:'');
 b.setAttribute('aria-label',`Play word ${word.word}, ${sec(word.start)} to ${sec(word.end)} seconds`);return b;}
records.forEach((r,index)=>{const card=node('article',undefined,'sample');card.dataset.language=r.language;
 const heading=node('div',undefined,'heading');
 heading.append(node('h2',`${index+1}. ${r.id}`),node('span',r.language,'tag'));card.append(heading);
 card.append(node('p',r.text,'transcript'));const controls=node('div',undefined,'controls');let full;
 full=button('▶ Play recording',()=>play(index,0,r.duration,full,'full recording'));
 full.disabled=!r.audio_wav_base64||r.duration<=0;
 controls.append(full,node('span',`${sec(r.duration)} s · ${r.words.length} aligned words`,'muted'));
 card.append(controls);
 const stats=node('div',undefined,'stats');
 for(const key of ['language_confidence','detected_language','language_probabilities','confidence']){
 if(r[key]!==undefined)stats.append(node('span',`${key.replaceAll('_',' ')}: ${label(r[key])}`));}
 if(stats.childElementCount)card.append(stats);
 if(r.error)card.append(node('div',label(r.error),'error'));
 if(r.viewer_warnings.length)card.append(node('div',r.viewer_warnings.join('\n'),'warning'));
 const words=node('div',undefined,'words');words.setAttribute('aria-label','Play individual word intervals');
 r.words.forEach(w=>words.append(wordButton(r,index,w)));card.append(words);
 const details=node('details'), summary=node('summary','Word times, gaps, and confidence');details.append(summary);
 const table=node('table'),thead=node('thead'),header=node('tr');
 ['#','Word','Start (s)','End (s)','Length (s)','Gap before (s)','Confidence'].forEach(v=>header.append(node('th',v)));
 thead.append(header);table.append(thead);const body=node('tbody');
 r.words.forEach((w,i)=>{const row=node('tr',undefined,w.interval_issue?'bad':undefined);row.append(node('td',i+1));
 const wordCell=node('td');wordCell.append(wordButton(r,index,w,true));row.append(wordCell);
 [sec(w.start),sec(w.end),sec(w.duration),sec(w.gap_before),label(w.confidence??w.score)]
 .forEach(v=>row.append(node('td',v)));
 body.append(row);});table.append(body);
 const wrap=node('div',undefined,'table-wrap');wrap.append(table);details.append(wrap);
 details.append(node('p','Gap before is the signed interval from the previous word end: '+
 'positive means a gap; negative means overlap.','muted'));
 card.append(details);
 const diagnosticKeys=['diagnostics','timings','alignment','alignment_confidence','language_scores'];
 const diagnostic=Object.fromEntries(diagnosticKeys.filter(k=>r[k]!==undefined).map(k=>[k,r[k]]));
 if(Object.keys(diagnostic).length){const d=node('details');
 d.append(node('summary','Alignment diagnostics and timings'),
 node('pre',JSON.stringify(diagnostic,null,2)));card.append(d);}samples.append(card);});
const filters=document.getElementById('filters');
['All',...languages].forEach(language=>{const b=button(language,()=>{
 stop();for(const card of samples.children)card.hidden=language!=='All'&&card.dataset.language!==language;
 for(const filter of filters.children)filter.setAttribute('aria-pressed',String(filter===b));});
 b.setAttribute('aria-pressed',String(language==='All'));filters.append(b);});
</script></body></html>'''


if __name__ == '__main__':
    raise SystemExit(main())
