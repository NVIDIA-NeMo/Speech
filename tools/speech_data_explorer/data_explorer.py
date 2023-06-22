# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import csv
import datetime
import difflib
import io
import json
import logging
import math
import operator
import os
import pickle
from collections import defaultdict
from os.path import expanduser
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import diff_match_patch
import editdistance
import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# number of items in a table per page
DATA_PAGE_SIZE = 10

# operators for filtering items
filter_operators = {
    '>=': 'ge',
    '<=': 'le',
    '<': 'lt',
    '>': 'gt',
    '!=': 'ne',
    '=': 'eq',
    'contains ': 'contains',
}
comparison_mode = False

# parse table filter queries
def split_filter_part(filter_part):
    for op in filter_operators:
        if op in filter_part:
            name_part, value_part = filter_part.split(op, 1)
            name = name_part[name_part.find('{') + 1 : name_part.rfind('}')]
            value_part = value_part.strip()
            v0 = value_part[0]
            if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                value = value_part[1:-1].replace('\\' + v0, v0)
            else:
                try:
                    value = float(value_part)
                except ValueError:
                    value = value_part
            return name, filter_operators[op], value
    return [None] * 3


# standard command-line arguments parser
def parse_args():
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument(
        '--disable-caching-metrics', action='store_true', help='disable caching metrics for errors analysis'
    )
    parser.add_argument(
        '--estimate-audio-metrics',
        '-a',
        action='store_true',
        help='estimate frequency bandwidth and signal level of audio recordings',
    )
    parser.add_argument(
        '--audio-base-path',
        default=None,
        type=str,
        help='A base path for the relative paths in manifest. It defaults to manifest path.',
    )
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')

    parser.add_argument(
        '--names_compared',
        '-nc',
        nargs=2,
        type=str,
        help='names of the two fields that will be compared, example: pred_text_contextnet pred_text_conformer. "pred_text_" prefix IS IMPORTANT!',
    )
    parser.add_argument(
        '--show_statistics',
        '-shst',
        type=str,
        help='field name for which you want to see statistics (optional). Example: pred_text_contextnet.',
    )
    args = parser.parse_args()

    # assume audio_filepath is relative to the directory where the manifest is stored
    if args.audio_base_path is None:
        args.audio_base_path = os.path.dirname(args.manifest)

    # automaticly going in comparison mode, if there is names_compared argument
    if args.names_compared is not None:
        comparison_mode = True
        logging.error("comparison mod set to true")
    else:
        comparison_mode = False

    print(args, comparison_mode)
    return args, comparison_mode


# estimate frequency bandwidth of signal
def eval_bandwidth(signal, sr, threshold=-50):
    time_stride = 0.01
    hop_length = int(sr * time_stride)
    n_fft = 512
    spectrogram = np.mean(
        np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window='blackmanharris')) ** 2, axis=1
    )
    power_spectrum = librosa.power_to_db(S=spectrogram, ref=np.max, top_db=100)
    freqband = 0
    for idx in range(len(power_spectrum) - 1, -1, -1):
        if power_spectrum[idx] > threshold:
            freqband = idx / n_fft * sr
            break
    return freqband


# load data from JSON manifest file
def load_data(
    data_filename,
    disable_caching=False,
    estimate_audio=False,
    vocab=None,
    audio_base_path=None,
    comparison_mode=False,
    names=None,
):
    if comparison_mode:
        if names is None:
            logging.error(f'Please, specify names of compared models')
        name_1, name_2 = names

    if not comparison_mode:
        if vocab is not None:
            # load external vocab
            vocabulary_ext = {}
            with open(vocab, 'r') as f:
                for line in f:
                    if '\t' in line:
                        # parse word from TSV file
                        word = line.split('\t')[0]
                    else:
                        # assume each line contains just a single word
                        word = line.strip()
                    vocabulary_ext[word] = 1

        if not disable_caching:
            pickle_filename = data_filename.split('.json')[0]
            json_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(data_filename))
            timestamp = json_mtime.strftime('%Y%m%d_%H%M')
            pickle_filename += '_' + timestamp + '.pkl'
            if os.path.exists(pickle_filename):
                with open(pickle_filename, 'rb') as f:
                    data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available = pickle.load(f)
                if vocab is not None:
                    for item in vocabulary_data:
                        item['OOV'] = item['word'] not in vocabulary_ext
                if estimate_audio:
                    for item in data:
                        filepath = absolute_audio_filepath(item['audio_filepath'], audio_base_path)
                        signal, sr = librosa.load(path=filepath, sr=None)
                        bw = eval_bandwidth(signal, sr)
                        item['freq_bandwidth'] = int(bw)
                        item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(
                        [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                        f,
                        pickle.HIGHEST_PROTOCOL,
                    )
                return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available

    data = []
    wer_count = 0
    cer_count = 0
    wmr_count = 0
    wer = 0
    cer = 0
    wmr = 0
    mwa = 0
    num_hours = 0
    match_vocab_1 = defaultdict(lambda: 0)
    match_vocab_2 = defaultdict(lambda: 0)

    def append_data(
        data_filename, estimate_audio, field_name='pred_text',
    ):
        data = []
        wer_dist = 0.0
        wer_count = 0
        cer_dist = 0.0
        cer_count = 0
        wmr_count = 0
        wer = 0
        cer = 0
        wmr = 0
        mwa = 0
        num_hours = 0
        vocabulary = defaultdict(lambda: 0)
        alphabet = set()
        match_vocab = defaultdict(lambda: 0)

        sm = difflib.SequenceMatcher()
        metrics_available = False
        with open(data_filename, 'r', encoding='utf8') as f:
            for line in tqdm.tqdm(f):
                item = json.loads(line)
                if not isinstance(item['text'], str):
                    item['text'] = ''
                num_chars = len(item['text'])
                orig = item['text'].split()
                num_words = len(orig)
                for word in orig:
                    vocabulary[word] += 1
                for char in item['text']:
                    alphabet.add(char)
                num_hours += item['duration']

                if field_name in item:
                    metrics_available = True
                    pred = item[field_name].split()
                    measures = jiwer.compute_measures(item['text'], item[field_name])
                    word_dist = measures['substitutions'] + measures['insertions'] + measures['deletions']
                    char_dist = editdistance.eval(item['text'], item[field_name])
                    wer_dist += word_dist
                    cer_dist += char_dist
                    wer_count += num_words
                    cer_count += num_chars

                    sm.set_seqs(orig, pred)
                    for m in sm.get_matching_blocks():
                        for word_idx in range(m[0], m[0] + m[2]):
                            match_vocab[orig[word_idx]] += 1
                    wmr_count += measures['hits']
                else:
                    if comparison_mode:
                        if field_name != 'pred_text':
                            if field_name == name_1:
                                logging.error(f"The .json file has no field with name: {name_1}")
                                exit()
                            if field_name == name_2:
                                logging.error(f"The .json file has no field with name: {name_2}")
                                exit()
                data.append(
                    {
                        'audio_filepath': item['audio_filepath'],
                        'duration': round(item['duration'], 2),
                        'num_words': num_words,
                        'num_chars': num_chars,
                        'word_rate': round(num_words / item['duration'], 2),
                        'char_rate': round(num_chars / item['duration'], 2),
                        'text': item['text'],
                    }
                )
                if metrics_available:
                    data[-1][field_name] = item[field_name]
                    if num_words == 0:
                        num_words = 1e-9
                    if num_chars == 0:
                        num_chars = 1e-9
                    data[-1]['WER'] = round(word_dist / num_words * 100.0, 2)
                    data[-1]['CER'] = round(char_dist / num_chars * 100.0, 2)
                    data[-1]['WMR'] = round(measures['hits'] / num_words * 100.0, 2)
                    data[-1]['I'] = measures['insertions']
                    data[-1]['D'] = measures['deletions']
                    data[-1]['D-I'] = measures['deletions'] - measures['insertions']
                if estimate_audio:
                    filepath = absolute_audio_filepath(item['audio_filepath'], data_filename)
                    signal, sr = librosa.load(path=filepath, sr=None)
                    bw = eval_bandwidth(signal, sr)
                    item['freq_bandwidth'] = int(bw)
                    item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
                for k in item:
                    if k not in data[-1]:
                        data[-1][k] = item[k]

            vocabulary_data = [{'word': word, 'count': vocabulary[word]} for word in vocabulary]
            return (
                vocabulary_data,
                metrics_available,
                data,
                wer_dist,
                wer_count,
                cer_dist,
                cer_count,
                wmr_count,
                wer,
                cer,
                wmr,
                mwa,
                num_hours,
                vocabulary,
                alphabet,
                match_vocab,
            )

    (
        vocabulary_data,
        metrics_available,
        data,
        wer_dist,
        wer_count,
        cer_dist,
        cer_count,
        wmr_count,
        wer,
        cer,
        wmr,
        mwa,
        num_hours,
        vocabulary,
        alphabet,
        match_vocab,
    ) = append_data(data_filename, estimate_audio, field_name=fld_nm)
    if comparison_mode:
        (
            vocabulary_data_1,
            metrics_available_1,
            data_1,
            wer_dist_1,
            wer_count_1,
            cer_dist_1,
            cer_count_1,
            wmr_count_1,
            wer_1,
            cer_1,
            wmr_1,
            mwa_1,
            num_hours_1,
            vocabulary_1,
            alphabet_1,
            match_vocab_1,
        ) = append_data(data_filename, estimate_audio, field_name=name_1)
        (
            vocabulary_data_2,
            metrics_available_2,
            data_2,
            wer_dist_2,
            wer_count_2,
            cer_dist_2,
            cer_count_2,
            wmr_count_2,
            wer_2,
            cer_2,
            wmr_2,
            mwa_2,
            num_hours_2,
            vocabulary_2,
            alphabet_2,
            match_vocab_2,
        ) = append_data(data_filename, estimate_audio, field_name=name_2)

    if not comparison_mode:
        if vocab is not None:
            for item in vocabulary_data:
                item['OOV'] = item['word'] not in vocabulary_ext

    if metrics_available or comparison_mode:
        if metrics_available:
            wer = wer_dist / wer_count * 100.0
            cer = cer_dist / cer_count * 100.0
            wmr = wmr_count / wer_count * 100.0
        if comparison_mode:
            if metrics_available_1 and metrics_available_2:
                wer_1 = wer_dist_1 / wer_count_1 * 100.0
                cer_1 = cer_dist_1 / cer_count_1 * 100.0
                wmr_1 = wmr_count_1 / wer_count_1 * 100.0

                wer = wer_dist_2 / wer_count_2 * 100.0
                cer = cer_dist_2 / cer_count_2 * 100.0
                wmr = wmr_count_2 / wer_count_2 * 100.0

                acc_sum_1 = 0
                acc_sum_2 = 0

                for item in vocabulary_data_1:
                    w = item['word']
                    word_accuracy_1 = match_vocab_1[w] / vocabulary_1[w] * 100.0
                    acc_sum_1 += word_accuracy_1
                    item['accuracy_1'] = round(word_accuracy_1, 1)
                mwa_1 = acc_sum_1 / len(vocabulary_data_1)

                for item in vocabulary_data_2:
                    w = item['word']
                    word_accuracy_2 = match_vocab_2[w] / vocabulary_2[w] * 100.0
                    acc_sum_2 += word_accuracy_2
                    item['accuracy_2'] = round(word_accuracy_2, 1)
                mwa_2 = acc_sum_2 / len(vocabulary_data_2)

        acc_sum = 0
        for item in vocabulary_data:
            w = item['word']
            word_accuracy = match_vocab[w] / vocabulary[w] * 100.0
            acc_sum += word_accuracy
            item['accuracy'] = round(word_accuracy, 1)
        mwa = acc_sum / len(vocabulary_data)

    num_hours /= 3600.0

    if not comparison_mode:
        if not disable_caching:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(
                    [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
    if comparison_mode:
        return (
            data,
            wer,
            cer,
            wmr,
            mwa,
            num_hours,
            vocabulary_data,
            alphabet,
            metrics_available,
            data_1,
            wer_1,
            cer_1,
            wmr_1,
            mwa_1,
            num_hours_1,
            vocabulary_data_1,
            alphabet_1,
            metrics_available_1,
            data_2,
            wer_2,
            cer_2,
            wmr_2,
            mwa_2,
            num_hours_2,
            vocabulary_data_2,
            alphabet_2,
            metrics_available_2,
        )

    return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available


# plot histogram of specified field in data list
def plot_histogram(data, key, label):
    fig = px.histogram(
        data_frame=[item[key] for item in data],
        nbins=50,
        log_y=True,
        labels={'value': label},
        opacity=0.5,
        color_discrete_sequence=['green'],
        height=200,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
    return fig


def plot_word_accuracy(vocabulary_data):
    labels = ['Unrecognized', 'Sometimes recognized', 'Always recognized']
    counts = [0, 0, 0]
    for word in vocabulary_data:
        if word['accuracy'] == 0:
            counts[0] += 1
        elif word['accuracy'] < 100:
            counts[1] += 1
        else:
            counts[2] += 1
    colors = ['red', 'orange', 'green']

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=['{:.2%}'.format(count / sum(counts)) for count in counts],
                textposition='auto',
            )
        ]
    )
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200, yaxis={'title_text': '#words'}
    )

    return fig


def absolute_audio_filepath(audio_filepath, audio_base_path):
    """Return absolute path to an audio file.

    Check if a file existst at audio_filepath.
    If not, assume that the path is relative to audio_base_path.
    """
    audio_filepath = Path(audio_filepath)

    if not audio_filepath.is_file() and not audio_filepath.is_absolute():
        audio_filepath = audio_base_path / audio_filepath
        if audio_filepath.is_file():
            filename = str(audio_filepath)
        else:
            filename = expanduser(audio_filepath)
    else:
        filename = expanduser(audio_filepath)

    return filename


# parse the CLI arguments
args, comparison_mode = parse_args()
if args.show_statistics is not None:
    fld_nm = args.show_statistics
else:
    fld_nm = 'pred_text'
# parse names of compared models, if any
if comparison_mode:
    name_1, name_2 = args.names_compared
    print(name_1, name_2)


print('Loading data...')
if not comparison_mode:
    data, wer, cer, wmr, mwa, num_hours, vocabulary, alphabet, metrics_available = load_data(
        args.manifest,
        args.disable_caching_metrics,
        args.estimate_audio_metrics,
        args.vocab,
        args.audio_base_path,
        comparison_mode,
        args.names_compared,
    )
else:
    (
        data,
        wer,
        cer,
        wmr,
        mwa,
        num_hours,
        vocabulary,
        alphabet,
        metrics_available,
        data_1,
        wer_1,
        cer_1,
        wmr_1,
        mwa_1,
        num_hours_1,
        vocabulary_1,
        alphabet_1,
        metrics_available_1,
        data_2,
        wer_2,
        cer_2,
        wmr_2,
        mwa_2,
        num_hours_2,
        vocabulary_2,
        alphabet_2,
        metrics_available_2,
    ) = load_data(
        args.manifest,
        args.disable_caching_metrics,
        args.estimate_audio_metrics,
        args.vocab,
        args.audio_base_path,
        comparison_mode,
        args.names_compared,
    )

print('Starting server...')
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=os.path.basename(args.manifest),
)

figures_labels = {
    'duration': ['Duration', 'Duration, sec'],
    'num_words': ['Number of Words', '#words'],
    'num_chars': ['Number of Characters', '#chars'],
    'word_rate': ['Word Rate', '#words/sec'],
    'char_rate': ['Character Rate', '#chars/sec'],
    'WER': ['Word Error Rate', 'WER, %'],
    'CER': ['Character Error Rate', 'CER, %'],
    'WMR': ['Word Match Rate', 'WMR, %'],
    'I': ['# Insertions (I)', '#words'],
    'D': ['# Deletions (D)', '#words'],
    'D-I': ['# Deletions - # Insertions (D-I)', '#words'],
    'freq_bandwidth': ['Frequency Bandwidth', 'Bandwidth, Hz'],
    'level_db': ['Peak Level', 'Level, dB'],
}
figures_hist = {}
for k in data[0]:
    val = data[0][k]
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if k in figures_labels:
            ylabel = figures_labels[k][0]
            xlabel = figures_labels[k][1]
        else:
            title = k.replace('_', ' ')
            title = title[0].upper() + title[1:].lower()
            ylabel = title
            xlabel = title
        figures_hist[k] = [ylabel + ' (per utterance)', plot_histogram(data, k, xlabel)]

if metrics_available:
    figure_word_acc = plot_word_accuracy(vocabulary)

stats_layout = [
    dbc.Row(dbc.Col(html.H5(children='Global Statistics'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        [
            dbc.Col(html.Div('Number of hours', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Number of utterances', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Vocabulary size', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Alphabet size', className='text-secondary'), width=3),
        ],
        class_name='bg-light mt-2 rounded-top border-top border-start border-end',
    ),
    dbc.Row(
        [
            dbc.Col(
                html.H5(
                    '{:.2f} hours'.format(num_hours),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(len(data), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(
                    '{} words'.format(len(vocabulary)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            dbc.Col(
                html.H5(
                    '{} chars'.format(len(alphabet)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
            ),
        ],
        class_name='bg-light rounded-bottom border-bottom border-start border-end',
    ),
]
if metrics_available:
    stats_layout += [
        dbc.Row(
            [
                dbc.Col(
                    html.Div('Word Error Rate (WER), %', className='text-secondary'), width=3, class_name='border-end'
                ),
                dbc.Col(
                    html.Div('Character Error Rate (CER), %', className='text-secondary'),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.Div('Word Match Rate (WMR), %', className='text-secondary'), width=3, class_name='border-end',
                ),
                dbc.Col(html.Div('Mean Word Accuracy, %', className='text-secondary'), width=3),
            ],
            class_name='bg-light mt-2 rounded-top border-top border-start border-end',
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(wer), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(cer), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(wmr), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(mwa), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                ),
            ],
            class_name='bg-light rounded-bottom border-bottom border-start border-end',
        ),
    ]
stats_layout += [
    dbc.Row(dbc.Col(html.H5(children='Alphabet'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        dbc.Col(html.Div('{}'.format(sorted(alphabet))),), class_name='mt-2 bg-light font-monospace rounded border'
    ),
]
for k in figures_hist:
    stats_layout += [
        dbc.Row(dbc.Col(html.H5(figures_hist[k][0]), class_name='text-secondary'), class_name='mt-3'),
        dbc.Row(dbc.Col(dcc.Graph(id='duration-graph', figure=figures_hist[k][1]),),),
    ]

if metrics_available:
    stats_layout += [
        dbc.Row(dbc.Col(html.H5('Word accuracy distribution'), class_name='text-secondary'), class_name='mt-3'),
        dbc.Row(dbc.Col(dcc.Graph(id='word-acc-graph', figure=figure_word_acc),),),
    ]

wordstable_columns = [{'name': 'Word', 'id': 'word'}, {'name': 'Count', 'id': 'count'}]
if 'OOV' in vocabulary[0]:
    wordstable_columns.append({'name': 'OOV', 'id': 'OOV'})
if metrics_available:
    wordstable_columns.append({'name': 'Accuracy, %', 'id': 'accuracy'})


stats_layout += [
    dbc.Row(dbc.Col(html.H5('Vocabulary'), class_name='text-secondary'), class_name='mt-3'),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='wordstable',
                columns=wordstable_columns,
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                cell_selectable=False,
                page_count=math.ceil(len(vocabulary) / DATA_PAGE_SIZE),
                sort_by=[{'column_id': 'word', 'direction': 'asc'}],
                style_cell={'maxWidth': 0, 'textAlign': 'left'},
                style_header={'color': 'text-primary'},
                css=[{'selector': '.dash-filter--case', 'rule': 'display: none'},],
            ),
        ),
        class_name='m-2',
    ),
    dbc.Row(dbc.Col([html.Button('Download Vocabulary', id='btn_csv'), dcc.Download(id='download-vocab-csv'),]),),
]


@app.callback(
    Output('download-vocab-csv', 'data'),
    [Input('btn_csv', 'n_clicks'), State('wordstable', 'sort_by'), State('wordstable', 'filter_query')],
    prevent_initial_call=True,
)
def download_vocabulary(n_clicks, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)

    with open('sde_vocab.csv', encoding='utf-8', mode='w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerow(vocabulary_view[0].keys())
        for item in vocabulary_view:
            writer.writerow([str(item[k]) for k in item])
    return dcc.send_file("sde_vocab.csv")


@app.callback(
    [Output('wordstable', 'data'), Output('wordstable', 'page_count')],
    [Input('wordstable', 'page_current'), Input('wordstable', 'sort_by'), Input('wordstable', 'filter_query')],
)
def update_wordstable(page_current, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(vocabulary_view):
        page_current = len(vocabulary_view) // DATA_PAGE_SIZE
    return [
        vocabulary_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(vocabulary_view) / DATA_PAGE_SIZE),
    ]


samples_layout = [
    dbc.Row(dbc.Col(html.H5('Data'), class_name='text-secondary'), class_name='mt-3'),
    html.Hr(),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='datatable',
                columns=[{'name': k.replace('_', ' '), 'id': k, 'hideable': True} for k in data[0]],
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                sort_by=[],
                row_selectable='single',
                selected_rows=[0],
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                page_count=math.ceil(len(data) / DATA_PAGE_SIZE),
                style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'center'},
                style_header={
                    'color': 'text-primary',
                    'text_align': 'center',
                    'height': 'auto',
                    'whiteSpace': 'normal',
                },
                css=[
                    {'selector': '.dash-spreadsheet-menu', 'rule': 'position:absolute; bottom: 8px'},
                    {'selector': '.dash-filter--case', 'rule': 'display: none'},
                    {'selector': '.column-header--hide', 'rule': 'display: none'},
                ],
            ),
        )
    ),
] + [
    dbc.Row(
        [
            dbc.Col(
                html.Div(children=k.replace('_', ' ')),
                width=2,
                class_name='mt-1 bg-light font-monospace text-break small rounded border',
            ),
            dbc.Col(html.Div(id='_' + k), class_name='mt-1 bg-light font-monospace text-break small rounded border'),
        ]
    )
    for k in data[0]
]

if metrics_available:
    samples_layout += [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(children='text diff'),
                    width=2,
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                ),
                dbc.Col(
                    html.Iframe(
                        id='_diff',
                        sandbox='',
                        srcDoc='',
                        style={'border': 'none', 'width': '100%', 'height': '100%'},
                        className='bg-light font-monospace text-break small',
                    ),
                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                ),
            ]
        )
    ]
samples_layout += [
    dbc.Row(dbc.Col(html.Audio(id='player', controls=True),), class_name='mt-3 '),
    dbc.Row(dbc.Col(dcc.Graph(id='signal-graph')), class_name='mt-3'),
]


# updating vocabulary to show


wordstable_columns_tool = [{'name': 'Word', 'id': 'word'}, {'name': 'Count', 'id': 'count'}]
wordstable_columns_tool.append({'name': 'Accuracy_1, %', 'id': 'accuracy_1'})
wordstable_columns_tool.append({'name': 'Accuracy_2, %', 'id': 'accuracy_2'})


if comparison_mode:
    model_name_1, model_name_2 = name_1, name_2

    for i in range(len(vocabulary_1)):
        vocabulary_1[i].update(vocabulary_2[i])

    def _wer_(grnd, pred):
        grnd_words = grnd.split()
        pred_words = pred.split()
        edit_distance = editdistance.eval(grnd_words, pred_words)
        wer = edit_distance / len(grnd_words)
        return wer

    def metric(a, b, met=None):
        cer = editdistance.distance(a, b) / len(a)
        wer = _wer_(a, b)
        return round(float(wer) * 100, 2), round(float(cer) * 100, 2)

    def write_metrics(data, Ox, Oy):
        da = pd.DataFrame.from_records(data)
        gt = da['text']
        tt_1 = da[Ox]
        tt_2 = da[Oy]

        wer_tt1_c, cer_tt1_c = [], []
        wer_tt2_c, cer_tt2_c = [], []

        for j in range(len(gt)):
            wer_tt1, cer_tt1 = metric(gt[j], tt_1[j])  # first model
            wer_tt2, cer_tt2 = metric(gt[j], tt_2[j])  # second model
            wer_tt1_c.append(wer_tt1)
            cer_tt1_c.append(cer_tt1)
            wer_tt2_c.append(wer_tt2)
            cer_tt2_c.append(cer_tt2)

        da['wer_' + Ox] = pd.Series(wer_tt1_c, index=da.index)
        da['wer_' + Oy] = pd.Series(wer_tt2_c, index=da.index)
        da['cer_' + Ox] = pd.Series(cer_tt1_c, index=da.index)
        da['cer_' + Oy] = pd.Series(cer_tt2_c, index=da.index)
        return da.to_dict('records')

    data_with_metrics = write_metrics(data, model_name_1, model_name_2)
    if args.show_statistics is not None:
        textdiffstyle = {'border': 'none', 'width': '100%', 'height': '100%'}
    else:
        textdiffstyle = {'border': 'none', 'width': '1%', 'height': '1%', 'display': 'none'}

    def prepare_data(df, name1=model_name_1, name2=model_name_2):
        res = pd.DataFrame()
        tmp = df['word']
        res.insert(0, 'word', tmp)
        res.insert(1, 'count', [float(i) for i in df['count']])
        res.insert(2, 'accuracy_model_' + name1, df['accuracy_1'])
        res.insert(3, 'accuracy_model_' + name2, df['accuracy_2'])
        res.insert(4, 'accuracy_diff ' + '(' + name1 + ' - ' + name2 + ')', df['accuracy_1'] - df['accuracy_2'])
        res.insert(2, 'count^(-1)', 1 / df['count'])
        return res

    for_col_names = pd.DataFrame()
    for_col_names.insert(0, 'word', ['a'])
    for_col_names.insert(1, 'count', [0])
    for_col_names.insert(2, 'accuracy_model_' + model_name_1, [0])
    for_col_names.insert(3, 'accuracy_model_' + model_name_2, [0])
    for_col_names.insert(4, 'accuracy_diff ' + '(' + model_name_1 + ' - ' + model_name_2 + ')', [0])
    for_col_names.insert(5, 'count^(-1)', [0])

    @app.callback(
        Output('voc_graph', 'figure'),
        [
            Input('xaxis-column', 'value'),
            Input('yaxis-column', 'value'),
            Input('color-column', 'value'),
            Input('size-column', 'value'),
            Input("datatable-advanced-filtering", "derived_virtual_data"),
            Input("dot_spacing", 'value'),
            Input("radius", 'value'),
        ],
        prevent_initial_call=False,
    )
    def draw_vocab(Ox, Oy, color, size, data, dot_spacing='no', rad=0.01):
        import math
        import random

        import pandas as pd

        df = pd.DataFrame.from_records(data)

        res = prepare_data(df)
        res_spacing = res.copy(deep=True)

        if dot_spacing == 'yes':
            rad = float(rad)
            if Ox[0] == 'a' or 'c':
                tmp = []
                for i in range(len(res[Ox])):
                    tmp.append(
                        res[Ox][i]
                        + rad
                        * random.randrange(1, 10)
                        * math.cos(random.randrange(1, len(res[Ox])) * 2 * math.pi / len(res[Ox]))
                    )
                res_spacing[Ox] = tmp
            if Ox[0] == 'a' or 'c':
                tmp = []
                for i in range(len(res[Oy])):
                    tmp.append(
                        res[Oy][i]
                        + rad
                        * random.randrange(1, 10)
                        * math.sin(random.randrange(1, len(res[Oy])) * 2 * math.pi / len(res[Oy]))
                    )
                res_spacing[Oy] = tmp

            res = res_spacing

        fig = px.scatter(
            res,
            x=Ox,
            y=Oy,
            color=color,
            size=size,
            hover_data={'word': True, Ox: True, Oy: True, 'count': True},
            width=1300,
            height=1000,
        )
        if (Ox == 'accuracy_model_' + model_name_1 and Oy == 'accuracy_model_' + model_name_2) or (
            Oy == 'accuracy_model_' + model_name_1 and Ox == 'accuracy_model_' + model_name_2
        ):
            fig.add_shape(
                type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="MediumPurple", width=1, dash="dot",)
            )

        return fig

    @app.callback(
        Output('filter-query-input', 'style'),
        Output('filter-query-output', 'style'),
        Input('filter-query-read-write', 'value'),
    )
    def query_input_output(val):
        input_style = {'width': '100%'}
        output_style = {}
        input_style.update(display='inline-block')
        output_style.update(display='none')
        return input_style, output_style

    @app.callback(Output('datatable-advanced-filtering', 'filter_query'), Input('filter-query-input', 'value'))
    def write_query(query):
        if query is None:
            return ''
        return query

    @app.callback(Output('filter-query-output', 'children'), Input('datatable-advanced-filtering', 'filter_query'))
    def read_query(query):
        if query is None:
            return "No filter query"
        return dcc.Markdown('`filter_query = "{}"`'.format(query))

    ############
    @app.callback(
        Output('filter-query-input-2', 'style'),
        Output('filter-query-output-2', 'style'),
        Input('filter-query-read-write', 'value'),
    )
    def query_input_output(val):
        input_style = {'width': '100%'}
        output_style = {}
        input_style.update(display='inline-block')
        output_style.update(display='none')
        return input_style, output_style

    @app.callback(Output('datatable-advanced-filtering-2', 'filter_query'), Input('filter-query-input-2', 'value'))
    def write_query(query):
        if query is None:
            return ''
        return query

    @app.callback(Output('filter-query-output-2', 'children'), Input('datatable-advanced-filtering-2', 'filter_query'))
    def read_query(query):
        if query is None:
            return "No filter query"
        return dcc.Markdown('`filter_query = "{}"`'.format(query))

    ############

    def display_query(query):
        if query is None:
            return ''
        return html.Details(
            [
                html.Summary('Derived filter query structure'),
                html.Div(
                    dcc.Markdown(
                        '''```json
    {}
    ```'''.format(
                            json.dumps(query, indent=4)
                        )
                    )
                ),
            ]
        )

    comparison_layout = [
        html.Div(
            [
                dcc.Markdown("model 1:" + ' ' + model_name_1[10:]),
                dcc.Markdown("model 2:" + ' ' + model_name_2[10:]),
                dcc.Dropdown(
                    ['word level', 'utterance level'],
                    'word level',
                    placeholder="choose comparison lvl",
                    id='lvl_choose',
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(for_col_names.columns[::], 'accuracy_model_' + model_name_1, id='xaxis-column'),
                        dcc.Dropdown(for_col_names.columns[::], 'accuracy_model_' + model_name_2, id='yaxis-column'),
                        dcc.Dropdown(
                            for_col_names.select_dtypes(include='number').columns[::],
                            placeholder='Select what will encode color of points',
                            id='color-column',
                        ),
                        dcc.Dropdown(
                            for_col_names.select_dtypes(include='number').columns[::],
                            placeholder='Select what will encode size of points',
                            id='size-column',
                        ),
                        dcc.Dropdown(
                            ['yes', 'no'],
                            placeholder='if you want to enable dot spacing',
                            id='dot_spacing',
                            style={'width': '200%'},
                        ),
                        dcc.Input(id='radius', placeholder='Enter radius of spacing (std is 0.01)'),
                        html.Hr(),
                        dcc.Input(id='filter-query-input', placeholder='Enter filter query',),
                    ],
                    style={'width': '200%', 'display': 'inline-block', 'float': 'middle'},
                ),
                html.Hr(),
                html.Div(id='filter-query-output'),
                dash_table.DataTable(
                    id='datatable-advanced-filtering',
                    columns=wordstable_columns_tool,
                    data=vocabulary_1,
                    editable=False,
                    page_action='native',
                    page_size=5,
                    filter_action="native",
                ),
                html.Hr(),
                html.Div(id='datatable-query-structure', style={'whitespace': 'pre'}),
                html.Hr(),
                dbc.Row(dbc.Col(dcc.Graph(id='voc_graph'),),),
                html.Hr(),
            ],
            id='wrd_lvl',
            style={'display': 'block'},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(['WER', 'CER'], 'WER', placeholder="Choose metric", id="choose_metric"),
                        dbc.Row(dbc.Col(html.H5('Data'), class_name='text-secondary'), class_name='mt-3'),
                        html.Hr(),
                        html.Hr(),
                        dcc.Input(
                            id='filter-query-input-2', placeholder='Enter filter query', style={'width': '100%'}
                        ),
                        html.Div(id='filter-query-output-2'),
                        dbc.Row(
                            dbc.Col(
                                [
                                    dash_table.DataTable(
                                        id='datatable-advanced-filtering-2',
                                        columns=[
                                            {'name': k.replace('_', ' '), 'id': k, 'hideable': True}
                                            for k in data_with_metrics[0]
                                        ],
                                        data=data_with_metrics,
                                        editable=False,
                                        page_action='native',
                                        page_size=5,
                                        row_selectable='single',
                                        selected_rows=[0],
                                        page_current=0,
                                        filter_action="native",
                                        style_cell={
                                            'overflow': 'hidden',
                                            'textOverflow': 'ellipsis',
                                            'maxWidth': 0,
                                            'textAlign': 'center',
                                        },
                                        style_header={
                                            'color': 'text-primary',
                                            'text_align': 'center',
                                            'height': 'auto',
                                            'whiteSpace': 'normal',
                                        },
                                        css=[
                                            {
                                                'selector': '.dash-spreadsheet-menu',
                                                'rule': 'position:absolute; bottom: 8px',
                                            },
                                            {'selector': '.dash-filter--case', 'rule': 'display: none'},
                                            {'selector': '.column-header--hide', 'rule': 'display: none'},
                                        ],
                                    ),
                                    dbc.Row(dbc.Col(html.Audio(id='player-1', controls=True),), class_name='mt-3'),
                                ]
                            )
                        ),
                    ]
                    + [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(children=k.replace('_', '-')),
                                    width=2,
                                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                                ),
                                dbc.Col(
                                    html.Div(id='__' + k),
                                    class_name='mt-1 bg-light font-monospace text-break small rounded border',
                                ),
                            ]
                        )
                        for k in data_with_metrics[0]
                    ]
                ),
            ],
            id='unt_lvl',
        ),
    ] + [
        html.Div(
            [
                html.Div(
                    [
                        dbc.Row(dbc.Col(dcc.Graph(id='utt_graph'),),),
                        html.Hr(),
                        dcc.Input(id='clicked_aidopath', style={'width': '100%'}),
                        html.Hr(),
                        dcc.Input(id='my-output-1', style={'display': 'none'}),  # we do need this
                    ]
                ),
                html.Div([dbc.Row(dbc.Col(dcc.Graph(id='signal-graph-1')), class_name='mt-3'),]),
            ],
            id='down_thing',
            style={'display': 'block'},
        )
    ]

if args.show_statistics is not None:
    comparison_layout += [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(children='text diff'),
                            width=2,
                            class_name='mt-1 bg-light font-monospace text-break small rounded border',
                        ),
                        dbc.Col(
                            html.Iframe(
                                id='__diff',
                                sandbox='',
                                srcDoc='',
                                style=textdiffstyle,
                                className='bg-light font-monospace text-break small',
                            ),
                            class_name='mt-1 bg-light font-monospace text-break small rounded border',
                        ),
                    ],
                    id="text_diff_div",
                )
            ],
            id='mid_thing',
            style={'display': 'block'},
        ),
    ]

    @app.callback(
        [
            Output(component_id='wrd_lvl', component_property='style'),
            Output(component_id='unt_lvl', component_property='style'),
            Output(component_id='mid_thing', component_property='style'),
            Output(component_id='down_thing', component_property='style'),
            Input(component_id='lvl_choose', component_property='value'),
        ]
    )
    def show_hide_element(visibility_state):
        if visibility_state == 'word level':
            return (
                {'width': '50%', 'display': 'inline-block', 'float': 'middle'},
                {'width': '50%', 'display': 'none', 'float': 'middle'},
                {'display': 'none'},
                {'display': 'none'},
            )
        else:
            return (
                {'width': '100%', 'display': 'none', 'float': 'middle'},
                {'width': '100%', 'display': 'inline-block', 'float': 'middle'},
                {'display': 'block'},
                {'display': 'block'},
            )


if args.show_statistics is None:

    @app.callback(
        [
            Output(component_id='wrd_lvl', component_property='style'),
            Output(component_id='unt_lvl', component_property='style'),
            Output(component_id='down_thing', component_property='style'),
            Input(component_id='lvl_choose', component_property='value'),
        ]
    )
    def show_hide_element(visibility_state):
        if args.show_statistics is not None:
            a = {'border': 'none', 'width': '100%', 'height': '100%', 'display': 'block'}
        else:
            a = {'border': 'none', 'width': '100%', 'height': '100%', 'display': 'none'}
        if visibility_state == 'word level':
            return (
                {'width': '50%', 'display': 'inline-block', 'float': 'middle'},
                {'width': '50%', 'display': 'none', 'float': 'middle'},
                {'display': 'none'},
            )
        else:
            return (
                {'width': '100%', 'display': 'none', 'float': 'middle'},
                {'width': '100%', 'display': 'inline-block', 'float': 'middle'},
                {'display': 'block'},
            )


store = []


@app.callback(
    [Output('datatable-advanced-filtering-2', 'page_current'), Output('my-output-1', 'value')],
    [Input('utt_graph', 'clickData'),],
)
def real_select_click(hoverData):
    if hoverData is not None:
        path = str(hoverData['points'][0]['customdata'][-1])
        for t in range(len(data_with_metrics)):
            if data_with_metrics[t]['audio_filepath'] == path:
                ind = t
                s = t  #% 5
                sel = s
                pg = math.ceil(ind // 5)
        return pg, sel
    else:
        return 0, 0


@app.callback(
    [Output('datatable-advanced-filtering-2', 'selected_rows')], [Input('my-output-1', 'value')],
)
def real_select_click(num):
    s = num
    return [[s]]


CALCULATED_METRIC = [False, False]


@app.callback(
    [
        Output('utt_graph', 'figure'),
        Output('clicked_aidopath', 'value'),
        Input('choose_metric', 'value'),
        Input('utt_graph', 'clickData'),
        Input('datatable-advanced-filtering-2', 'derived_virtual_data'),
    ],
)
def draw_table_with_metrics(met, hoverData, data_virt):
    Ox = name_1
    Oy = name_2
    if met == "WER":
        cerower = 'wer_'
    else:
        cerower = 'cer_'
    da = pd.DataFrame.from_records(data_virt)

    c = da
    fig = px.scatter(
        c,
        x=cerower + Ox,
        y=cerower + Oy,
        width=1000,
        height=900,
        color='num_words',
        hover_data={
            'text': True,
            Ox: True,
            Oy: True,
            'wer_' + Ox: True,
            'wer_' + Oy: True,
            'cer_' + Ox: True,
            'cer_' + Oy: True,
            'audio_filepath': True,
        },
    )  #'numwords': True,
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="Red", width=1, dash="dot",))
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=10)
    path = None

    if hoverData is not None:
        path = str(hoverData['points'][0]['customdata'][-1])

    return fig, path


@app.callback(
    [Output('datatable', 'data'), Output('datatable', 'page_count')],
    [Input('datatable', 'page_current'), Input('datatable', 'sort_by'), Input('datatable', 'filter_query')],
)
def update_datatable(page_current, sort_by, filter_query):
    data_view = data
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            data_view = [x for x in data_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            data_view = [x for x in data_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        data_view = sorted(data_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(data_view):
        page_current = len(data_view) // DATA_PAGE_SIZE
    return [
        data_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(data_view) / DATA_PAGE_SIZE),
    ]


if comparison_mode:
    app.layout = html.Div(
        [
            dcc.Location(id='url', refresh=False),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                    dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                    dbc.NavItem(dbc.NavLink('Comparison tool', id='comp_tool', href='/comparison')),
                ],
                brand='Speech Data Explorer',
                sticky='top',
                color='green',
                dark=True,
            ),
            dbc.Container(id='page-content'),
        ]
    )
else:
    app.layout = html.Div(
        [
            dcc.Location(id='url', refresh=False),
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                    dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
                ],
                brand='Speech Data Explorer',
                sticky='top',
                color='green',
                dark=True,
            ),
            dbc.Container(id='page-content'),
        ]
    )


if comparison_mode:

    @app.callback(
        [
            Output('page-content', 'children'),
            Output('stats_link', 'active'),
            Output('samples_link', 'active'),
            Output('comp_tool', 'active'),
        ],
        [Input('url', 'pathname')],
    )
    def nav_click(url):
        if url == '/samples':
            return [samples_layout, False, True, False]
        elif url == '/comparison':
            return [comparison_layout, False, False, True]
        else:
            return [stats_layout, True, False, False]


else:

    @app.callback(
        [Output('page-content', 'children'), Output('stats_link', 'active'), Output('samples_link', 'active'),],
        [Input('url', 'pathname')],
    )
    def nav_click(url):
        if url == '/samples':
            return [samples_layout, False, True]
        else:
            return [stats_layout, True, False]


@app.callback(
    [Output('_' + k, 'children') for k in data[0]], [Input('datatable', 'selected_rows'), Input('datatable', 'data')]
)
def show_item(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    return [data[idx[0]][k] for k in data[0]]


if comparison_mode:

    @app.callback(
        [Output('__' + k, 'children') for k in data_with_metrics[0]],
        [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
    )
    def show_item(idx, data):
        if len(idx) == 0:
            raise PreventUpdate
        return [data[idx[0]][k] for k in data_with_metrics[0]]


@app.callback(Output('_diff', 'srcDoc'), [Input('datatable', 'selected_rows'), Input('datatable', 'data'),])
def show_diff(
    idx, data,
):
    if len(idx) == 0:
        raise PreventUpdate
    orig_words = data[idx[0]]['text']
    orig_words = '\n'.join(orig_words.split()) + '\n'

    pred_words = data[idx[0]][fld_nm]
    pred_words = '\n'.join(pred_words.split()) + '\n'

    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []
    for d in diffs:
        diffs_post.append((d[0], d[1].replace('\n', ' ')))

    diff_html = diff.diff_prettyHtml(diffs_post)

    return diff_html


@app.callback(
    Output('__diff', 'srcDoc'),
    [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data'),],
)
def show_diff(
    idx, data,
):
    if len(idx) == 0:
        raise PreventUpdate
    orig_words = data[idx[0]]['text']
    orig_words = '\n'.join(orig_words.split()) + '\n'

    pred_words = data[idx[0]][fld_nm]
    pred_words = '\n'.join(pred_words.split()) + '\n'

    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []
    for d in diffs:
        diffs_post.append((d[0], d[1].replace('\n', ' ')))

    diff_html = diff.diff_prettyHtml(diffs_post)

    return diff_html


@app.callback(Output('signal-graph', 'figure'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        filename = absolute_audio_filepath(data[idx[0]]['audio_filepath'], args.audio_base_path)
        audio, fs = librosa.load(path=filename, sr=None)
        if 'offset' in data[idx[0]]:
            audio = audio[
                int(data[idx[0]]['offset'] * fs) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * fs)
            ]
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(S=np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[[0, 'rgb(30,62,62)'], [0.5, 'rgb(30,128,128)'], [1, 'rgb(30,255,30)'],],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception as ex:
        app.logger.error(f'ERROR in plot signal: {ex}')

    return figs


@app.callback(
    Output('signal-graph-1', 'figure'),
    [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
)
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        filename = absolute_audio_filepath(data[idx[0]]['audio_filepath'], args.audio_base_path)
        audio, fs = librosa.load(path=filename, sr=None)
        if 'offset' in data[idx[0]]:
            audio = audio[
                int(data[idx[0]]['offset'] * fs) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * fs)
            ]
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(S=np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[[0, 'rgb(30,62,62)'], [0.5, 'rgb(30,128,128)'], [1, 'rgb(30,255,30)'],],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception as ex:
        app.logger.error(f'ERROR in plot signal: {ex}')

    return figs


@app.callback(Output('player', 'src'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        filename = absolute_audio_filepath(data[idx[0]]['audio_filepath'], args.audio_base_path)
        signal, sr = librosa.load(path=filename, sr=None)
        if 'offset' in data[idx[0]]:
            signal = signal[
                int(data[idx[0]]['offset'] * sr) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * sr)
            ]
        with io.BytesIO() as buf:
            # convert to PCM .wav
            sf.write(buf, signal, sr, format='WAV')
            buf.seek(0)
            encoded = base64.b64encode(buf.read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception as ex:
        app.logger.error(f'ERROR in audio player: {ex}')
        return ''


@app.callback(
    Output('player-1', 'src'),
    [Input('datatable-advanced-filtering-2', 'selected_rows'), Input('datatable-advanced-filtering-2', 'data')],
)
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        filename = absolute_audio_filepath(data[idx[0]]['audio_filepath'], args.audio_base_path)
        signal, sr = librosa.load(path=filename, sr=None)
        if 'offset' in data[idx[0]]:
            signal = signal[
                int(data[idx[0]]['offset'] * sr) : int((data[idx[0]]['offset'] + data[idx[0]]['duration']) * sr)
            ]
        with io.BytesIO() as buf:
            # convert to PCM .wav
            sf.write(buf, signal, sr, format='WAV')
            buf.seek(0)
            encoded = base64.b64encode(buf.read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception as ex:
        app.logger.error(f'ERROR in audio player: {ex}')
        return ''


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=args.port, debug=args.debug)
