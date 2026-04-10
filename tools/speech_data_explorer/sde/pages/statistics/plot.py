import numpy as np
import plotly.graph_objects as go

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


def gpu_plot_histogram(samples_datatable, field, nbins=50):
    data = samples_datatable[field]

    if field in figures_labels:
        x_axis_title = figures_labels[field][1]
    else:
        x_axis_title = field

    min_value = data.min()
    max_value = data.max()
    step = (max_value - min_value) / nbins
    bins = np.arange(min_value, max_value + step, step)
    hist, _ = np.histogram(data, bins=bins)

    midpoints = np.arange(min_value + step / 2, max_value + step / 2, step)
    trace = go.Bar(x=midpoints, y=hist, marker_color="green")
    layout = go.Layout(xaxis=dict(title=x_axis_title), yaxis=dict(title='Amount'))
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200)
    fig.update_yaxes(type="log")
    return fig


def gpu_plot_word_accuracy(vocabulary_data, field):
    data = vocabulary_data[field]

    epsilon = 1e-6
    amounts, _ = np.histogram(data, bins=[0, epsilon, 100, 100])
    percentage = ['{:.2%}'.format(amount / sum(amounts)) for amount in amounts]
    labels = ["Unrecognized", "Sometimes recognized", "Always recognized"]
    colors = ['red', 'orange', 'green']
    trace = go.Bar(x=labels, y=amounts, text=percentage, textposition='auto', marker_color=colors)
    layout = go.Layout(xaxis=dict(title='Amount'), yaxis=dict(title='Words amount'))
    fig = go.Figure(data=[trace], layout=layout)

    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200, yaxis={'title_text': '#words'}
    )

    return fig
