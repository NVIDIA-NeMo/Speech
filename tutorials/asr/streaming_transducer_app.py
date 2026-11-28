# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Gradio Demo for Streaming Speech Recognition using NeMo Models

Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU devices.
"""

import threading

import gradio as gr
import librosa
import numpy as np

from nemo.collections.asr.parts.utils.one_stream_session import StreamingConfig, StreamingSession

ASR_HF_MODELS: list[str] = [
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-tdt-1.1b",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-0.6b-v3",
]

audio_processing_lock = threading.Lock()


def initialize_session(
    session: StreamingSession, model_name: str, left_context: float, chunk_size: float, right_context: float
):
    """Update the streaming session"""
    try:
        config = StreamingConfig(
            model_name=model_name,
            left_context_secs=left_context,
            chunk_secs=chunk_size,
            right_context_secs=right_context,
        )
        with audio_processing_lock:
            session.setup_streaming_session(config)
        # Return success status and clear transcription
        status = f"Model loaded: {model_name} on device '{session.config.device}'"
        return session, status, gr.update(interactive=True), "", ""
    except Exception as e:
        status = f"Error loading model: {str(e)}"
        return session, status, gr.update(interactive=False), "", ""


def process_microphone_chunk(session: StreamingSession, audio_data, transcription_text):
    """Process audio chunk from microphone"""
    if audio_data is None:
        return transcription_text

    # NB: we need to ensure that each chunk is processed sequentially, thus we need a lock
    with audio_processing_lock:
        sample_rate, audio = audio_data

        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sample_rate != session.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=session.sample_rate)

        # Process the audio chunk
        session.process_audio_chunk(audio, is_last=False)

    # Return updated transcription
    return session, session.transcription, f"{session.rtfx:.3f}" if session.rtfx else ""


def stop_microphone(session: StreamingSession):
    """Handle microphone stop event"""
    with audio_processing_lock:
        session.flush()
    return session, session.transcription


# Create Gradio interface
with gr.Blocks(title="Streaming Speech Recognition") as demo:
    gr.Markdown("# Streaming Speech Recognition with NeMo Models")

    with gr.Row():
        with gr.Column(scale=1):
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=ASR_HF_MODELS,
                value=ASR_HF_MODELS[0],
                label="Model Selection",
                info="Select from available models",
                interactive=True,
            )

            # Streaming parameters
            left_context = gr.Slider(
                minimum=0.5,
                maximum=20.0,
                value=5.0,
                step=0.5,
                label="Left Context (seconds)",
                info="Larger values improve quality",
                interactive=True,
            )

            chunk_size = gr.Slider(
                minimum=0.08,
                maximum=5.0,
                value=0.32,
                step=0.08,
                label="Chunk Size (seconds)",
                info="Processing chunk size",
                interactive=True,
            )

            right_context = gr.Slider(
                minimum=0.16,
                maximum=5.0,
                value=0.96,
                step=0.08,
                label="Right Context (seconds)",
                info="Future context for better accuracy",
                interactive=True,
            )

            # Control buttons
            load_button = gr.Button("Load Session", variant="primary")

            # Status
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
            rtfx_field = gr.Textbox(label="RTFx", value="", interactive=False)

        with gr.Column(scale=2):
            # Audio input
            audio_input = gr.Audio(
                label="Microphone Input",
                sources=["microphone"],
                type="numpy",
                streaming=True,
                interactive=False,  # Initially disabled
                waveform_options=dict(sample_rate=16000),
            )

            # Transcription output
            transcription_output = gr.Textbox(
                label="Transcription",
                placeholder="Transcribed text will appear here...",
                lines=10,
                max_lines=20,
                interactive=False,
            )
        gr_session = gr.State(StreamingSession())

    # Add usage instructions
    gr.Markdown(
        """
    ## Instructions:
    1. Select a model and configure streaming parameters
    2. Click "Load Session" to load the model and initialize the session
    3. Wait for the model to load (buttons will be disabled during loading)
    4. Once loaded, click on the microphone to start recording
    5. Speak into your microphone - transcription will appear in real-time
    6. Click the microphone again to stop recording
    
    **Note**: 
    - First-time model download can take 10-30 minutes depending on your internet connection
    - Subsequent loads will be much faster as models are cached locally
    - Check the terminal/console for detailed progress information
    
    **Known issues:**:
    - you can start stream only once; you need to reload the page if you want to start one more session
    - with RTFx < 1 the output can be incorrect

    **Device Support**: Automatically uses CUDA (NVIDIA GPU) -> MPS (Apple Silicon) -> CPU.
    """
    )

    # Event handlers
    load_button.click(
        fn=initialize_session,
        inputs=[gr_session, model_dropdown, left_context, chunk_size, right_context],
        outputs=[gr_session, status_text, audio_input, transcription_output, rtfx_field],
    )

    # Handle streaming audio
    audio_input.stream(
        fn=process_microphone_chunk,
        stream_every=0.16,  # 0.16 sec - 2 frames
        inputs=[gr_session, audio_input, transcription_output],
        outputs=[gr_session, transcription_output, rtfx_field],
    )

    # Handle microphone stop
    audio_input.stop_recording(
        fn=stop_microphone,
        inputs=[gr_session],
        outputs=[gr_session, transcription_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
