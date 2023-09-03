# Streaming / Buffered ASR

Contained within this directory are scripts to perform streaming or buffered inference of audio files using CTC / Transducer ASR models.

## Difference between streaming and buffered ASR

While we primarily showcase the defaults of these models in buffering mode, note that the major difference between streaming ASR and buffered ASR is the chunk size and the total context buffer size.

If you reduce your chunk size, the latency for your first prediction is reduced, and the model appears to predict the text with shorter delay. On the other hand, since the amount of information in the chunk is reduced, it causes higher WER.

On the other hand, if you increase your chunk size, then the delay between spoken sentence and the transcription increases (this is buffered ASR). While the latency is increased, you are able to obtain more accurate transcripts since the model has more context to properly transcribe the text.
