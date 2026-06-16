Speaker Diarization
===================

Speaker Diarization Overview
----------------------------
Speaker diarization is the process of segmenting audio recordings by speaker labels and aims to answer the question "who spoke when?". Speaker diarization makes a clear distinction when it is compared with speech recognition. As shown in the figure below, before we perform speaker diarization, we know "what is spoken" yet we do not know "who spoke it". Therefore, speaker diarization is an essential feature for a speech recognition system to enrich the transcription with speaker labels.

.. image:: images/asr_sd_diagram.png
        :align: center
        :width: 800px
        :alt: Speaker diarization pipeline- VAD, segmentation, speaker embedding extraction, clustering

To figure out "who spoke when", speaker diarization systems need to capture the characteristics of unseen speakers and tell apart which regions in the audio recording belong to which speaker. To achieve this, speaker diarization systems extract voice characteristics, count the number of speakers, then assign the audio segments to the corresponding speaker index.

Diarize speech with 3 lines of code
------------------------------------
After :ref:`installing NeMo<installation>`, you can diarize an audio file as follows:

.. code-block:: python

    from nemo.collections.asr.models import SortformerEncLabelModel
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1").eval()
    predicted_segments = diar_model.diarize(audio=["/path/to/your/audio.wav"], batch_size=1)

For help choosing the right model for your use case, see :doc:`Choosing a Model <../../starthere/choosing_a_model>`.

Types of Speaker Diarization Systems
-------------------------------------

.. image:: images/e2e_and_cascaded_diar_systems.png
        :align: center
        :width: 800px
        :alt: End-to-End and Cascaded Diar Systems

1. End-to-End Speaker Diarization System:

End-to-end speaker diarization systems pursue a much more simplified version of a system where a single neural network model accepts raw audio signals and outputs speaker activity for each audio frame. Therefore, end-to-end diarization models have an advantage in ease of optimization and deployments.

Currently, NeMo Speech AI provides the following end-to-end speaker diarization models:

- **Sortformer Diarizer** : A transformer-based model that estimates speaker labels from the given audio input giving the speaker indexes in arrival-time order.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Model
     - Type
     - HuggingFace Link
   * - diar_sortformer_4spk-v1
     - Offline
     - `nvidia/diar_sortformer_4spk-v1 <https://huggingface.co/nvidia/diar_sortformer_4spk-v1>`__
   * - diar_streaming_sortformer_4spk-v2
     - Streaming
     - `nvidia/diar_streaming_sortformer_4spk-v2 <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2>`__
   * - diar_streaming_sortformer_4spk-v2.1
     - Streaming
     - `nvidia/diar_streaming_sortformer_4spk-v2.1 <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>`__

2. Cascaded Speaker Diarization System:

Traditional cascaded (also referred to as modular or pipelined) speaker diarization systems consist of multiple modules such as voice activity detection (VAD), speaker embedding extraction, and clustering.
Cascaded speaker diarization systems are more challenging to optimize and deploy together but still have the advantage of fewer restrictions on the number of speakers and session length.

Cascaded NeMo Speech AI speaker diarization system consists of the following modules:

- **Voice Activity Detector (VAD)**: A trainable model which detects the presence or absence of speech to generate timestamps for speech activity from the given audio recording.

- **Speaker Embedding Extractor**: A trainable model that extracts speaker embedding vectors containing voice characteristics from raw audio signal.

- **Clustering Module**: A non-trainable module that groups speaker embedding vectors into a number of clusters.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Module
     - Model
     - Link
   * - Voice Activity Detector
     - Frame VAD Multilingual MarbleNet v2.0
     - `HuggingFace <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>`__ · `NGC <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet>`__
   * - Speaker Embedding Extractor
     - TitaNet-Large
     - `HuggingFace <https://huggingface.co/nvidia/speakerverification_en_titanet_large>`__ · `NGC <https://ngc.nvidia.com/catalog/models/nvidia:nemo:titanet_large>`__
   * - Clustering Module
     - Spectral Clustering
     - Non-trainable (no checkpoint)


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   api
   resources

