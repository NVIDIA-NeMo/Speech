
Resource and Documentation Guide
--------------------------------

Hands-on speaker diarization tutorial notebooks for both end-to-end and cascaded systems can be found under ``<NeMo_root>/tutorials/speaker_tasks``.

There are also tutorials for performing both end-to-end speaker diarization cascaded speaker diarization. 
We also provide tutorials about getting ASR transcriptions combined with speaker labels along with voice activity timestamps with NeMo ASR collections.

Most of the tutorials can be run on Google Colab by specifying the link to the notebooks' GitHub pages on Colab.

If you are looking for information about a particular model used for speaker diarization inference, or would like to find out more about the model
architectures available in the `nemo_asr` collection, check out the :doc:`Models <./models>` page.

Documentation on dataset preprocessing can be found on the :doc:`Datasets <./datasets>` page.
NeMo includes preprocessing scripts for several common ASR datasets, and this page contains instructions on running
those scripts.
It also includes guidance for creating your own NeMo-compatible dataset, if you have your own data.

Information about how to load model checkpoints (either local files or pretrained ones from NGC), perform inference, as well as a list
of the checkpoints available on NGC are located on the :doc:`Checkpoints <./results>` page.

Documentation for configuration files specific to the ``nemo_asr`` models can be found on the
:doc:`Configuration Files <./configs>` page.
