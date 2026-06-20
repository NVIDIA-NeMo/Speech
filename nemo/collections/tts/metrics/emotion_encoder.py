#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight LAION Empathic Insight Voice interface.

This script provides a Hugging Face-style Python class for LAION's
Empathic Insight Voice models without using ModelScope.

It supports:

  1. Restoring the Whisper encoder from Hugging Face.
  2. Restoring LAION classifier MLP heads from Hugging Face .pth files.
  3. Extracting the full Whisper encoder embedding:
       [B, 1500, 768]
  4. Extracting one classifier projection embedding:
       Small: [B, 64]
       Large: [B, 128]
  5. Extracting an SV-style emotion similarity embedding by concatenating
     multiple classifier projection embeddings:
       Small, 40 labels: [B, 40 * 64]   = [B, 2560]
       Large, 40 labels: [B, 40 * 128] = [B, 5120]
  6. Extracting an official-style emotion score vector:
       [B, num_labels]
  7. Computing ranked emotion predictions and cosine similarity.

Recommended for emotion similarity:

    model = EmpathicInsightVoice.from_pretrained(size="small", device="cuda")
    emb = model.extract_emotion_embedding("audio.wav", embedding_type="head_concat")
    sim = model.emotion_similarity("a.wav", "b.wav", embedding_type="head_concat")

Notes:

  - The official model is a collection of independent expert heads. Each head
    predicts one emotion or attribute score.
  - The "head_concat" embedding is an engineering adaptation for
    speaker-verification-style similarity. It concatenates the learned
    projection outputs from multiple classifier heads.
  - The "score_vector" embedding is closer to the documented inference output:
    a vector of raw emotion intensity scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# =============================================================================
# Label mapping
# =============================================================================

# MIRRORING-style 12 emotions (https://bench.theliva.ai/legacy/mirroring.html):
# Amusement, Anger, Elation, Impatience, Surprise,
# Emotional Numbness, Contemplation, Disappointment,
# Confusion, Pride, Affection, Sadness.
#
# The public-facing labels below are Python-friendly.
# Some labels map to LAION's original longer checkpoint names:
#   impatience -> model_Impatience_and_Irritability_best.pth
#   surprise   -> model_Astonishment_Surprise_best.pth

LAION_LABEL_TO_FILENAME: dict[str, str] = {
    "amusement": "model_Amusement_best.pth",
    "anger": "model_Anger_best.pth",
    "elation": "model_Elation_best.pth",
    "impatience": "model_Impatience_and_Irritability_best.pth",
    "surprise": "model_Astonishment_Surprise_best.pth",
    "emotional_numbness": "model_Emotional_Numbness_best.pth",
    "contemplation": "model_Contemplation_best.pth",
    "disappointment": "model_Disappointment_best.pth",
    "confusion": "model_Confusion_best.pth",
    "pride": "model_Pride_best.pth",
    "affection": "model_Affection_best.pth",
    "sadness": "model_Sadness_best.pth",
}


PRIMARY_EMOTION_LABELS: list[str] = [
    "amusement",
    "anger",
    "elation",
    "impatience",
    "surprise",
    "emotional_numbness",
    "contemplation",
    "disappointment",
    "confusion",
    "pride",
    "affection",
    "sadness",
]


AUXILIARY_SIMILARITY_LABELS: list[str] = []

# =============================================================================
# Model architecture specs
# =============================================================================

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "small": {
        "repo_id": "laion/Empathic-Insight-Voice-Small",
        "whisper_model_id": "mkrausio/EmoWhisper-AnS-Small-v0.1",
        "sample_rate": 16000,
        "max_audio_seconds": 30.0,
        "seq_len": 1500,
        "embed_dim": 768,
        "projection_dim": 64,
        "mlp_hidden_dims": [64, 32, 16],
        "mlp_dropouts": [0.0, 0.1, 0.1, 0.1],
    },
    "large": {
        "repo_id": "laion/Empathic-Insight-Voice-Large",
        "whisper_model_id": "mkrausio/EmoWhisper-AnS-Small-v0.1",
        "sample_rate": 16000,
        "max_audio_seconds": 30.0,
        "seq_len": 1500,
        "embed_dim": 768,
        "projection_dim": 128,
        "mlp_hidden_dims": [128, 64, 32],
        "mlp_dropouts": [0.0, 0.1, 0.1, 0.1],
    },
}


# =============================================================================
# MLP head
# =============================================================================

class FullEmbeddingMLP(nn.Module):
    """Classifier head used by Empathic Insight Voice.

    The model receives a full Whisper encoder sequence embedding:

        [batch, seq_len, embed_dim]

    For Empathic Insight Voice this is normally:

        [batch, 1500, 768]

    It then performs:

        flatten -> projection -> MLP -> scalar score

    The projection output is useful as an SV-style emotion embedding:

        Small: [batch, 64]
        Large: [batch, 128]

    Each restored classifier head has its own projection layer. Therefore,
    "anger" projection, "sadness" projection, and "arousal" projection are
    all different learned spaces.
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        projection_dim: int,
        mlp_hidden_dims: Sequence[int],
        mlp_dropout_rates: Sequence[float],
    ) -> None:
        super().__init__()

        if len(mlp_dropout_rates) != len(mlp_hidden_dims) + 1:
            raise ValueError(
                "Dropout rates length error. "
                f"Expected {len(mlp_hidden_dims) + 1}, "
                f"got {len(mlp_dropout_rates)}."
            )

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)

        layers: list[nn.Module] = [
            nn.ReLU(),
            nn.Dropout(mlp_dropout_rates[0]),
        ]

        current_dim = projection_dim
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout_rates[i + 1]),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def extract_projected_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the classifier projection embedding before the MLP.

        Args:
            x:
                Whisper embedding with shape [B, seq_len, embed_dim], or
                [B, 1, seq_len, embed_dim].

        Returns:
            Projected embedding with shape [B, projection_dim].
        """
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, T, C], got shape {tuple(x.shape)}."
            )

        return self.proj(self.flatten(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per input example."""
        projected = self.extract_projected_embedding(x)
        return self.mlp(projected)


# =============================================================================
# Main class
# =============================================================================

class EmpathicInsightVoice(nn.Module):
    """Lightweight Hugging Face-style Empathic Insight Voice class.

    This class intentionally does not inherit from NeMo ModelPT. It behaves
    like a normal PyTorch/Hugging Face utility model.

    Main methods:

      - extract_whisper_embedding(audio_path)
          Returns [1, 1500, 768].

      - extract_classifier_projection(audio_path, label)
          Returns one head-specific projection:
            Small: [1, 64]
            Large: [1, 128]

      - extract_emotion_embedding(audio_path, embedding_type="head_concat")
          Recommended SV-style emotion similarity embedding.

      - predict_emotions_from_embedding(embedding)
          Returns raw scores and ranked softmax-like top emotions.

      - compute(audio_path)
          Returns emotion predictions and optionally an embedding.

      - emotion_similarity(audio_path_a, audio_path_b)
          Returns cosine similarity between extracted emotion embeddings.
    """

    def __init__(
        self,
        size: str = "small",
        device: Union[str, torch.device] = "cuda",
        mlp_device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_classifiers: bool = True,
        load_all_classifiers: bool = False,
        top_k_emotions: int = 5,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        if size not in MODEL_SPECS:
            raise ValueError(
                f"Unsupported size={size!r}. Expected one of {sorted(MODEL_SPECS)}."
            )

        self.size = size
        self.spec = MODEL_SPECS[size]
        self.cache_classifiers = cache_classifiers
        self.top_k_emotions = top_k_emotions
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")

        self.device = requested_device
        self.mlp_device = (
            torch.device(mlp_device)
            if mlp_device is not None
            else self.device
        )

        self.sample_rate = int(self.spec["sample_rate"])
        self.max_audio_seconds = float(self.spec["max_audio_seconds"])

        # Load Whisper processor and encoder model.
        self.processor = WhisperProcessor.from_pretrained(
            self.spec["whisper_model_id"],
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            trust_remote_code=trust_remote_code,
        )

        whisper_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            whisper_kwargs["torch_dtype"] = torch_dtype

        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.spec["whisper_model_id"],
            **whisper_kwargs,
        ).to(self.device)

        self.whisper_model.eval()

        # Restored MLP heads are stored here.
        #
        # ModuleDict keys must be sanitized because labels can contain punctuation.
        self.classifiers = nn.ModuleDict()

        if load_all_classifiers:
            self.load_classifiers()

    @classmethod
    def from_pretrained(
        cls,
        size: str = "small",
        **kwargs: Any,
    ) -> "EmpathicInsightVoice":
        """Construct the model using Hugging Face checkpoints.

        Example:
            model = EmpathicInsightVoice.from_pretrained(
                size="small",
                device="cuda",
                mlp_device="cuda",
            )
        """
        return cls(size=size, **kwargs)

    @property
    def repo_id(self) -> str:
        """Hugging Face repo ID for the selected model size."""
        return str(self.spec["repo_id"])

    @property
    def available_labels(self) -> list[str]:
        """All labels known to this script."""
        return list(LAION_LABEL_TO_FILENAME.keys())

    @property
    def projection_dim(self) -> int:
        """Classifier projection dimension for the selected model size."""
        return int(self.spec["projection_dim"])

    # -------------------------------------------------------------------------
    # Audio and Whisper embedding extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_whisper_embedding(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Extract the full Whisper encoder embedding from an audio file.

        Args:
            audio_path:
                Path to an audio file readable by librosa.

        Returns:
            Tensor with shape [1, 1500, 768].
        """
        waveform, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        waveform = self._prepare_waveform(waveform)
        return self.extract_whisper_embedding_from_waveform(waveform)

    @torch.no_grad()
    def extract_whisper_embedding_from_waveform(
        self,
        waveform: np.ndarray,
    ) -> torch.Tensor:
        """Extract the full Whisper encoder embedding from a waveform.

        Args:
            waveform:
                Mono waveform at self.sample_rate.

        Returns:
            Tensor with shape [1, 1500, 768].
        """
        waveform = self._prepare_waveform(waveform)

        input_features = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features

        input_features = input_features.to(self.device)
        input_features = input_features.to(self.whisper_model.dtype)

        encoder_outputs = self.whisper_model.get_encoder()(
            input_features=input_features
        )

        embedding = encoder_outputs.last_hidden_state
        embedding = self._pad_or_trim_embedding(embedding)

        return embedding

    # -------------------------------------------------------------------------
    # Classifier projection extraction
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_classifier_projection(
        self,
        audio_path: Union[str, Path],
        label: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract one head-specific classifier projection embedding.

        This is the closest equivalent to extracting an x-vector-like embedding
        from one specific emotion classifier head.

        Flow:
            audio -> Whisper encoder -> label-specific classifier projection

        Args:
            audio_path:
                Input audio path.
            label:
                Label whose classifier projection should be used, for example:
                "anger", "sadness", "arousal".
            normalize:
                If True, apply L2 normalization.

        Returns:
            Small: [1, 64]
            Large: [1, 128]
        """
        whisper_embedding = self.extract_whisper_embedding(audio_path)
        return self.extract_classifier_projection_from_whisper_embedding(
            whisper_embedding=whisper_embedding,
            label=label,
            normalize=normalize,
        )

    @torch.no_grad()
    def extract_classifier_projection_from_whisper_embedding(
        self,
        whisper_embedding: torch.Tensor,
        label: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract one classifier projection from an existing Whisper embedding."""
        classifier = self._get_classifier(label)
        param = next(classifier.parameters())

        working_embedding = whisper_embedding.to(device=param.device, dtype=param.dtype)
        projected = classifier.extract_projected_embedding(working_embedding)

        projected = projected.float()
        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected

    # -------------------------------------------------------------------------
    # Emotion similarity embeddings
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def extract_emotion_embedding(
        self,
        audio_path: Union[str, Path],
        labels: Optional[Sequence[str]] = None,
        embedding_type: str = "head_concat",
        normalize: bool = True,
        include_auxiliary: bool = False,
    ) -> torch.Tensor:
        """Extract a fixed-dimensional emotion embedding.

        Recommended for SV-style emotion similarity:
            embedding_type="head_concat"

        Supported embedding types:

          1. "head_concat"
             Concatenate the projection output of each selected classifier head.

             Small, 40 primary labels:
                 [1, 40 * 64] = [1, 2560]

             Large, 40 primary labels:
                 [1, 40 * 128] = [1, 5120]

          2. "head_mean"
             Average the projection outputs across selected heads.

             Small:
                 [1, 64]

             Large:
                 [1, 128]

          3. "score_vector"
             Use raw scalar outputs from the selected classifier heads.

             Shape:
                 [1, num_labels]

             This is closest to the official annotation output.

        Args:
            audio_path:
                Input audio path.
            labels:
                Labels to use. If None, PRIMARY_EMOTION_LABELS are used.
            embedding_type:
                "head_concat", "head_mean", or "score_vector".
            normalize:
                If True, apply L2 normalization to the final embedding.
            include_auxiliary:
                If labels is None, append AUXILIARY_SIMILARITY_LABELS.

        Returns:
            torch.Tensor fixed-dimensional emotion embedding.
        """
        whisper_embedding = self.extract_whisper_embedding(audio_path)
        labels_to_run = self._default_similarity_labels(
            labels=labels,
            include_auxiliary=include_auxiliary,
        )

        return self.extract_emotion_embedding_from_whisper_embedding(
            whisper_embedding=whisper_embedding,
            labels=labels_to_run,
            embedding_type=embedding_type,
            normalize=normalize,
        )

    @torch.no_grad()
    def extract_emotion_embedding_from_whisper_embedding(
        self,
        whisper_embedding: torch.Tensor,
        labels: Optional[Sequence[str]] = None,
        embedding_type: str = "head_concat",
        normalize: bool = True,
        include_auxiliary: bool = False,
    ) -> torch.Tensor:
        """Extract a fixed-dimensional emotion embedding from Whisper features."""
        labels_to_run = self._default_similarity_labels(
            labels=labels,
            include_auxiliary=include_auxiliary,
        )

        if embedding_type == "score_vector":
            prediction = self.predict_emotions_from_embedding(
                embedding=whisper_embedding,
                labels=labels_to_run,
                return_raw_scores=True,
                rank_scores=False,
            )
            raw_scores = prediction["raw_scores"]

            output = torch.tensor(
                [raw_scores[label] for label in labels_to_run],
                dtype=torch.float32,
            ).unsqueeze(0)

        elif embedding_type in {"head_concat", "head_mean"}:
            projected_embeddings: list[torch.Tensor] = []

            for label in labels_to_run:
                classifier = self._get_classifier(label)
                param = next(classifier.parameters())

                working_embedding = whisper_embedding.to(
                    device=param.device,
                    dtype=param.dtype,
                )

                projected = classifier.extract_projected_embedding(working_embedding)
                projected_embeddings.append(projected.float().cpu())

            if embedding_type == "head_concat":
                output = torch.cat(projected_embeddings, dim=-1)
            else:
                output = torch.stack(projected_embeddings, dim=0).mean(dim=0)

        else:
            raise ValueError(
                f"Unsupported embedding_type={embedding_type!r}. "
                "Expected 'head_concat', 'head_mean', or 'score_vector'."
            )

        if normalize:
            output = F.normalize(output, p=2, dim=-1)

        return output

    @torch.no_grad()
    def emotion_similarity(
        self,
        audio_path_a: Union[str, Path],
        audio_path_b: Union[str, Path],
        labels: Optional[Sequence[str]] = None,
        embedding_type: str = "head_concat",
        include_auxiliary: bool = False,
    ) -> float:
        """Compute cosine similarity between two audios in emotion space.

        Args:
            audio_path_a:
                First audio path.
            audio_path_b:
                Second audio path.
            labels:
                Optional label subset.
            embedding_type:
                "head_concat", "head_mean", or "score_vector".
            include_auxiliary:
                If labels is None, append auxiliary labels.

        Returns:
            Cosine similarity as a Python float.
        """
        emb_a = self.extract_emotion_embedding(
            audio_path=audio_path_a,
            labels=labels,
            embedding_type=embedding_type,
            normalize=True,
            include_auxiliary=include_auxiliary,
        )
        emb_b = self.extract_emotion_embedding(
            audio_path=audio_path_b,
            labels=labels,
            embedding_type=embedding_type,
            normalize=True,
            include_auxiliary=include_auxiliary,
        )

        return float(F.cosine_similarity(emb_a, emb_b, dim=-1).item())

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def compare_emotion_pair(
        self,
        audio_path_a: Union[str, Path],
        audio_path_b: Union[str, Path],
        labels: Optional[Sequence[str]] = None,
        embedding_type: str = "score_vector",
    ) -> dict[str, Any]:
        """Compare two audio files using the 12-emotion set.

        This method does not perform corpus-level ranking. It only returns:

        - top emotion for audio A
        - top emotion for audio B
        - matched emotion label if both top emotions match
        - emotion similarity

        Args:
            audio_path_a:
                First audio file.
            audio_path_b:
                Second audio file.
            labels:
                Optional subset of labels. Defaults to PRIMARY_EMOTION_LABELS,
                which is the 12-emotion set.
            embedding_type:
                Similarity representation:
                - "score_vector": cosine over raw 12-emotion score vector.
                - "head_concat": cosine over concatenated classifier projections.
                - "head_mean": cosine over averaged classifier projections.

                For MIRRORING-style emotion-vector similarity, use "score_vector".

        Returns:
            {
                "audio_path_a": str,
                "audio_path_b": str,
                "audio_a_top_emotion": str | None,
                "audio_b_top_emotion": str | None,
                "top_emotion_match": bool ,
                "emotion_similarity": float,
                "audio_a_raw_scores": dict[str, float],
                "audio_b_raw_scores": dict[str, float],
            }
        """
        labels_to_run = self._validate_labels(labels or PRIMARY_EMOTION_LABELS)

        result_a = self.compute(
            audio_path=audio_path_a,
            labels=labels_to_run,
            return_embedding=False,
            return_raw_scores=True,
        )

        result_b = self.compute(
            audio_path=audio_path_b,
            labels=labels_to_run,
            return_embedding=False,
            return_raw_scores=True,
        )

        top_a = result_a["top_emotion"]
        top_b = result_b["top_emotion"]

        similarity = self.emotion_similarity(
            audio_path_a=audio_path_a,
            audio_path_b=audio_path_b,
            labels=labels_to_run,
            embedding_type=embedding_type,
            include_auxiliary=False,
        )

        return {
            "audio_path_a": str(audio_path_a),
            "audio_path_b": str(audio_path_b),
            "audio_a_top_emotion": top_a,
            "audio_b_top_emotion": top_b,
            "top_emotion_match": top_a is not None and top_a == top_b,
            "emotion_similarity": similarity,
            "audio_a_raw_scores": result_a["raw_scores"],
            "audio_b_raw_scores": result_b["raw_scores"],
        }

    @torch.no_grad()
    def compute(
        self,
        audio_path: Union[str, Path],
        labels: Optional[Sequence[str]] = None,
        return_embedding: bool = True,
        embedding_type: str = "head_concat",
        return_raw_scores: bool = True,
        include_auxiliary_for_embedding: bool = False,
    ) -> dict[str, Any]:
        """Compute emotion predictions and optionally an embedding.

        Args:
            audio_path:
                Input audio file.
            labels:
                Prediction labels. If None, all known labels are attempted.
                For the official 40-emotion profile, pass PRIMARY_EMOTION_LABELS.
            return_embedding:
                If True, return an emotion embedding.
            embedding_type:
                Embedding type to return:
                  "head_concat", "head_mean", or "score_vector".
            return_raw_scores:
                If True, return raw classifier outputs.
            include_auxiliary_for_embedding:
                If True and return_embedding=True, include auxiliary labels in the
                returned embedding.

        Returns:
            {
              "audio_path": str,
              "model_size": "small" | "large",
              "top_emotion": str | None,
              "emotions": {
                  label: {"score": float, "rank": int}
              },
              "raw_scores": {
                  label: float
              },
              "embedding": torch.Tensor,
              "embedding_type": str
            }
        """
        whisper_embedding = self.extract_whisper_embedding(audio_path)

        prediction = self.predict_emotions_from_embedding(
            embedding=whisper_embedding,
            labels=labels,
            return_raw_scores=return_raw_scores,
            rank_scores=True,
        )

        top_emotion = None
        if prediction["emotions"]:
            top_emotion = next(iter(prediction["emotions"]))

        output: dict[str, Any] = {
            "audio_path": str(audio_path),
            "model_size": self.size,
            "top_emotion": top_emotion,
            "emotions": prediction["emotions"],
        }

        if return_raw_scores:
            output["raw_scores"] = prediction["raw_scores"]

        if return_embedding:
            output["embedding"] = self.extract_emotion_embedding_from_whisper_embedding(
                whisper_embedding=whisper_embedding,
                labels=None,
                embedding_type=embedding_type,
                normalize=True,
                include_auxiliary=include_auxiliary_for_embedding,
            )
            output["embedding_type"] = embedding_type

        return output

    @torch.no_grad()
    def predict_emotions_from_embedding(
        self,
        embedding: torch.Tensor,
        labels: Optional[Sequence[str]] = None,
        return_raw_scores: bool = True,
        rank_scores: bool = True,
    ) -> dict[str, Any]:
        """Predict emotion or attribute scores from a Whisper embedding.

        Args:
            embedding:
                Whisper encoder embedding, normally [1, 1500, 768].
            labels:
                Labels to evaluate. If None, all known labels are attempted.
            return_raw_scores:
                If True, include raw classifier scores.
            rank_scores:
                If True, softmax and rank scores into top-k emotions.

        Returns:
            Dict containing:
                "emotions": ranked top-k scores if rank_scores=True
                "raw_scores": raw scalar outputs if return_raw_scores=True
        """
        labels_to_run = self._validate_labels(labels)

        raw_scores: dict[str, float] = {}

        for label in labels_to_run:
            classifier = self._get_classifier(label)
            param = next(classifier.parameters())

            working_embedding = embedding.to(device=param.device, dtype=param.dtype)
            score = classifier(working_embedding).detach().cpu().item()
            raw_scores[label] = float(score)

            if not self.cache_classifiers:
                cache_key = self._cache_key(label)
                if cache_key in self.classifiers:
                    del self.classifiers[cache_key]

        output: dict[str, Any] = {}

        if rank_scores:
            output["emotions"] = self._softmax_and_rank(raw_scores)
        else:
            output["emotions"] = {}

        if return_raw_scores:
            output["raw_scores"] = raw_scores

        return output

    # -------------------------------------------------------------------------
    # Classifier loading
    # -------------------------------------------------------------------------

    def load_classifiers(
        self,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Eagerly download and restore classifier MLPs.

        By default the class lazy-loads heads when needed. This method is useful
        when you want to pre-load selected heads before repeated inference.
        """
        labels_to_load = self._validate_labels(labels)
        for label in labels_to_load:
            self._get_classifier(label)

    def _get_classifier(self, label: str) -> FullEmbeddingMLP:
        """Download, reconstruct, and restore one classifier MLP head.

        This is where the classifier MLP is restored:

            classifier = FullEmbeddingMLP(...)
            state_dict = torch.load(...)
            state_dict = strip "_orig_mod." prefix if needed
            classifier.load_state_dict(state_dict)

        Args:
            label:
                Python-friendly label key, such as "anger" or "arousal".

        Returns:
            Restored FullEmbeddingMLP.
        """
        if label not in LAION_LABEL_TO_FILENAME:
            raise ValueError(
                f"Unknown label {label!r}. Available labels: "
                f"{sorted(LAION_LABEL_TO_FILENAME)}"
            )

        cache_key = self._cache_key(label)

        if cache_key in self.classifiers:
            classifier = self.classifiers[cache_key]
            if not isinstance(classifier, FullEmbeddingMLP):
                raise TypeError(
                    f"Cached classifier for {label!r} has unexpected type "
                    f"{type(classifier)}."
                )
            return classifier

        filename = LAION_LABEL_TO_FILENAME[label]

        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            repo_type="model",
        )

        classifier = FullEmbeddingMLP(
            seq_len=int(self.spec["seq_len"]),
            embed_dim=int(self.spec["embed_dim"]),
            projection_dim=int(self.spec["projection_dim"]),
            mlp_hidden_dims=list(self.spec["mlp_hidden_dims"]),
            mlp_dropout_rates=list(self.spec["mlp_dropouts"]),
        )

        state_dict = torch.load(local_path, map_location="cpu")

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Expected {local_path} to contain a state_dict, "
                f"but got {type(state_dict)}."
            )

        state_dict = self._strip_orig_mod_prefix_if_needed(state_dict)

        try:
            classifier.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load classifier for label={label!r} from {local_path}. "
                f"This often means the selected size={self.size!r} has different "
                "MLP dimensions than MODEL_SPECS declares."
            ) from exc

        classifier.eval()
        classifier = classifier.to(self.mlp_device)

        # Keep classifier dtype compatible with Whisper dtype if user loaded
        # Whisper in fp16/bf16.
        if self.whisper_model.dtype in (torch.float16, torch.bfloat16):
            classifier = classifier.to(dtype=self.whisper_model.dtype)

        if self.cache_classifiers:
            self.classifiers[cache_key] = classifier

        return classifier

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _validate_labels(
        self,
        labels: Optional[Sequence[str]],
    ) -> list[str]:
        """Validate label names and return a concrete list."""
        if labels is None:
            return list(LAION_LABEL_TO_FILENAME.keys())

        labels_list = list(labels)
        unknown = sorted(set(labels_list) - set(LAION_LABEL_TO_FILENAME.keys()))

        if unknown:
            raise ValueError(
                f"Unknown labels: {unknown}. "
                f"Available labels: {sorted(LAION_LABEL_TO_FILENAME.keys())}"
            )

        return labels_list

    def _default_similarity_labels(
        self,
        labels: Optional[Sequence[str]],
        include_auxiliary: bool,
    ) -> list[str]:
        """Choose the default labels for emotion similarity embeddings."""
        if labels is not None:
            return self._validate_labels(labels)

        labels_to_run = list(PRIMARY_EMOTION_LABELS)

        if include_auxiliary:
            labels_to_run.extend(AUXILIARY_SIMILARITY_LABELS)

        return self._validate_labels(labels_to_run)

    def _prepare_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Convert waveform to mono float32 and trim to max_audio_seconds."""
        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0).astype(np.float32)

        max_samples = int(self.sample_rate * self.max_audio_seconds)

        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        return waveform

    def _pad_or_trim_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Pad or trim Whisper encoder output to the expected sequence length."""
        seq_len = int(self.spec["seq_len"])
        embed_dim = int(self.spec["embed_dim"])

        if embedding.ndim != 3:
            raise RuntimeError(
                f"Expected Whisper embedding with shape [B, T, C], "
                f"got {tuple(embedding.shape)}."
            )

        if embedding.shape[-1] != embed_dim:
            raise RuntimeError(
                f"Unexpected embedding dim. Expected {embed_dim}, "
                f"got {embedding.shape[-1]}."
            )

        current_seq_len = embedding.shape[1]

        if current_seq_len < seq_len:
            padding = torch.zeros(
                (
                    embedding.shape[0],
                    seq_len - current_seq_len,
                    embed_dim,
                ),
                device=embedding.device,
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, padding], dim=1)

        elif current_seq_len > seq_len:
            embedding = embedding[:, :seq_len, :]

        return embedding

    def _softmax_and_rank(
        self,
        raw_scores: dict[str, float],
    ) -> dict[str, dict[str, Union[float, int]]]:
        """Convert raw scores to a sorted top-k softmax dictionary.

        The raw MLP outputs are independent regression scores. This method is
        mainly for producing an easy top emotion label. For similarity, prefer
        raw score vectors or projection embeddings.
        """
        if not raw_scores:
            return {}

        labels = list(raw_scores.keys())
        values = np.array([raw_scores[label] for label in labels], dtype=np.float32)

        values = values - np.max(values)
        exp_values = np.exp(values)
        probs = exp_values / np.sum(exp_values)

        ranked = sorted(
            zip(labels, probs.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

        ranked = ranked[: self.top_k_emotions]

        return {
            label: {
                "score": float(prob),
                "rank": rank,
            }
            for rank, (label, prob) in enumerate(ranked, start=1)
        }

    @staticmethod
    def _strip_orig_mod_prefix_if_needed(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Strip torch.compile '_orig_mod.' prefixes if present."""
        if not any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            return state_dict

        return {
            key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key: value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _cache_key(label: str) -> str:
        """Convert an arbitrary label into a safe ModuleDict key."""
        return (
            label.replace(".", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("&", "and")
        )

    def cleanup(self) -> None:
        """Move modules to CPU and clear classifier cache."""
        for key in list(self.classifiers.keys()):
            self.classifiers[key].cpu()

        self.classifiers.clear()
        self.whisper_model.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# CLI utilities
# =============================================================================

def _tensor_info(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _parse_labels(labels: Optional[str]) -> Optional[list[str]]:
    if labels is None or labels.strip() == "":
        return None

    return [item.strip() for item in labels.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LAION Empathic Insight Voice embeddings and similarity."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Input audio path.",
    )
    parser.add_argument(
        "--audio-b",
        type=str,
        default=None,
        help="Optional second audio path for similarity.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Model size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Whisper encoder.",
    )
    parser.add_argument(
        "--mlp-device",
        type=str,
        default=None,
        help="Device for MLP classifier heads. Defaults to --device.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="head_concat",
        choices=["head_concat", "head_mean", "score_vector"],
        help="Emotion embedding type.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "Comma-separated labels to use. "
            "Example: anger,sadness,arousal. "
            "If omitted, primary emotion labels are used for similarity."
        ),
    )
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Include auxiliary similarity labels when --labels is omitted.",
    )
    parser.add_argument(
        "--load-all-classifiers",
        action="store_true",
        help="Eagerly load all known classifiers at startup.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked emotions to return.",
    )

    args = parser.parse_args()

    labels = _parse_labels(args.labels)

    model = EmpathicInsightVoice.from_pretrained(
        size=args.size,
        device=args.device,
        mlp_device=args.mlp_device,
        cache_dir=args.cache_dir,
        cache_classifiers=True,
        load_all_classifiers=args.load_all_classifiers,
        top_k_emotions=args.top_k,
    )

    result = model.compute(
        audio_path=args.audio,
        labels=labels,
        return_embedding=True,
        embedding_type=args.embedding_type,
        return_raw_scores=True,
        include_auxiliary_for_embedding=args.include_auxiliary,
    )

    printable: dict[str, Any] = {
        "audio_path": result["audio_path"],
        "model_size": result["model_size"],
        "top_emotion": result["top_emotion"],
        "embedding_type": result["embedding_type"],
        "embedding": _tensor_info(result["embedding"]),
        "emotions": result["emotions"],
        "raw_scores": result["raw_scores"],
    }

    if args.audio_b is not None:
        printable["audio_b"] = args.audio_b
        printable["similarity"] = model.emotion_similarity(
            audio_path_a=args.audio,
            audio_path_b=args.audio_b,
            labels=labels,
            embedding_type=args.embedding_type,
            include_auxiliary=args.include_auxiliary,
        )
    
    result = model.compare_emotion_pair(
        audio_path_a=args.audio,
        audio_path_b=args.audio_b,
        embedding_type="head_concat",
    )

    result_score_vector = model.compare_emotion_pair(
        audio_path_a=args.audio,
        audio_path_b=args.audio_b,
        embedding_type="score_vector",
    )

    result_score_mean = model.compare_emotion_pair(
        audio_path_a=args.audio,
        audio_path_b=args.audio_b,
        embedding_type="head_mean",
    )

    print("embedding_type=head_concat")
    print(result["audio_a_top_emotion"])
    print(result["audio_b_top_emotion"])
    print(result["top_emotion_match"])
    print(result["emotion_similarity"])


    print("embedding_type=score_vector")
    print(result_score_vector["audio_a_top_emotion"])
    print(result_score_vector["audio_b_top_emotion"])
    print(result_score_vector["top_emotion_match"])
    print(result_score_vector["emotion_similarity"])

    print("embedding_type=head_mean")
    print(result_score_mean["audio_a_top_emotion"])
    print(result_score_mean["audio_b_top_emotion"])
    print(result_score_mean["top_emotion_match"])
    print(result_score_mean["emotion_similarity"])

if __name__ == "__main__":
    main()