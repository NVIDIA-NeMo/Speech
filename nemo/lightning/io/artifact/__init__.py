from nemo.lightning.io.artifact.base import Artifact
from nemo.lightning.io.artifact.file import DirArtifact, DirOrStringArtifact, FileArtifact, PathArtifact
from nemo.lightning.io.artifact.hf_auto import HFAutoArtifact

__all__ = ["Artifact", "FileArtifact", "PathArtifact", "DirArtifact", "DirOrStringArtifact", "HFAutoArtifact"]
