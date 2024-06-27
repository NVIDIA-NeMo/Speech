from nemo.lightning.io.artifact import FileArtifact
from nemo.lightning.io.mixin import track_io

__all__ = []

try:
    from nemo.collections.common.tokenizers import AutoTokenizer

    track_io(
        AutoTokenizer,
        artifacts=[
            FileArtifact("vocab_file"),
            FileArtifact("merges_file"),
        ],
    )
    __all__.append("AutoTokenizer")
except ImportError:
    pass


try:
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    track_io(SentencePieceTokenizer, artifacts=[FileArtifact("model_path")])
    __all__.append("SentencePieceTokenizer")
except ImportError:
    pass
