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


from functools import lru_cache

from nemo.collections.asr.inference.utils.constants import DEFAULT_SEMIOTIC_CLASS, SEP_REPLACEABLE_PUNCTUATION


@lru_cache(maxsize=5)
def get_translation_table(punct_marks_frozen: frozenset[str], sep: str) -> dict:
    """
    Create and cache translation table for text normalization.

    Args:
        punct_marks_frozen (frozenset[str]): Frozen set of punctuation marks to process
        sep (str): Separator to replace certain punctuation marks

    Returns:
        (dict) Translation table for str.translate()
    """
    replace_map = {mark: sep if mark in SEP_REPLACEABLE_PUNCTUATION else "" for mark in punct_marks_frozen}
    return str.maketrans(replace_map)


def normalize_text(text: str, punct_marks: set[str], sep: str) -> str:
    """
    Helper to normalize text by removing/replacing punctuation and lowercasing.

    Args:
        text (str): Text to normalize
        punct_marks (set[str]): Set of punctuation marks to process
        sep (str): Separator to replace certain punctuation marks

    Returns:
        (str) Normalized text
    """
    trans_table = get_translation_table(frozenset(punct_marks), sep)
    return text.translate(trans_table).lower()


def validate_init_params(
    text: str, start: float, end: float, conf: float, semiotic_class: str = None, strict: bool = False
) -> None:
    """
    Validate initialization parameters.
    Args:
        text: (str) Text to validate
        start: (float) Start time
        end: (float) End time
        conf: (float) Confidence score
        semiotic_class: (str) Semiotic class
        strict: (bool) Whether to strict validation
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(start, (int, float)):
        raise TypeError(f"start must be numeric, got {type(start).__name__}")
    if not isinstance(end, (int, float)):
        raise TypeError(f"end must be numeric, got {type(end).__name__}")
    if not isinstance(conf, (int, float)):
        raise TypeError(f"conf must be numeric, got {type(conf).__name__}")

    if semiotic_class is not None and not isinstance(semiotic_class, str):
        raise TypeError(f"semiotic_class must be a string, got {type(semiotic_class).__name__}")

    if strict:
        if start >= end:
            raise ValueError(f"start time ({start}) must be less than end time ({end})")
        if conf < 0 or conf > 1:
            raise ValueError(f"confidence ({conf}) must be between 0 and 1")


class TextSegment:
    """
    Text segment class.
    Represents a continuous text segment with a start time, end time, and confidence score.
    """

    __slots__ = ['_text', '_start', '_end', '_conf']

    def __init__(self, text: str, start: float, end: float, conf: float) -> None:
        """
        Initialize a TextSegment instance.

        Args:
            text: The content of the text segment
            start: Start time in seconds
            end: End time in seconds
            conf: Confidence score [0.0, 1.0]
        Raises:
            ValueError: If start >= end or if confidence is negative
            TypeError: If text is not a string
        """
        validate_init_params(text, start, end, conf, strict=True)

        self._text = text
        self._start = start
        self._end = end
        self._conf = conf

    @property
    def text(self) -> str:
        """The content of the text segment."""
        return self._text

    @property
    def start(self) -> float:
        """Start time of the text segment in seconds."""
        return self._start

    @property
    def end(self) -> float:
        """End time of the text segment in seconds."""
        return self._end

    @property
    def duration(self) -> float:
        """Duration of the text segment in seconds."""
        return self._end - self._start

    @property
    def conf(self) -> float:
        """Confidence score of the text segment."""
        return self._conf

    @text.setter
    def text(self, value: str) -> None:
        """Set the content of the text segment."""
        if not isinstance(value, str):
            raise TypeError(f"text must be a string, got {type(value).__name__}")
        self._text = value

    @start.setter
    def start(self, value: float) -> None:
        """Set the start time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"start time must be numeric, got {type(value).__name__}")
        self._start = value

    @end.setter
    def end(self, value: float) -> None:
        """Set the end time."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"end must be numeric, got {type(value).__name__}")
        self._end = value

    @conf.setter
    def conf(self, value: float) -> None:
        """Set the confidence score."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"conf must be numeric, got {type(value).__name__}")
        if value < 0 or value > 1:
            raise ValueError(f"confidence ({value}) must be between 0 and 1")
        self._conf = value

    def copy(self) -> 'TextSegment':
        """
        Create a deep copy of this TextSegment instance.

        Returns:
            A new TextSegment instance with identical properties
        """
        return TextSegment(text=self.text, start=self.start, end=self.end, conf=self.conf)

    def capitalize(self) -> None:
        """Capitalize first letter of the text segment."""
        self._text = self._text.capitalize()

    def with_normalized_text(self, punct_marks: set[str], sep: str = "") -> 'TextSegment':
        """
        Create a new TextSegment with normalized text (punctuation removed/replaced and lowercased).

        Args:
            punct_marks (set[str]): Set of punctuation marks to process
            sep: Separator to replace certain punctuation marks

        Returns:
            New TextSegment instance with normalized text
        """
        # Return new instance instead of modifying in place
        obj_copy = self.copy()
        obj_copy._text = normalize_text(self._text, punct_marks, sep)  # Direct access
        return obj_copy

    def normalize_text_inplace(self, punct_marks: set[str], sep: str = "") -> None:
        """
        Normalize text in place (punctuation removed/replaced and lowercased).

        Args:
            punct_marks (set[str]): Set of punctuation marks to process
            sep (str): Separator to replace certain punctuation marks

        Note:
            This method modifies the current instance. Consider using
            with_normalized_text() for a functional approach.
        """
        self._text = normalize_text(self._text, punct_marks, sep)  # Direct access

    def to_dict(self) -> dict:
        """
        Convert the TextSegment to a JSON-compatible dictionary.
        """
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "conf": self.conf,
        }


class Word(TextSegment):
    """
    Word class.
    Represents a word with a text, start time, end time, confidence score, and semiotic class.
    """

    __slots__ = ['_semiotic_class']

    def __init__(
        self, text: str, start: float, end: float, conf: float, semiotic_class: str = DEFAULT_SEMIOTIC_CLASS
    ) -> None:
        """
        Initialize a Word instance.

        Args:
            text: The text content of the word
            start: Start time in seconds
            end: End time in seconds
            conf: Confidence score [0.0, 1.0]
            semiotic_class: Semiotic class of the word

        Raises:
            ValueError: If start >= end or if confidence is negative
            TypeError: If text is not a string
        """
        validate_init_params(text, start, end, conf, semiotic_class, strict=True)
        super().__init__(text, start, end, conf)
        self._semiotic_class = semiotic_class

    @property
    def semiotic_class(self) -> str:
        """Semiotic class of the word."""
        return self._semiotic_class

    @semiotic_class.setter
    def semiotic_class(self, value: str) -> None:
        """Set the semiotic class."""
        if not isinstance(value, str):
            raise TypeError(f"semiotic_class must be a string, got {type(value).__name__}")
        self._semiotic_class = value

    def copy(self) -> 'Word':
        """
        Create a deep copy of this Word instance.

        Returns:
            A new Word instance with identical properties
        """
        return Word(text=self.text, start=self.start, end=self.end, conf=self.conf, semiotic_class=self.semiotic_class)

    def to_dict(self) -> dict:
        """
        Convert the Word to a JSON-compatible dictionary.
        """
        return super().to_dict() | {"semiotic_class": self.semiotic_class}


def join_segments(segments: list[list[TextSegment]], sep: str) -> list[str]:
    """
    Join the text segments to form transcriptions.

    Args:
        segments (list[list[TextSegment]]): List of text segment sequences to join
        sep (str): Separator to use when joining text segments

    Returns:
        List of transcriptions, one for each text segment sequence
    """
    return [sep.join([s.text for s in items]) for items in segments]


def normalize_segments_inplace(
    segments: list[TextSegment] | list[list[TextSegment]], punct_marks: set[str], sep: str = ' '
) -> None:
    """
    Normalize text in text segments by removing punctuation and converting to lowercase.

    This function modifies the text segments in-place by calling normalize_text_inplace
    on each TextSegment object. It handles both flat lists of text segments and nested lists.

    Args:
        segments (list[TextSegment] | list[list[TextSegment]]): List of TextSegment objects or list of lists of TextSegment objects
        punct_marks (set[str]): Set of punctuation marks to be processed
        sep (str): Separator to replace certain punctuation marks (default: ' ')

    Note:
        This function modifies the input text segments in-place. The original text
        content of the text segments will be permanently changed.
    """
    for item in segments:
        if isinstance(item, list):
            for segment in item:
                segment.normalize_text_inplace(punct_marks, sep)
        elif isinstance(item, TextSegment):
            item.normalize_text_inplace(punct_marks, sep)
        else:
            raise ValueError(f"Invalid item type: {type(item)}. Expected `TextSegment` or `List[TextSegment]`.")
