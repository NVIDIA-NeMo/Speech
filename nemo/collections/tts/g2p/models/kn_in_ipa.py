# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pathlib
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Union

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    get_grapheme_character_set,
    get_ipa_punctuation_list,
)
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.utils import set_grapheme_case
from nemo.utils import logging


class KannadaG2p(BaseG2p):
    """Kannada Grapheme-to-Phoneme (G2P) conversion module.

    This module converts Kannada text to IPA phoneme sequences using a hybrid approach:
    1. Dictionary lookup for known words
    2. Rule-based conversion for unknown words (OOV)

    Kannada script is an abugida where consonants carry an inherent 'a' vowel
    that can be modified by dependent vowel signs (matras) or suppressed by virama (್).

    Example:
        >>> g2p = KannadaG2p(phoneme_dict="kannada_lexicon.txt")
        >>> g2p("ಕನ್ನಡ")
        ['k', 'a', 'n', 'n', 'a', 'ɖ', 'a']
    """

    def __init__(
        self,
        phoneme_dict: Optional[Union[str, pathlib.Path, Dict[str, List[str]]]] = None,
        phoneme_prefix: str = "",
        grapheme_prefix: str = "",
        grapheme_case: str = "lower",
        word_tokenize_func=None,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        """Initialize Kannada G2P module.

        Args:
            phoneme_dict: Path to Kannada pronunciation dictionary file or a dict object.
                Format: word<TAB>pronunciation (IPA characters without spaces)
            phoneme_prefix: Prefix to prepend to phoneme symbols to distinguish from graphemes.
                Default is "" (no prefix).
            grapheme_prefix: Prefix to prepend to graphemes (ASCII letters in code-mixed text)
                to distinguish them from phonemes. Default is "" (no prefix).
            grapheme_case: Case for graphemes: "upper", "lower", or "mixed".
                Default is "lower".
            word_tokenize_func: Custom function for tokenizing text into words.
                Should return List[Tuple[Union[str, List[str]], bool]].
            apply_to_oov_word: Custom function to apply to out-of-vocabulary words.
                If None, rule-based G2P is used.
            mapping_file: Optional path to character mapping file.
        """
        if phoneme_prefix is None:
            phoneme_prefix = ""
        if grapheme_prefix is None:
            grapheme_prefix = ""

        self.phoneme_prefix = phoneme_prefix
        self.grapheme_prefix = grapheme_prefix
        self.grapheme_case = grapheme_case

        # Load phoneme dictionary if provided
        if phoneme_dict is not None:
            if isinstance(phoneme_dict, (str, pathlib.Path)):
                phoneme_dict = self._parse_phoneme_dict(phoneme_dict, phoneme_prefix)
            else:
                # Normalize dict input: split string pronunciations into character lists
                phoneme_dict = self._normalize_phoneme_dict(phoneme_dict, phoneme_prefix)
            self.phoneme_list = sorted({pron for prons in phoneme_dict.values() for pron in prons})
        else:
            phoneme_dict = {}
            self.phoneme_list = []

        # Grapheme handling for code-mixed text (Kannada + English)
        self.grapheme_dict = {
            x: grapheme_prefix + x
            for x in get_grapheme_character_set(locale="en-US", case=grapheme_case)
        }
        self.grapheme_list = sorted(self.grapheme_dict)

        # Punctuation set
        self.punctuation = get_ipa_punctuation_list('kn-IN')

        # Kannada grapheme character set from ipa_lexicon
        self.kannada_grapheme_set = set(get_grapheme_character_set(locale="kn-IN"))

        # Initialize Kannada phonological rules
        self._init_kannada_rules()

        if apply_to_oov_word is None:
            logging.info(
                "apply_to_oov_word=None. Using rule-based G2P for out-of-vocabulary words."
            )

        super().__init__(
            phoneme_dict=phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

        # Build symbols set for IPATokenizer compatibility
        self._build_symbols()

    def _build_symbols(self):
        """Build the symbols set containing all valid graphemes and phonemes.

        This is required for compatibility with IPATokenizer which uses g2p.symbols.
        """
        from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
            IPA_CHARACTER_SETS,
        )

        symbols = set()
        prefix = self.phoneme_prefix

        # Add Kannada graphemes
        symbols.update(self.kannada_grapheme_set)

        # Add IPA phonemes from the character set (with prefix if set)
        for char in IPA_CHARACTER_SETS.get("kn-IN", ()):
            symbols.add(prefix + char)

        # Add phonemes from dictionary (already prefixed during parsing)
        symbols.update(self.phoneme_list)

        # Add graphemes for code-mixed text (use dict values which include prefix)
        symbols.update(self.grapheme_dict.values())

        # Add punctuation
        symbols.update(self.punctuation)

        # Add ASCII digits (emitted by G2P for both Kannada and ASCII digits)
        symbols.update('0123456789')

        self.symbols = symbols

    @staticmethod
    def _normalize_phoneme_dict(
        phoneme_dict: Dict[str, List[str]],
        phoneme_prefix: str
    ) -> Dict[str, List[str]]:
        """Normalize a dict-provided phoneme dictionary.

        Supports two input formats:
        1. {"word": ["phonemestring"]} - string gets split into characters
        2. {"word": ["p", "h", "o", ...]} - already flat, used as-is

        Args:
            phoneme_dict: Dictionary mapping words to pronunciations.
            phoneme_prefix: Prefix to add to each phoneme character.

        Returns:
            Normalized dictionary with pronunciations as flat character lists.
        """
        normalized = {}
        for word, prons in phoneme_dict.items():
            if not isinstance(prons, list) or len(prons) == 0:
                normalized[word] = prons
                continue

            # Detect format: flat list of tokens vs list containing pronunciation string(s)
            # Flat format: ["k", "a", "n", ...] - all single chars
            # String format: ["kannaɖa"] - one or more multi-char strings
            is_flat_token_list = all(
                isinstance(p, str) and len(p) == 1 for p in prons
            )

            if is_flat_token_list:
                # Already flat list of single-char tokens, just apply prefix
                if phoneme_prefix:
                    normalized[word] = [phoneme_prefix + p for p in prons]
                else:
                    normalized[word] = prons
            else:
                # List contains pronunciation string(s), take first and split
                pron = prons[0]
                if isinstance(pron, str):
                    normalized[word] = [phoneme_prefix + char for char in pron]
                else:
                    normalized[word] = prons

        return normalized

    def replace_symbols(self, symbols, keep_alternate=True):
        """Replace the vocabulary of symbols and filter entries with illegal symbols.

        This method is required for compatibility with IPATokenizer's fixed_vocab feature.

        Args:
            symbols: User-provided set of valid symbols (graphemes and phonemes).
            keep_alternate: Unused, kept for API compatibility with IpaG2p.
        """
        new_symbols = set(symbols)

        # Filter phoneme dictionary entries
        deletion_words = []

        for word, prons in self.phoneme_dict.items():
            # Check for illegal graphemes in the word
            word_graphemes = set(word)
            if word_graphemes - new_symbols:
                deletion_words.append(word)
                continue

            # Check for illegal phonemes in pronunciation
            # prons is a flat list of phoneme tokens (possibly prefixed like "#k")
            # Check each token as a whole, not character by character
            pron_set = set(prons)
            if pron_set - new_symbols:
                deletion_words.append(word)

        # Update dictionary
        for word in deletion_words:
            del self.phoneme_dict[word]

        self.symbols = new_symbols

    def _init_kannada_rules(self):
        """Initialize Kannada grapheme-to-phoneme mapping rules based on Kannada phonology."""

        # Independent vowels (Swaras)
        self.vowel_map = {
            'ಅ': 'a',
            'ಆ': 'aː',
            'ಇ': 'i',
            'ಈ': 'iː',
            'ಉ': 'u',
            'ಊ': 'uː',
            'ಋ': 'ɾɯ',
            'ೠ': 'ɾɯː',
            'ಌ': 'lu',
            'ೡ': 'luː',
            'ಎ': 'e',
            'ಏ': 'eː',
            'ಐ': 'ai',
            'ಒ': 'o',
            'ಓ': 'oː',
            'ಔ': 'au',
        }

        # Dependent vowel signs (Matras) - modify the inherent vowel
        self.matra_map = {
            'ಾ': 'aː',
            'ಿ': 'i',
            'ೀ': 'iː',
            'ು': 'u',
            'ೂ': 'uː',
            'ೃ': 'ɾɯ',
            'ೄ': 'ɾɯː',
            'ೆ': 'e',
            'ೇ': 'eː',
            'ೈ': 'ai',
            'ೊ': 'o',
            'ೋ': 'oː',
            'ೌ': 'au',
        }

        # Consonants (Vyanjanas) - base form without inherent vowel
        self.consonant_map = {
            # Velar (kanthya)
            'ಕ': 'k',
            'ಖ': 'kʰ',
            'ಗ': 'g',
            'ಘ': 'gʰ',
            'ಙ': 'ŋ',
            # Palatal (talavya)
            'ಚ': 'tʃ',
            'ಛ': 'tʃʰ',
            'ಜ': 'dʒ',
            'ಝ': 'dʒʰ',
            'ಞ': 'ɲ',
            # Retroflex (murdhanya)
            'ಟ': 'ʈ',
            'ಠ': 'ʈʰ',
            'ಡ': 'ɖ',
            'ಢ': 'ɖʰ',
            'ಣ': 'ɳ',
            # Dental (dantya)
            'ತ': 't',
            'ಥ': 'tʰ',
            'ದ': 'd',
            'ಧ': 'dʰ',
            'ನ': 'n',
            # Labial (oshthya)
            'ಪ': 'p',
            'ಫ': 'pʰ',
            'ಬ': 'b',
            'ಭ': 'bʰ',
            'ಮ': 'm',
            # Approximants and liquids
            'ಯ': 'j',
            'ರ': 'ɾ',
            'ಱ': 'ɾ',  # Archaic 'rra', often pronounced same as 'ra'
            'ಲ': 'l',
            'ಳ': 'ɭ',  # Retroflex lateral
            'ೞ': 'ɻ',  # Archaic retroflex approximant
            'ವ': 'ʋ',
            # Sibilants and aspirate
            'ಶ': 'ʃ',
            'ಷ': 'ʂ',
            'ಸ': 's',
            'ಹ': 'h',
        }

        # Special marks
        self.virama = '್'  # Halant - suppresses inherent vowel
        self.anusvara = 'ಂ'  # Nasalization
        self.visarga = 'ಃ'  # Voiceless glottal fricative
        self.avagraha = 'ಽ'  # Indicates elision

        # Nukta (for borrowed sounds) - currently not widely used in Kannada
        self.nukta = '಼'

        # All matras for checking
        self.all_matras = set(self.matra_map.keys())

        # Inherent vowel (schwa in Kannada is typically 'a')
        self.inherent_vowel = 'a'

    def _split_phoneme(self, phoneme: str, prefix: str) -> List[str]:
        """Split multi-character phonemes into separate tokens for consistency.

        Splits multi-character phonemes into individual characters for consistent tokenization.

        Args:
            phoneme: The phoneme string to potentially split.
            prefix: Prefix to add to each token.

        Returns:
            List of prefixed phoneme tokens.
        """
        # Split phonemes character-by-character
        return [prefix + char for char in phoneme]

    @staticmethod
    def _parse_phoneme_dict(
        phoneme_dict_path: Union[str, pathlib.Path],
        phoneme_prefix: str
    ) -> Dict[str, List[str]]:
        """Load pronunciation dictionary file.

        Args:
            phoneme_dict_path: Path to the dictionary file.
            phoneme_prefix: Prefix to add to each phoneme.

        Returns:
            Dictionary mapping words to their phoneme sequences.

        File format:
            word<TAB>pronunciation (IPA characters without spaces)
            Lines starting with ;;; are comments.
        """
        g2p_dict = defaultdict(list)
        with open(phoneme_dict_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith(";;;"):
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    logging.warning(f"Skipping malformed line {line_num}: {line}")
                    continue

                word = parts[0]
                pronunciation = parts[1]

                # Add prefix to each character
                pronunciation_with_prefix = [phoneme_prefix + pron for pron in pronunciation]
                g2p_dict[word] = pronunciation_with_prefix

        logging.info(f"Loaded {len(g2p_dict)} entries from Kannada phoneme dictionary")
        return g2p_dict

    def _is_kannada_char(self, char: str) -> bool:
        """Check if a character is in the Kannada grapheme character set."""
        if not char:
            return False
        return char in self.kannada_grapheme_set

    def _get_anusvara_phoneme(self, next_consonant: Optional[str] = None) -> str:
        """Get the appropriate nasal phoneme for anusvara based on following consonant.

        Anusvara assimilates to the place of articulation of the following consonant.
        """
        if next_consonant is None:
            return 'm'  # Default to bilabial nasal

        # Map consonants to their corresponding nasal by place of articulation
        if next_consonant in ['ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ']:
            return 'ŋ'  # Velar nasal
        elif next_consonant in ['ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ']:
            return 'ɲ'  # Palatal nasal
        elif next_consonant in ['ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ']:
            return 'ɳ'  # Retroflex nasal
        elif next_consonant in ['ತ', 'ಥ', 'ದ', 'ಧ', 'ನ']:
            return 'n'  # Dental nasal
        elif next_consonant in ['ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ']:
            return 'm'  # Bilabial nasal
        else:
            return 'm'  # Default

    def _rule_based_g2p(self, text: str) -> List[str]:
        """Convert Kannada text to phonemes using rule-based approach.

        This handles the aksara (syllabic) structure of Kannada:
        - Consonant + inherent vowel 'a'
        - Consonant + matra (dependent vowel)
        - Consonant + virama (no vowel)
        - Consonant clusters (consonant + virama + consonant)

        Args:
            text: Kannada text to convert.

        Returns:
            List of IPA phoneme symbols.
        """
        phonemes = []
        chars = list(text)
        i = 0
        prefix = self.phoneme_prefix

        while i < len(chars):
            char = chars[i]

            # Handle independent vowels
            if char in self.vowel_map:
                phoneme = self.vowel_map[char]
                phonemes.extend(self._split_phoneme(phoneme, prefix))
                i += 1
                continue

            # Handle consonants
            if char in self.consonant_map:
                consonant = self.consonant_map[char]
                i += 1

                # Look ahead for modifiers
                has_vowel = False
                while i < len(chars):
                    next_char = chars[i]

                    # Virama (halant) - consonant cluster or final consonant
                    if next_char == self.virama:
                        phonemes.extend(self._split_phoneme(consonant, prefix))
                        i += 1
                        has_vowel = True  # Virama means no inherent vowel
                        break

                    # Matra (dependent vowel)
                    elif next_char in self.matra_map:
                        phonemes.extend(self._split_phoneme(consonant, prefix))
                        vowel = self.matra_map[next_char]
                        phonemes.extend(self._split_phoneme(vowel, prefix))
                        i += 1
                        has_vowel = True
                        break

                    # Anusvara
                    elif next_char == self.anusvara:
                        phonemes.extend(self._split_phoneme(consonant, prefix))
                        phonemes.append(prefix + self.inherent_vowel)
                        # Look ahead for following consonant to determine nasal place
                        next_cons = chars[i + 1] if i + 1 < len(chars) else None
                        nasal = self._get_anusvara_phoneme(next_cons)
                        phonemes.append(prefix + nasal)
                        i += 1
                        has_vowel = True
                        break

                    # Visarga
                    elif next_char == self.visarga:
                        phonemes.extend(self._split_phoneme(consonant, prefix))
                        phonemes.append(prefix + self.inherent_vowel)
                        phonemes.append(prefix + 'h')
                        i += 1
                        has_vowel = True
                        break

                    else:
                        # No modifier found, break the loop
                        break

                # If no vowel modifier was found, add inherent vowel
                if not has_vowel:
                    phonemes.extend(self._split_phoneme(consonant, prefix))
                    phonemes.append(prefix + self.inherent_vowel)

                continue

            # Handle standalone anusvara (rare, but possible)
            if char == self.anusvara:
                next_cons = chars[i + 1] if i + 1 < len(chars) else None
                nasal = self._get_anusvara_phoneme(next_cons)
                phonemes.append(prefix + nasal)
                i += 1
                continue

            # Handle standalone visarga
            if char == self.visarga:
                phonemes.append(prefix + 'h')
                i += 1
                continue

            # Handle ASCII letters (code-mixed text)
            if char.upper() in self.grapheme_dict or char.lower() in self.grapheme_dict:
                processed_char = set_grapheme_case(char, case=self.grapheme_case)
                if processed_char in self.grapheme_dict:
                    phonemes.append(self.grapheme_dict[processed_char])
                else:
                    phonemes.append(processed_char)
                i += 1
                continue

            # Handle Kannada digits (must check before isdigit() as isdigit() is True for Kannada digits)
            kannada_digits = '೦೧೨೩೪೫೬೭೮೯'
            if char in kannada_digits:
                # Convert to Arabic numeral
                phonemes.append(str(kannada_digits.index(char)))
                i += 1
                continue

            # Handle ASCII digits (pass through)
            if char.isascii() and char.isdigit():
                phonemes.append(char)
                i += 1
                continue

            # Handle punctuation
            if char in self.punctuation:
                phonemes.append(char)
                i += 1
                continue

            # Handle whitespace
            if char.isspace():
                phonemes.append(' ')
                i += 1
                continue

            # Handle avagraha (elision marker) - typically silent
            if char == self.avagraha:
                i += 1
                continue

            # Unknown character - pass through with warning
            if self._is_kannada_char(char):
                logging.debug(f"Unknown Kannada character: {char} (U+{ord(char):04X})")
            phonemes.append(char)
            i += 1

        return phonemes

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization for Kannada text.

        Splits on whitespace and keeps punctuation as separate tokens.
        """
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        # Split on whitespace while preserving punctuation as separate tokens
        tokens = []
        current_token = []

        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                tokens.append(' ')
            elif char in self.punctuation:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                tokens.append(char)
            else:
                current_token.append(char)

        if current_token:
            tokens.append(''.join(current_token))

        return tokens

    def __call__(self, text: str) -> List[str]:
        """Convert Kannada text to IPA phoneme sequence.

        Args:
            text: Input text in Kannada (may include English/numbers).

        Returns:
            List of IPA phoneme symbols.

        Example:
            >>> g2p = KannadaG2p()
            >>> g2p("ನಮಸ್ಕಾರ")
            ['n', 'a', 'm', 'a', 's', 'k', 'aː', 'r', 'a']
        """
        # Normalize unicode representation
        text = unicodedata.normalize('NFC', text)

        # Apply case transformation for ASCII letters
        text = set_grapheme_case(text, case=self.grapheme_case)

        # Tokenize into words
        if self.word_tokenize_func is not None:
            # Custom tokenizer returns List[Tuple[Union[str, List[str]], bool]]
            # where bool (without_changes) indicates whether to pass through unchanged
            words_and_flags = self.word_tokenize_func(text)
        else:
            # Default tokenizer returns List[str], convert to expected format
            # without_changes=False means process the word
            words_and_flags = [([token], False) for token in self._tokenize(text)]

        phoneme_seq = []
        for words, without_changes in words_and_flags:
            if without_changes:
                # Pass through unchanged: prefix ASCII letters (case-normalized), leave others as-is
                for word in words:
                    for char in word:
                        normalized_char = set_grapheme_case(char, case=self.grapheme_case)
                        if normalized_char in self.grapheme_dict:
                            phoneme_seq.append(self.grapheme_dict[normalized_char])
                        else:
                            phoneme_seq.append(char)
                continue

            # Process the word(s)
            assert len(words) == 1, f"{words} should have single item when without_changes is False"
            token = words[0]

            # Skip whitespace tokens
            if token.isspace():
                phoneme_seq.append(' ')
                continue

            # Skip punctuation
            if token in self.punctuation:
                phoneme_seq.append(token)
                continue

            # Try dictionary lookup first
            if token in self.phoneme_dict:
                phoneme_seq.extend(self.phoneme_dict[token])
                continue

            # Try custom OOV handler
            if self.apply_to_oov_word is not None:
                result = self.apply_to_oov_word(token)
                if result:
                    phoneme_seq.extend(result)
                    continue

            # Use rule-based G2P
            token_phonemes = self._rule_based_g2p(token)
            phoneme_seq.extend(token_phonemes)

        return phoneme_seq
