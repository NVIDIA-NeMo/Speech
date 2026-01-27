# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


# fmt: off

SUPPORTED_LOCALES = ["en-US", "de-DE", "es-ES", "it-IT", "fr-FR", "vi-VN", "ja-JP", "hi-IN"]

DEFAULT_PUNCTUATION = (
    ',', '.', '!', '?', '-',
    ':', ';', '/', '"', '(',
    ')', '[', ']', '{', '}',
)

VITS_PUNCTUATION = (
    ',', '.', '!', '?', '-',
    ':', ';', '"', '«', '»',
    '“', '”', '¡', '¿', '—', 
    '…',
)

GRAPHEME_CHARACTER_SETS = {
    "en-US": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ),
    "es-ES": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Á', 'É', 'Í', 'Ñ',
        'Ó', 'Ú', 'Ü'
    ),
    # ref: https://en.wikipedia.org/wiki/German_orthography#Alphabet
    "de-DE": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö', 'Ü', 'ẞ',
    ),
    # ref: https://en.wikipedia.org/wiki/Vietnamese_alphabet
    "vi-VN": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Đ', 'Á', 'À', 'Ã', 
        'Ả', 'Ạ', 'Ă', 'Ắ', 'Ằ', 'Ẵ', 'Ẳ', 'Ặ', 'Â', 'Ấ', 
        'Ầ', 'Ẫ', 'Ẩ', 'Ậ', 'Ó', 'Ò', 'Õ', 'Ỏ', 'Ọ', 'Ô', 
        'Ố', 'Ồ', 'Ỗ', 'Ổ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ỡ', 'Ở', 
        'Ợ', 'É', 'È', 'Ẽ', 'Ẻ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ễ', 
        'Ể', 'Ệ', 'Ú', 'Ù', 'Ũ', 'Ủ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 
        'Ữ', 'Ử', 'Ự', 'Í', 'Ì', 'Ĩ', 'Ỉ', 'Ị', 'Ý', 'Ỳ', 
        'Ỹ', 'Ỷ', 'Ỵ',
    ),
    "fr-FR": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Â', 'Ä', 'Æ', 
        'Ç', 'È', 'É', 'Ê', 'Ë', 'Í', 'Î', 'Ï', 'Ñ', 'Ô', 
        'Ö', 'Ù', 'Û', 'Ü', 'Ō', 'Œ',
    ),
    "it-IT": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'È', 'É', 'Ì',
        'Ò', 'Ù'
    ),
    "ja-JP": (
        # Katakana basic
        'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
        'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
        'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
        'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ',
        'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン',
        # Dakuten / Handakuten
        'ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
        'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ',
        'パ', 'ピ', 'プ', 'ペ', 'ポ', 'ヴ',
        # Small kana
        'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ', 'ッ', 'ヮ',
        'ヵ', 'ヶ',
        # Special
        'ー',
    ),
    # ref: https://en.wikipedia.org/wiki/Devanagari
    "hi-IN": (
        # Independent Vowels
        'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ए', 'ऐ',
        'ओ', 'औ', 'ऍ', 'ऑ',
        # Consonants
        'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
        'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
        'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
        'ष', 'स', 'ह', 'ळ', 'ऩ', 'ऱ',
        # Dependent Vowels
        'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'े', 'ै', 'ो', 'ौ',
        'ॅ', 'ॉ', 'ँ', 'ं', 'ः', '्', '़', 'ॊ', 'ॢ', 'ॣ', 'ॆ',
        # Danda (period)
        '।',
    ),
}

IPA_CHARACTER_SETS = {
    "en-US": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ',
        'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ',
        'ʒ', 'ʔ', 'ʲ', '̃', '̩', 'θ', 'ᵻ'
    ),
    "es-ES": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x',
        'ð', 'ŋ', 'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɾ', 'ʃ', 'ʊ',
        'ʎ', 'ʒ', 'ʝ', 'β', 'θ'
    ),
    "de-DE": (
        '1', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', 'ç', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ',
        'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ',
        'ʊ', 'ʌ', 'ʒ', '̃', 'θ'
    ),
    "fr-FR": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 
        'y', 'z', 'ð', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 
        'ə', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɲ', 'ɹ', 'ʁ', 'ʃ', 'ʊ', 
        'ʌ', 'ʒ', 'θ', 'ː', '̃'
    ),
    "it-IT": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'z', 'æ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ',
        'ɜ', 'ɬ', 'ɹ', 'ʌ', 'ʔ', 'ʲ', '̃', '̩', 'ᵻ',
        'ð', 'ŋ', 'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɾ', 'ʃ', 
        'ʊ', 'ʎ', 'ʒ', 'ʝ', 'β', 'θ', 'd͡', 't͡', 'ø', 'ɒ',
        'ɕ', 'ɓ', 'ç', 'ɖ', 'ɘ', 'ɝ', 'ɞ', 'ɟ','ʄ','ɡ','ɠ',
        'ɢ','ʛ','ɦ','ɧ','ħ','ɥ','ʜ','ɨ','ɬ','ɫ','ɮ','ʟ',
        'ɱ','ɯ','ɰ','ɳ','ɵ','ɸ','œ','ɶ','ʘ','ɺ','ɻ','ʀ','ʁ',
        'ɽ','ʂ','ʈ','ʧ','ʉ','ʋ','ⱱ','ɤ','ʍ','χ','ʏ','ʑ','ʐ',
        'ʔ','ʡ','ʕ','ʢ','ǀ','ǁ','ǂ','ᵻ', 'ʃ','ː',
    ),
    "vi-VN": (
        'a', 'ə', 'ɛ', 'e', 'i', 'o', 'ɔ', 'u', 'ɨ',
        'b', 'c', 'z', 'j', 'd', 'g', 'h', 'x', 'l',
        'm', 'n', 'ŋ', 'ɲ', 'p', 'f', 'w', 'r', 's',
        'ʃ', 't', 'ʈ', 'ʂ', 'v', 'ʔ', 'ɓ', 'ɗ', 'ɣ',
        'k', 'ʰ', 'ʷ', 'ɕ', 'ʑ', 'ʝ', '̚', '̟', 't͡',
        '˧', 'ː', 'ɯ', '̀', '̄', '̌', '̂', 'ˀ', '͡', '˥',
        '˩', '̤', '˨', 'ɹ', 'ʲ', '̯', 'ă', 'ə̆', 'ǐ',
        '˦', 'æ', 'ɐ',
        'ɜ', 'ɡ', 'ɪ', 'ɬ' 'ɾ', 'ʊ', 'ʌ', 'ʒ', '̃',
        '̩', 'θ', 'ᵻ',
    ),
    "ja-JP": (
        'a', 'i', 'u', 'e', 'o', 'ɯ', 'I', 'ɑ' , 'ɨ ', 'ɒ',
        'ɔ', 'iᵑ', 'eᵑ', 'a', 'ʊ', 'ə', 'eᵝ', 'ɐ', 'ɛ',
        'w', 'k', 'ɾ', 's', 't', 'ʃ', 'r', 'h', 'n', 'nʲ',
        'ɲ', 'ç', 'b', 'm', 'j', 'ɸ', 'z', 'p', 'd', 'N',
        'ʒ', 'ŋ', 'g', 'f', 'ʔ', 'y', 'ɟ', 'v', 'ɥ', 'ɰ',
        'ɰᵝ', 'ɣ', 'ʄ', 'ʑ', 'c', 'ɕ', 'ɠ', 'x', 'l', 'β',
        'ð', 'ø', 'ʁ', 'ts', 'tʃ', 'dʒ', 'y', 'dʑ', 't͡s',
        'ɑ̃', 'ĩ', 'ũ', 'ẽ', 'õ', 'ɑ̃', 'ĩ', 'ũ', 'w̃',
        'ẽ', 'õ', 'hʲ', 'ɪ', 'ː', 'o̞', 'e̞',
    ),
    # Note: '.' is intentionally included for Hindi IPA. It is used in the
    # Hindi pronunciation lexicon/transcriptions (e.g., as a boundary or
    # prosodic marker) and therefore must be part of the allowed phoneme set.
    "hi-IN": (
        '.', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'w', 'x', 'z', 'ŋ', 'ɔ', 'ɖ', 'ə', 'ɛ', 'ɟ',
        'ɡ', 'ɣ', 'ɪ', 'ɭ', 'ɲ', 'ɳ', 'ɾ', 'ʂ', 'ʃ', 'ʈ',
        'ʊ', 'ʋ', 'ʌ', 'ʰ', 'ː', '̃', '̩', 'χ',
    ),
    "hi-IN": (
        '.', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'w', 'x', 'z', 'ŋ', 'ɔ', 'ɖ', 'ə', 'ɛ', 'ɟ',
        'ɡ', 'ɣ', 'ɪ', 'ɭ', 'ɲ', 'ɳ', 'ɾ', 'ʂ', 'ʃ', 'ʈ',
        'ʊ', 'ʋ', 'ʌ', 'ʰ', 'ː', '̃', '̩', 'χ',
    ),
    "hi-IN": (
        '.', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'w', 'x', 'z', 'ŋ', 'ɔ', 'ɖ', 'ə', 'ɛ', 'ɟ',
        'ɡ', 'ɣ', 'ɪ', 'ɭ', 'ɲ', 'ɳ', 'ɾ', 'ʂ', 'ʃ', 'ʈ',
        'ʊ', 'ʋ', 'ʌ', 'ʰ', 'ː', '̃', '̩', 'χ',
    ),
}

GRAPHEME_CHARACTER_CASES = ["upper", "lower", "mixed"]

# fmt: on


def validate_locale(locale):
    """Check if locale is supported"""
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale '{locale}'. " f"Supported locales {SUPPORTED_LOCALES}")


def get_grapheme_character_set(locale: str, case: str = "upper") -> str:
    """Gets set of graphemes for given 'locale' and 'case'"""
    if locale not in GRAPHEME_CHARACTER_SETS:
        raise ValueError(
            f"Grapheme character set not found for locale '{locale}'. "
            f"Supported locales {GRAPHEME_CHARACTER_SETS.keys()}"
        )

    charset_str_origin = ''.join(GRAPHEME_CHARACTER_SETS[locale])
    if case == "upper":
        # Directly call .upper() will convert 'ß' into 'SS' according to https://bugs.python.org/issue30810.
        charset_str = charset_str_origin.replace('ß', 'ẞ').upper()
    elif case == "lower":
        charset_str = charset_str_origin.lower()
    elif case == "mixed":
        charset_str = charset_str_origin.replace('ß', 'ẞ').upper() + charset_str_origin.lower()
    else:
        raise ValueError(
            f"Grapheme character case not found: '{case}'. Supported cases are {GRAPHEME_CHARACTER_CASES}"
        )

    return charset_str


def get_ipa_character_set(locale):
    """Gets set of phones for given 'locale'"""
    if locale not in IPA_CHARACTER_SETS:
        raise ValueError(
            f"IPA character set not found for locale '{locale}'. " f"Supported locales {IPA_CHARACTER_SETS.keys()}"
        )
    char_set = set(IPA_CHARACTER_SETS[locale])
    return char_set


def get_ipa_punctuation_list(locale):
    """Gets set of punctuation for given 'locale'"""
    if locale is None:
        return sorted(list(DEFAULT_PUNCTUATION))

    validate_locale(locale)

    punct_set = set(DEFAULT_PUNCTUATION)
    # TODO @xueyang: verify potential mismatches with locale-specific punctuation sets used
    #  in nemo_text_processing.text_normalization.en.taggers.punctuation.py
    if locale in ["de-DE", "es-ES", "it-IT", "fr-FR", "ja-JP"]:
        # ref: https://en.wikipedia.org/wiki/Guillemet#Uses
        punct_set.update(['«', '»', '‹', '›'])
    if locale == "de-DE":
        # ref: https://en.wikipedia.org/wiki/German_orthography#Punctuation
        punct_set.update(
            [
                '„',  # double low-9 quotation mark, U+201E, decimal 8222
                '“',  # left double quotation mark, U+201C, decimal 8220
                '‚',  # single low-9 quotation mark, U+201A, decimal 8218
                '‘',  # left single quotation mark, U+2018, decimal 8216
                '‒',  # figure dash, U+2012, decimal 8210
                '–',  # en dash, U+2013, decimal 8211
                '—',  # em dash, U+2014, decimal 8212
            ]
        )
    if locale == "it-IT":
        # ref: https://en.wikipedia.org/wiki/German_orthography#Punctuation
        punct_set.update(
            [
                '„',  # double low-9 quotation mark, U+201E, decimal 8222
                '“',  # left double quotation mark, U+201C, decimal 8220
                '‚',  # single low-9 quotation mark, U+201A, decimal 8218
                '‘',  # left single quotation mark, U+2018, decimal 8216
                '‒',  # figure dash, U+2012, decimal 8210
                '–',  # en dash, U+2013, decimal 8211
                '—',  # em dash, U+2014, decimal 8212
                'ʴ',
                'ʰ',
                'ʱ',
                'ʲ',
                'ʷ',
                'ˠ',
                'ˤ',
                '˞↓',
                '↑',
                '→',
                '↗',
                '↘',
                '”',
                '’',
                '-',
            ]
        )
    elif locale == "es-ES":
        # ref: https://en.wikipedia.org/wiki/Spanish_orthography#Punctuation
        punct_set.update(['¿', '¡'])
    elif locale == "fr-FR":
        punct_set.update(
            [
                '–',  # en dash, U+2013, decimal 8211
                '“',  # left double quotation mark, U+201C, decimal 8220
                '”',  # right double quotation mark, U+201D, decimal 8221
                '…',  # horizontal ellipsis, U+2026, decimal 8230
                '̀',  # combining grave accent, U+0300, decimal 768
                '́',  # combining acute accent, U+0301, decimal 769
                '̂',  # combining circumflex accent, U+0302, decimal 770
                '̈',  # combining diaeresis, U+0308, decimal 776
                '̧',  # combining cedilla, U+0327, decimal 807
            ]
        )
    elif locale == "ja-JP":
        # ref: https://en.wikipedia.org/wiki/List_of_Japanese_typographic_symbols
        punct_set.update(
            [
                '【',
                '】',
                '…',
                '‥',
                '「',
                '」',
                '『',
                '』',
                '〜',
                '。',
                '、',
                'ー',
                '・・・',
                '〃',
                '〔',
                '〕',
                '｟',
                '｠',
                '〈',
                '〉',
                '《',
                '》',
                '〖',
                '〗',
                '〘',
                '〙',
                '〚',
                '〛',
                '•',
                '◦',
                '﹅',
                '﹆',
                '※',
                '＊',
                '〽',
                '〓',
                '〒',
                '！',
                '？',
                '・',
            ]
        )
    punct_list = sorted(list(punct_set))
    return punct_list
