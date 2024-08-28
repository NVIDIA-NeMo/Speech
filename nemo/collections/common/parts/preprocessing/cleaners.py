# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import re

from text_unidecode import unidecode

from nemo.utils import logging

NUM_CHECK = re.compile(r'([$]?)(^|\s)(\S*[0-9]\S*)(?=(\s|$)((\S*)(\s|$))?)')

TIME_CHECK = re.compile(r'([0-9]{1,2}):([0-9]{2})(am|pm)?')
CURRENCY_CHECK = re.compile(r'\$')
ORD_CHECK = re.compile(r'([0-9]+)(st|nd|rd|th)')
THREE_CHECK = re.compile(r'([0-9]{3})([.,][0-9]{1,2})?([!.?])?$')
DECIMAL_CHECK = re.compile(r'([.,][0-9]{1,2})$')

ABBREVIATIONS_COMMON = [
    (re.compile('\\b%s\\.' % x[0]), x[1])
    for x in [
        ("ms", "miss"),
        ("mrs", "misess"),
        ("mr", "mister"),
        ("messrs", "messeurs"),
        ("dr", "doctor"),
        ("drs", "doctors"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("sr", "senior"),
        ("rev", "reverend"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("maj", "major"),
        ("col", "colonel"),
        ("lt", "lieutenant"),
        ("gen", "general"),
        ("prof", "professor"),
        ("lb", "pounds"),
        ("rep", "representative"),
        ("st", "street"),
        ("ave", "avenue"),
        ("etc", "et cetera"),
        ("jan", "january"),
        ("feb", "february"),
        ("mar", "march"),
        ("apr", "april"),
        ("jun", "june"),
        ("jul", "july"),
        ("aug", "august"),
        ("sep", "september"),
        ("oct", "october"),
        ("nov", "november"),
        ("dec", "december"),
    ]
]

ABBREVIATIONS_EXPANDED = [
    (re.compile('\\b%s\\.' % x[0]), x[1])
    for x in [
        ("ltd", "limited"),
        ("fig", "figure"),
        ("figs", "figures"),
        ("gent", "gentlemen"),
        ("ft", "fort"),
        ("esq", "esquire"),
        ("prep", "preperation"),
        ("bros", "brothers"),
        ("ind", "independent"),
        ("mme", "madame"),
        ("pro", "professional"),
        ("vs", "versus"),
        ("inc", "include"),
    ]
]

ABBREVIATIONS_TTS_FASTPITCH = [
    (re.compile('\\b%s\\.' % x[0]), x[1])
    for x in [
        ("ms", "miss"),
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("drs", "doctors"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("sr", "senior"),
        ("rev", "reverend"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("maj", "major"),
        ("col", "colonel"),
        ("lt", "lieutenant"),
        ("gen", "general"),
        ("prof", "professor"),
        ("lb", "pounds"),
        ("rep", "representative"),
        ("st", "street"),
        ("ave", "avenue"),
        ("jan", "january"),
        ("feb", "february"),
        ("mar", "march"),
        ("apr", "april"),
        ("jun", "june"),
        ("jul", "july"),
        ("aug", "august"),
        ("sep", "september"),
        ("oct", "october"),
        ("nov", "november"),
        ("dec", "december"),
        ("ltd", "limited"),
        ("fig", "figure"),
        ("figs", "figures"),
        ("gent", "gentlemen"),
        ("ft", "fort"),
        ("esq", "esquire"),
        ("prep", "preperation"),
        ("bros", "brothers"),
        ("ind", "independent"),
        ("mme", "madame"),
        ("pro", "professional"),
        ("vs", "versus"),
    ]
]


from functools import cache


@cache
def inflect_engine():
    import inflect

    return inflect.engine()


def clean_text(string, table, punctuation_to_replace, abbreviation_version=None):
    warn_common_chars(string)
    string = unidecode(string)
    string = string.lower()
    string = re.sub(r'\s+', " ", string)
    string = clean_numbers(string)
    string = clean_abbreviations(string, version=abbreviation_version)
    string = clean_punctuations(string, table, punctuation_to_replace)
    string = re.sub(r'\s+', " ", string).strip()
    return string


def warn_common_chars(string):
    if re.search(r'[£€]', string):
        logging.warning("Your transcript contains one of '£' or '€' which we do not currently handle")


def clean_numbers(string):
    cleaner = NumberCleaner()
    string = NUM_CHECK.sub(cleaner.clean, string)
    return string


def clean_abbreviations(string, version=None):
    abbbreviations = ABBREVIATIONS_COMMON
    if version == "fastpitch":
        abbbreviations = ABBREVIATIONS_TTS_FASTPITCH
    elif version == "expanded":
        abbbreviations.extend = ABBREVIATIONS_EXPANDED
    for regex, replacement in abbbreviations:
        string = re.sub(regex, replacement, string)
    return string


def clean_punctuations(string, table, punctuation_to_replace):
    for punc, replacement in punctuation_to_replace.items():
        string = re.sub('\\{}'.format(punc), " {} ".format(replacement), string)
    if table:
        string = string.translate(table)
    return string


class NumberCleaner:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.curr_num = []
        self.currency = None

    def format_final_number(self, whole_num, decimal):
        inflect = inflect_engine()
        if self.currency:
            return_string = inflect.number_to_words(whole_num)
            return_string += " dollar" if whole_num == 1 else " dollars"
            if decimal:
                return_string += " and " + inflect_engine().number_to_words(decimal)
                return_string += " cent" if whole_num == decimal else " cents"
            self.reset()
            return return_string

        self.reset()
        if decimal:
            whole_num += "." + decimal
            return inflect.number_to_words(whole_num)
        else:
            # Check if there are non-numbers
            def convert_to_word(match):
                return " " + inflect_engine().number_to_words(match.group(0)) + " "

            return re.sub(r'[0-9,]+', convert_to_word, whole_num)

    def clean(self, match):
        inflect = inflect_engine()
        ws = match.group(2)
        number = match.group(3)
        _proceeding_symbol = match.group(7)

        time_match = TIME_CHECK.match(number)
        if time_match:
            string = ws + inflect.number_to_words(time_match.group(1)) + "{}{}"
            mins = int(time_match.group(2))
            min_string = ""
            if mins != 0:
                min_string = " " + inflect.number_to_words(time_match.group(2))
            ampm_string = ""
            if time_match.group(3):
                ampm_string = " " + time_match.group(3)
            return string.format(min_string, ampm_string)

        ord_match = ORD_CHECK.match(number)
        if ORD_CHECK.match(number):
            return ws + inflect.number_to_words(ord_match.group(0))

        if self.currency is None:
            # Check if it is a currency
            self.currency = match.group(1) or CURRENCY_CHECK.match(number)

        # Check to see if next symbol is a number
        # If it is a number and it has 3 digits, then it is probably a
        # continuation
        three_match = THREE_CHECK.match(match.group(6))
        if three_match:
            self.curr_num.append(number)
            return " "
        # Else we can output
        else:
            # Check for decimals
            whole_num = "".join(self.curr_num) + number
            decimal = None
            decimal_match = DECIMAL_CHECK.search(whole_num)
            if decimal_match:
                decimal = decimal_match.group(1)[1:]
                whole_num = whole_num[: -len(decimal) - 1]
            whole_num = re.sub(r'\.', '', whole_num)
            return ws + self.format_final_number(whole_num, decimal)
