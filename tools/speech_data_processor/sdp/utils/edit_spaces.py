# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


def remove_extra_spaces(input_string):
    """
	Removes extra spaces in between words and at the start and end
	of the string.
	e.g. "abc  xyz   abc xyz" --> "abc xyz abc xyz"
	e.g. " abc xyz " --> "abc xyz"
	"""
    output_string = " ".join(input_string.split())
    return output_string


def add_start_end_spaces(input_string):
    """
	Adds spaces at the start and end of the input string.
	This is useful for when we specify we are looking for a particular
	word " <word> ". This will ensure we will find the word even
	if it is at the beginning or end of the utterances (ie. there will
	definitely be two spaces around the word).

	e.g. "abc xyz" --> " abc xyz "
	"""
    # ensure no extra spaces
    no_extra_spaces_string = remove_extra_spaces(input_string)
    output_string = f" {no_extra_spaces_string} "

    return output_string
