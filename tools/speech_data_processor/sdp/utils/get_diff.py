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

from typing import List

import diff_match_patch
from sdp.utils.edit_spaces import remove_extra_spaces

diff = diff_match_patch.diff_match_patch()
diff.Diff_Timeout = 0


def get_diff_with_subs_grouped(orig_words: str, pred_words: str) -> List[tuple]:
    """
    Function to produce a list of word-level diffs, but with the substitutions 
    grouped together.
        e.g. 
        orig_words = "hello there nemo"
        pred_words = "hello my name is nemo"
        will give an output of:
        [(0, 'hello '), ((-1, 'there '), (1, 'my name is ')), (0, 'nemo ')]
        (note how the 'there' nad 'my name is' entry are grouped together in a tuple)

        This is to make it easier to find substitutions in the diffs, as 
        dif_match_patch does not show substitutions clearly, only as a deletion followed by
        an insertion.

    Args:
        orig_words: a string containing the groud truth.
        pred_words: a string containing the text predicted by ASR.

    Returns:
        A list of tuples containing the word-level diffs between the ground truth
        and ASR. 
    """

    orig_words = remove_extra_spaces(orig_words)
    orig_words = orig_words.replace(" ", "\n") + "\n"

    pred_words = remove_extra_spaces(pred_words)
    pred_words = pred_words.replace(" ", "\n") + "\n"

    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []

    for d in diffs:
        diffs_post.append((d[0], d[1].replace("\n", " ")))
    diffs = diffs_post

    diffs_group_subs = []
    i = 0
    while i < len(diffs):
        if i < len(diffs) - 1:  # if i == len(diffs), line accessing diffs[i+1] will raise error
            if diffs[i][0] == -1 and diffs[i + 1][0] == 1:
                diffs_group_subs.append((diffs[i], diffs[i + 1]))
                i += 1  # skip extra diff entry so we don't append diffs[i+1] again
            else:
                diffs_group_subs.append(diffs[i])
        else:
            diffs_group_subs.append(diffs[i])

        i += 1

    return diffs_group_subs
