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


import argparse
import json
import os

from nemo.collections.asr.metrics.der import evaluate_der
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.parts.utils.manifest_utils import read_file
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
    labels_to_pyannote_object,
    rttm_to_labels,
)


"""
Evaluation script for diarization with ASR.
Calculates Diarization Error Rate (DER) with RTTM files and WER and cpWER with CTM files.
In the output ctm_eval.csv file in the output folder,
session-level DER, WER, cpWER and speaker counting accuracies are evaluated.

- Evaluation mode

diar_eval_mode == "full":
    DIHARD challenge style evaluation, the most strict way of evaluating diarization
    (collar, ignore_overlap) = (0.0, False)
diar_eval_mode == "fair":
    Evaluation setup used in VoxSRC challenge
    (collar, ignore_overlap) = (0.25, False)
diar_eval_mode == "forgiving":
    Traditional evaluation setup
    (collar, ignore_overlap) = (0.25, True)
diar_eval_mode == "all":
    Compute all three modes (default)


Use CTM files to calculate WER and cpWER
```
python eval_diar_with_asr.py \
 --hyp_rttm_list="/path/to/hypothesis_rttm_filepaths.list" \
 --ref_rttm_list="/path/to/reference_rttm_filepaths.list" \
 --hyp_ctm_list="/path/to/hypothesis_ctm_filepaths.list" \
 --ref_ctm_list="/path/to/reference_ctm_filepaths.list" \
 --root_path="/path/to/output/directory"
```

Use .json files to calculate WER and cpWER
```
python eval_diar_with_asr.py \
 --hyp_rttm_list="/path/to/hypothesis_rttm_filepaths.list" \
 --ref_rttm_list="/path/to/reference_rttm_filepaths.list" \
 --hyp_json_list="/path/to/hypothesis_json_filepaths.list" \
 --ref_ctm_list="/path/to/reference_ctm_filepaths.list" \
 --root_path="/path/to/output/directory"
```

Only use RTTMs to calculate DER
```
python eval_diar_with_asr.py \
 --hyp_rttm_list="/path/to/hypothesis_rttm_filepaths.list" \
 --ref_rttm_list="/path/to/reference_rttm_filepaths.list" \
 --root_path="/path/to/output/directory"
```

"""


def get_pyannote_objs_from_rttms(rttm_file_path_list):
    """Generate PyAnnote objects from RTTM file list
    """
    pyannote_obj_list = []
    for rttm_file in rttm_file_path_list:
        rttm_file = rttm_file.strip()
        if rttm_file is not None and os.path.exists(rttm_file):
            uniq_id = get_uniqname_from_filepath(rttm_file)
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            pyannote_obj_list.append([uniq_id, reference])
    return pyannote_obj_list


def make_meta_dict(hyp_rttm_list, ref_rttm_list):
    """Create a temporary `audio_rttm_map_dict` for evaluation
    """
    meta_dict = {}
    for k, rttm_file in enumerate(ref_rttm_list):
        uniq_id = get_uniqname_from_filepath(rttm_file)
        meta_dict[uniq_id] = {"rttm_filepath": rttm_file.strip()}
        if hyp_rttm_list is not None:
            hyp_rttm_file = hyp_rttm_list[k]
            meta_dict[uniq_id].update({"hyp_rttm_filepath": hyp_rttm_file.strip()})
    return meta_dict


def make_trans_info_dict(hyp_json_list_path):
    """Create `trans_info_dict` from the `.json` files
    """
    trans_info_dict = {}
    for json_file in hyp_json_list_path:
        json_file = json_file.strip()
        with open(json_file) as jsf:
            json_data = json.load(jsf)
        uniq_id = get_uniqname_from_filepath(json_file)
        trans_info_dict[uniq_id] = json_data
    return trans_info_dict


def read_file_path(list_path):
    """Read file path and strip to remove line change symbol
    """
    return sorted([x.strip() for x in read_file(list_path)])


def main(
    hyp_rttm_list_path: str,
    ref_rttm_list_path: str,
    hyp_ctm_list_path: str,
    ref_ctm_list_path: str,
    hyp_json_list_path: str,
    diar_eval_mode: str = "all",
    root_path: str = "./",
):

    # Read filepath list files
    hyp_rttm_list = read_file_path(hyp_rttm_list_path) if hyp_rttm_list_path else None
    ref_rttm_list = read_file_path(ref_rttm_list_path) if ref_rttm_list_path else None
    hyp_ctm_list = read_file_path(hyp_ctm_list_path) if hyp_ctm_list_path else None
    ref_ctm_list = read_file_path(ref_ctm_list_path) if ref_ctm_list_path else None
    hyp_json_list = read_file_path(hyp_json_list_path) if hyp_json_list_path else None

    audio_rttm_map_dict = make_meta_dict(hyp_rttm_list, ref_rttm_list)

    trans_info_dict = make_trans_info_dict(hyp_json_list) if hyp_json_list else None

    all_hypothesis = get_pyannote_objs_from_rttms(hyp_rttm_list)
    all_reference = get_pyannote_objs_from_rttms(ref_rttm_list)

    diar_score = evaluate_der(
        audio_rttm_map_dict=audio_rttm_map_dict,
        all_reference=all_reference,
        all_hypothesis=all_hypothesis,
        diar_eval_mode=diar_eval_mode,
    )

    # Get session-level diarization error rate and speaker counting error
    der_results = OfflineDiarWithASR.gather_eval_results(
        diar_score=diar_score,
        audio_rttm_map_dict=audio_rttm_map_dict,
        trans_info_dict=trans_info_dict,
        root_path=root_path,
    )

    if ref_ctm_list is not None:
        # Calculate WER and cpWER if reference CTM files exist
        if hyp_ctm_list is not None:
            wer_results = OfflineDiarWithASR.evaluate(
                audio_file_list=hyp_rttm_list,
                hyp_trans_info_dict=None,
                hyp_ctm_file_list=hyp_ctm_list,
                ref_ctm_file_list=ref_ctm_list,
            )
        elif hyp_json_list is not None:
            wer_results = OfflineDiarWithASR.evaluate(
                audio_file_list=hyp_rttm_list,
                hyp_trans_info_dict=trans_info_dict,
                hyp_ctm_file_list=None,
                ref_ctm_file_list=ref_ctm_list,
            )
        else:
            raise ValueError("Hypothesis information is not provided in the correct format.")
    else:
        wer_results = {}

    # Print average DER, WER and cpWER
    OfflineDiarWithASR.print_errors(der_results=der_results, wer_results=wer_results)

    # Save detailed session-level evaluation results in `root_path`.
    OfflineDiarWithASR.write_session_level_result_in_csv(
        der_results=der_results,
        wer_results=wer_results,
        root_path=root_path,
        csv_columns=OfflineDiarWithASR.get_csv_columns(),
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp_rttm_list", help="path to the filelist of hypothesis RTTM files", type=str, required=True, default=None
    )
    parser.add_argument(
        "--ref_rttm_list", help="path to the filelist of reference RTTM files", type=str, required=True, default=None
    )
    parser.add_argument(
        "--hyp_ctm_list", help="path to the filelist of hypothesis CTM files", type=str, required=False, default=None
    )
    parser.add_argument(
        "--ref_ctm_list", help="path to the filelist of reference CTM files", type=str, required=False, default=None
    )
    parser.add_argument(
        "--hyp_json_list",
        help="(Optional) path to the filelist of hypothesis JSON files",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--diar_eval_mode",
        help='evaluation mode: "all", "full", "fair", "forgiving"',
        type=str,
        required=False,
        default="all",
    )
    parser.add_argument(
        "--root_path", help='directory for saving result files', type=str, required=False, default="./"
    )

    args = parser.parse_args()

    main(
        args.hyp_rttm_list,
        args.ref_rttm_list,
        args.hyp_ctm_list,
        args.ref_ctm_list,
        args.hyp_json_list,
        args.diar_eval_mode,
        args.root_path,
    )
