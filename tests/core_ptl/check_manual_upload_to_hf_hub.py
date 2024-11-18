# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import shutil

from huggingface_hub import HfApi
from lightning.pytorch.utilities import rank_zero_only

from nemo.core import ModelPT
from nemo.utils import AppState, logging


@rank_zero_only
def load_model():
    # Load an ASR model for convenience
    model = ModelPT.from_pretrained('nvidia/stt_en_conformer_ctc_small', map_location='cpu')
    model.eval()
    return model


@rank_zero_only
def load_model_from_unpacked_hf_dir(repo_id):
    # Load an ASR model for convenience
    model = ModelPT.from_pretrained(repo_id, map_location='cpu')
    model.eval()
    return model


@rank_zero_only
def upload_model_as_single_nemo_file(model: ModelPT, repo_id, token):
    # Upload the model to HF Hub
    model.push_to_hf_hub(
        repo_id=repo_id,
        pack_nemo_file=True,
        token=token,
    )


@rank_zero_only
def upload_model_as_single_nemo_file(model: ModelPT, repo_id, token):
    # Upload the model to HF Hub
    model.push_to_hf_hub(
        repo_id=repo_id,
        pack_nemo_file=True,
        token=token,
    )


@rank_zero_only
def upload_model_as_unpacked_files(model: ModelPT, repo_id, token):
    # Upload the model to HF Hub
    model.push_to_hf_hub(
        repo_id=repo_id,
        pack_nemo_file=False,
        token=token,
    )


@rank_zero_only
def check_repo_exists(repo_id, token):
    api = HfApi(token=token)
    try:
        model_info = api.repo_info(repo_id, repo_type='model')
        assert model_info.id == repo_id, f"Repo {repo_id} does not exist or you don't have access to it"

    except Exception as e:
        # logging.error(f"Error checking repo {repo_id}: {e}")
        return False
    return True


@rank_zero_only
def cleanup(repo_id, token):
    api = HfApi(token=token)

    single_repo_id = repo_id + '/temp_single'
    if check_repo_exists(single_repo_id, token):
        api.delete_repo(single_repo_id, repo_type='model')
        logging.info(f"Deleted {single_repo_id}")

    unpacked_repo_id = repo_id + '/temp_unpacked'
    if check_repo_exists(unpacked_repo_id, token):
        api.delete_repo(unpacked_repo_id, repo_type='model')
        logging.info(f"Deleted {unpacked_repo_id}")


@rank_zero_only
def run_checks(repo_id, token, run_cleanup=True):
    if run_cleanup:
        cleanup(repo_id, token)

    model = load_model()

    # single_repo_id = repo_id + '/temp_single'
    # upload_model_as_single_nemo_file(model, single_repo_id, token)
    # assert check_repo_exists(single_repo_id, token)

    unpacked_repo_id = repo_id + '/temp_unpacked'
    upload_model_as_unpacked_files(model, unpacked_repo_id, token)
    assert check_repo_exists(unpacked_repo_id, token)

    # Try to restore model from the unpacked repo
    model2 = load_model_from_unpacked_hf_dir(unpacked_repo_id)
    assert (
        model.num_weights == model2.num_weights
    ), "Number of parameters in the restored model is different from the original model"

    # Cleanup nemo file that was cached
    appstate = AppState()
    model2_metadata = appstate.get_model_metadata_from_guid(model2.model_guid)

    restore_path = model2_metadata.restoration_path
    print(restore_path)
    if unpacked_repo_id in restore_path:
        print("Cleaning up the cached nemo file")
        shutil.rmtree(restore_path, ignore_errors=True)


if __name__ == '__main__':
    HF_TOKEN = ''  # Set it explicitly or set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env variable
    repo_id = (
        ''  # Provide some repo id here (you need to be able to have sufficient permissions to write to this repo)
    )

    assert repo_id is not None, "Please set `repo_id` to the name of the repo you want to upload to"
    run_checks(repo_id, HF_TOKEN, run_cleanup=True)

    # Cleanup at the end
    # COMMENT THIS OUT TO MANUALLY INSPECT THE REPOS CREATED BY THIS SCRIPT
    # cleanup(repo_id, HF_TOKEN)
