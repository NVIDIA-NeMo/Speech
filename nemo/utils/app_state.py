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

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

from nemo.utils.metaclasses import Singleton


@dataclass()
class ModelMetadataRegistry:
    guid: str
    gidx: int
    restoration_path: Optional[str] = None


class AppState(metaclass=Singleton):
    def __init__(self):
        # method call lock
        self.__lock = Lock()

        # TODO: should we store global config in hydra_runner?
        self._app_cfg = None

        # World info
        self._device_id = None
        self._local_rank = None
        self._global_rank = None
        self._tensor_model_parallel_rank = None
        self._expert_model_parallel_rank = None
        self._pipeline_model_parallel_rank = None
        self._data_parallel_rank = None

        self._world_size = None
        self._model_parallel_size = None
        self._tensor_model_parallel_size = None
        self._tensor_model_parallel_group = None
        self._expert_model_parallel_size = None
        self._pipeline_model_parallel_size = None
        self._virtual_pipeline_model_parallel_size = None
        self._pipeline_model_parallel_group = None
        self._pipeline_model_parallel_split_rank = None
        self._is_megatron_initialized = False
        self._data_parallel_size = None
        self._data_parallel_group = None
        self._use_tp_pp_dp_mapping = False
        self._megatron_checkpoint_version = None
        self._use_fp8 = False
        self._context_parallel_size = None
        self._init_mpi_proc_gruop = False

        self._random_seed = None

        # Logging info
        self._log_dir = None
        self._exp_dir = None
        self._name = None
        self._checkpoint_name = None
        self._version = None
        self._create_checkpoint_callback = None
        self._checkpoint_callback_params = None

        # Save and Restore (.nemo)
        self._tmpdir_name = None
        self._is_model_being_restored = False
        self._nemo_file_folder = None
        self._model_restore_path = None
        self._all_model_restore_paths = []
        self._model_guid_map = {}  # type: Dict[str, ModelMetadataRegistry]
        self._restore = False  # TODO: are this and _is_model_being_restored both needed?

        # files from a previous run to move into a new directory
        self.files_to_move = []
        # files to copy into log dir
        self._files_to_copy = []
        # command-ling arguments for run
        self._cmd_args = None

    @property
    def device_id(self):
        """Property returns the device_id
        Returns:
            device_id
        """
        return self._device_id

    @device_id.setter
    def device_id(self, id):
        """Property sets the device_id.
        Args:
            size (int): The device id.
        """
        self._device_id = id

    @property
    def world_size(self):
        """Property returns the total number of GPUs.
        Returns:
            Total number of GPUs.
        """
        return self._world_size

    @world_size.setter
    def world_size(self, size):
        """Property sets the total number of GPUs.
        Args:
            size (int):  Total number of GPUs.
        """
        self._world_size = size

    @property
    def model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group.
        Returns:
            Number of GPUs in each model parallel group.
        """
        return self._model_parallel_size

    @model_parallel_size.setter
    def model_parallel_size(self, size):
        """Property sets the number of GPUs in each model parallel group.
        Args:
            size (int):  Number of GPUs in each model parallel group.
        """
        self._model_parallel_size = size

    @property
    def tensor_model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group.
        Returns:
            Number of GPUs in each model parallel group.
        """
        return self._tensor_model_parallel_size

    @tensor_model_parallel_size.setter
    def tensor_model_parallel_size(self, size):
        """Property sets the number of GPUs in each model parallel group.
        Args:
            size (int):  Number of GPUs in each model parallel group.
        """
        self._tensor_model_parallel_size = size

    @property
    def expert_model_parallel_rank(self):
        """Property returns the expert model parallel rank.
        Returns:
            Tensor model parallel rank.
        """
        return self._expert_model_parallel_rank

    @expert_model_parallel_rank.setter
    def expert_model_parallel_rank(self, rank):
        """Property sets the expert model parallel rank.
        Args:
            rank (int):  Tensor model parallel rank.
        """
        self._expert_model_parallel_rank = rank

    @property
    def expert_model_parallel_size(self):
        """Property returns the number of GPUs in each expert parallel group.
        Returns:
            Number of GPUs in each expert parallel group.
        """
        return self._expert_model_parallel_size

    @expert_model_parallel_size.setter
    def expert_model_parallel_size(self, size):
        """Property sets the number of GPUs in each expert parallel group.
        Args:
            size (int):  Number of GPUs in each expert parallel group.
        """
        self._expert_model_parallel_size = size

    @property
    def pipeline_model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group.
        Returns:
            Number of GPUs in each model parallel group.
        """
        return self._pipeline_model_parallel_size

    @pipeline_model_parallel_size.setter
    def pipeline_model_parallel_size(self, size):
        """Property sets the number of GPUs in each model parallel group.
        Args:
            size (int):  Number of GPUs in each model parallel group.
        """
        self._pipeline_model_parallel_size = size

    @property
    def use_tp_pp_dp_mapping(self):
        return self._use_tp_pp_dp_mapping

    @use_tp_pp_dp_mapping.setter
    def use_tp_pp_dp_mapping(self, use_new_mapping):
        self._use_tp_pp_dp_mapping = use_new_mapping

    @property
    def virtual_pipeline_model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group.
        Returns:
            Number of GPUs in each model parallel group.
        """
        return self._virtual_pipeline_model_parallel_size

    @virtual_pipeline_model_parallel_size.setter
    def virtual_pipeline_model_parallel_size(self, size):
        """Property sets the size of the virtual pipeline parallel model.
        Args:
            size (int):  Number of modules in each pipeline parallel model.
        """
        self._virtual_pipeline_model_parallel_size = size

    @property
    def data_parallel_size(self):
        """Property returns the number of GPUs in each data parallel group.
        Returns:
            Number of GPUs in each data parallel group.
        """
        return self._data_parallel_size

    @data_parallel_size.setter
    def data_parallel_size(self, size):
        """Property sets the number of GPUs in each data parallel group.
        Args:
            size (int):  Number of GPUs in each data parallel group.
        """
        self._data_parallel_size = size

    @property
    def local_rank(self):
        """Property returns the local rank.
        Returns:
            Local rank.
        """
        return self._local_rank

    @local_rank.setter
    def local_rank(self, rank):
        """Property sets the local rank.
        Args:
            rank (int):  Local rank.
        """
        self._local_rank = rank

    @property
    def global_rank(self):
        """Property returns the global rank.
        Returns:
            Global rank.
        """
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank):
        """Property sets the global rank.
        Args:
            rank (int):  Global rank.
        """
        self._global_rank = rank

    @property
    def tensor_model_parallel_rank(self):
        """Property returns the tensor model parallel rank.
        Returns:
            Tensor model parallel rank.
        """
        return self._tensor_model_parallel_rank

    @tensor_model_parallel_rank.setter
    def tensor_model_parallel_rank(self, rank):
        """Property sets the tensor model parallel rank.
        Args:
            rank (int):  Tensor model parallel rank.
        """
        self._tensor_model_parallel_rank = rank

    @property
    def tensor_model_parallel_group(self):
        """Property returns the tensor model parallel group.
        Returns:
            Tensor model parallel group.
        """
        return self._tensor_model_parallel_group

    @tensor_model_parallel_group.setter
    def tensor_model_parallel_group(self, group):
        """Property sets the tensor model parallel group.
        Args:
            group:  Tensor model parallel group.
        """
        self._tensor_model_parallel_group = group

    @property
    def pipeline_model_parallel_rank(self):
        """Property returns the pipeline model parallel rank.
        Returns:
            Pipeline model parallel rank.
        """
        return self._pipeline_model_parallel_rank

    @pipeline_model_parallel_rank.setter
    def pipeline_model_parallel_rank(self, rank):
        """Property sets the pipeline model parallel rank.
        Args:
            rank (int):  Pipeline model parallel rank.
        """
        self._pipeline_model_parallel_rank = rank

    @property
    def virtual_pipeline_model_parallel_rank(self):
        """Property returns the virtual pipeline parallel rank.
        Returns:
            Model parallel rank.
        """
        return self._virtual_pipeline_model_parallel_rank

    @virtual_pipeline_model_parallel_rank.setter
    def virtual_pipeline_model_parallel_rank(self, rank):
        """Property sets the virtual pipeline parallel rank.
        Args:
            rank (int):  Virtual pipeline parallel rank.
        """
        self._virtual_pipeline_model_parallel_rank = rank

    @property
    def pipeline_model_parallel_split_rank(self):
        """Property returns the rank at which Encoder and Decoder are split into different pipelines for Megatrron Encoder-Decoder models.
        Returns:
            Pipeline model parallel split rank.
        """
        return self._pipeline_model_parallel_split_rank

    @pipeline_model_parallel_split_rank.setter
    def pipeline_model_parallel_split_rank(self, rank):
        """Property sets the rank at which Encoder and Decoder are split into different pipelines for Megatrron Encoder-Decoder models.
        Args:
            rank (int): Model parallel split rank.
        """
        self._pipeline_model_parallel_split_rank = rank

    @property
    def pipeline_model_parallel_group(self):
        """Property returns the pipeline model parallel group.
        Returns:
            Pipeline model parallel group.
        """
        return self._pipeline_model_parallel_group

    @pipeline_model_parallel_group.setter
    def pipeline_model_parallel_group(self, group):
        """Property sets the pipeline model parallel group.
        Args:
            group:  Pipeline model parallel group.
        """
        self._pipeline_model_parallel_group = group

    @property
    def data_parallel_rank(self):
        """Property returns the data parallel rank.
        Returns:
            Data parallel rank.
        """
        return self._data_parallel_rank

    @data_parallel_rank.setter
    def data_parallel_rank(self, rank):
        """Property sets the data parallel rank.
        Args:
            rank (int):  Data parallel rank.
        """
        self._data_parallel_rank = rank

    @property
    def data_parallel_group(self):
        """Property returns the data parallel group.
        Returns:
            Data parallel group.
        """
        return self._data_parallel_group

    @data_parallel_group.setter
    def data_parallel_group(self, group):
        """Property sets the data parallel group.
        Args:
            group:  Data parallel group.
        """
        self._data_parallel_group = group

    @property
    def use_fp8(self):
        """Property returns the use of fp8 precision.
        Returns:
            Use of FP8.
        """
        return self._use_fp8

    @use_fp8.setter
    def use_fp8(self, use_fp8):
        """Property sets the use of fp8 precision.
        Args:
            use_fp8:  Use of FP8.
        """
        self._use_fp8 = use_fp8

    @property
    def context_parallel_size(self):
        """Property returns the number of GPUs in each context parallel group.
        Returns:
            Number of GPUs in each context parallel group.
        """
        return self._context_parallel_size

    @context_parallel_size.setter
    def context_parallel_size(self, size):
        """Property sets the number of GPUs in each context parallel group.
        Args:
            size (int):  Number of GPUs in each context parallel group.
        """
        self._context_parallel_size = size

    @property
    def init_mpi_proc_group(self):
        """Property sets the initialization of mpi process group.
        Returns:
            Initialize mpi process group.
        """
        return self._init_mpi_proc_group

    @init_mpi_proc_group.setter
    def init_mpi_proc_group(self, init_mpi_proc_group):
        """Property sets the initialization of mpi process group.
        Args:
            init_mpi_proc_group:  Initialize mpi process group.
        """
        self._init_mpi_proc_group = init_mpi_proc_group

    @property
    def random_seed(self):
        """Property returns the random seed.
        Returns:
            Random seed.
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Property sets the random seed.
        Args:
            seed (int):  Random seed.
        """
        self._random_seed = seed

    @property
    def log_dir(self):
        """Returns the log_dir set by exp_manager."""
        return self._log_dir

    @log_dir.setter
    def log_dir(self, dir):
        """Sets the log_dir property.

        Args:
            dir (str): Log_dir set by exp_manager.
        """
        self._log_dir = dir

    @property
    def exp_dir(self):
        """Returns the exp_dir set by exp_manager."""
        return self._exp_dir

    @exp_dir.setter
    def exp_dir(self, dir):
        """Sets the log_dir property.

        Args:
            dir (str): Log_dir set by exp_manager.
        """
        self._exp_dir = dir

    @property
    def name(self):
        """Returns the name set by exp_manager."""
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name property.

        Args:
            dir (str): name set by exp_manager.
        """
        self._name = name

    @property
    def checkpoint_name(self):
        """Returns the name set by exp_manager."""
        return self._checkpoint_name

    @checkpoint_name.setter
    def checkpoint_name(self, name):
        """Sets the name property.

        Args:
            dir (str): name set by exp_manager.
        """
        self._checkpoint_name = name

    @property
    def version(self):
        """Returns the version set by exp_manager."""
        return self._version

    @version.setter
    def version(self, version):
        """Sets the version property.

        Args:
            dir (str): version set by exp_manager.
        """
        self._version = version

    @property
    def create_checkpoint_callback(self):
        """Returns the create_checkpoint_callback set by exp_manager."""
        return self._create_checkpoint_callback

    @create_checkpoint_callback.setter
    def create_checkpoint_callback(self, create_checkpoint_callback):
        """Sets the create_checkpoint_callback property.

        Args:
            dir (bool): create_checkpoint_callback set by exp_manager.
        """
        self._create_checkpoint_callback = create_checkpoint_callback

    @property
    def checkpoint_callback_params(self):
        """Returns the version set by exp_manager."""
        return self._checkpoint_callback_params

    @checkpoint_callback_params.setter
    def checkpoint_callback_params(self, params):
        """Sets the name property.

        Args:
            params (dict): checkpoint_callback_params set by exp_manager.
        """
        self._checkpoint_callback_params = params

    @property
    def files_to_move(self):
        """Returns the list of files to move into a separate directory."""
        return self._files_to_move

    @files_to_move.setter
    def files_to_move(self, files):
        """Sets the files_to_move property.

        Args:
            files (list[str]): list of filenames to move.
        """
        self._files_to_move = files

    @property
    def files_to_copy(self):
        """Returns the list of files to copy into the log dir."""
        return self._files_to_copy

    @files_to_copy.setter
    def files_to_copy(self, files):
        """Sets the files_to_copy property.

        Args:
            files (list[str]): list of filenames to copy.
        """
        self._files_to_copy = files

    @property
    def cmd_args(self):
        """Returns the command line arguments for the current run."""
        return self._cmd_args

    @cmd_args.setter
    def cmd_args(self, args):
        """Sets the cmd_args property.

        Args:
            args (list[str]): list of the command line arguments
                used to run the experiment.
        """
        self._cmd_args = args

    @property
    def model_restore_path(self):
        restore_path = self._all_model_restore_paths[-1] if len(self._all_model_restore_paths) > 0 else None
        return restore_path

    @model_restore_path.setter
    def model_restore_path(self, path):
        with self.__lock:
            self._model_restore_path = path
            self._all_model_restore_paths.append(path)

    def register_model_guid(self, guid: str, restoration_path: Optional[str] = None):
        # Maps a guid to its restore path (None or last absolute path)
        with self.__lock:
            if guid in self._model_guid_map:
                idx = self._model_guid_map[guid].gidx
            else:
                idx = len(self._model_guid_map)
            self._model_guid_map[guid] = ModelMetadataRegistry(guid, idx, restoration_path=restoration_path)

    def reset_model_guid_registry(self):
        # Reset the guid mapping
        with self.__lock:
            self._model_guid_map.clear()

    def get_model_metadata_from_guid(self, guid) -> ModelMetadataRegistry:
        # Returns the global model idx and restoration path
        metadata = self._model_guid_map[guid]
        return metadata

    @property
    def is_model_being_restored(self) -> bool:
        return self._is_model_being_restored

    @is_model_being_restored.setter
    def is_model_being_restored(self, is_restored: bool):
        self._is_model_being_restored = is_restored

    @property
    def nemo_file_folder(self) -> str:
        return self._nemo_file_folder

    @nemo_file_folder.setter
    def nemo_file_folder(self, path: str):
        self._nemo_file_folder = path

    @property
    def restore(self) -> bool:
        return self._restore

    @restore.setter
    def restore(self, restore: bool):
        self._restore = restore
