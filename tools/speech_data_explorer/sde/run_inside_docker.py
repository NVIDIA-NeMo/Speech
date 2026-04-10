import json
import logging
import os
from pathlib import Path

import docker


class DockerSDE:
    def __init__(self, sde_args, image_tag: str = "rapidsai-nemo-sde", data_dir_path: str = None):
        self.client = docker.from_env()
        self.sde_args = sde_args
        self.dockerfile_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.image_tag = image_tag
        self.manifest_filename = os.path.basename(self.sde_args.manifest)
        self.manifest_volume_name = f"{Path(self.manifest_filename).stem}_volume"
        self.container_name = Path(self.manifest_filename).stem
        self.volume = None
        self.container = None

        self.gpus = self.sde_args.gpu
        if self.gpus:
            self._set_gpus_param()
        else:

            self.gpus = None

        self.data_dir_path = data_dir_path
        if self.data_dir_path is None:
            self._set_data_dir_path()

        self.sde_args_line = f"/manifest/manifest.json --port={self.sde_args.port} "
        self.set_sde_args_line()

    def _set_gpus_param(self):
        if self.gpus:
            self.gpus == "all"
        elif type(self.gpus) is int:
            self.gpus = f"device={self.gpus}"
        else:
            raise f'Invalid value of "gpus" = {self.gpus}'

    def _set_data_dir_path(self):
        if self.sde_args.audio_base_path is not None:
            self.data_dir_path = self.sde_args.audio_base_path

        else:
            with open(self.sde_args.manifest, 'r') as manifest:
                line = manifest.readline()
                sample = json.loads(line)
                common_path = sample['audio_filepath']
                line = manifest.readline()

                while line:
                    sample = json.loads(line)
                    audio_filepath = sample['audio_filepath']
                    common_path = os.path.commonpath([common_path, audio_filepath])
                    line = manifest.readline()

                self.data_dir_path = common_path

        if not os.path.exists(self.data_dir_path):
            raise FileNotFoundError(f"The data dir does not exist: {self.data_dir_path}")

        logging.info(f"Data dir {self.data_dir_path} will be mounted to \"/data\" dir in container (mode: read-only).")

    def build_docker_image(self):
        image, logs = self.client.images.build(path=self.dockerfile_dir, dockerfile='Dockerfile', tag=self.image_tag)
        logging.info(f"Image {self.image_tag} successfully built.")

    def create_docker_volume(self):
        self.volume = self.client.volumes.create(self.manifest_volume_name)
        logging.info(f"Volume {self.manifest_volume_name} successfully built.")

    def copy_manifest_to_volume(self):
        manifest_dirpath = os.path.dirname(self.sde_args.manifest)
        manifest_filename = os.path.basename(self.sde_args.manifest)

        self.client.containers.run(
            image="busybox",
            remove=True,
            volumes={
                manifest_dirpath: {"bind": "/host", "mode": "ro"},
                self.manifest_volume_name: {"bind": "/manifest", "mode": "rw"},
            },
            command=["cp", f"/host/{manifest_filename}", "/manifest/manifest.json"],
        )

        logging.info(f"Manifest {self.sde_args.manifest} successfully copied to docker volume.")

    def set_sde_args_line(self):
        for sde_arg in ["vocab", "names_compared", "show_statistics"]:
            attr_value = getattr(self.sde_args, sde_arg)
            if attr_value is not None:
                self.sde_args_line += f"--{sde_arg}={attr_value} "

        for sde_arg in ["disable_caching_metrics", "estimate_audio_metrics", "debug", "gpu"]:
            attr_value = getattr(self.sde_args, sde_arg)
            if attr_value:
                sde_arg = sde_arg.replace("_", "-")
                self.sde_args_line += f"--{sde_arg} "

    def run_docker_container(self):
        environment_vars = {}
        environment_vars['SDE_ARGS'] = self.sde_args_line
        environment_vars['INIT_DATA_DIR'] = self.data_dir_path

        self.container = self.client.containers.run(
            image=self.image_tag,
            name=self.container_name,
            remove=True,
            volumes={
                self.data_dir_path: {"bind": "/data", "mode": "ro"},
                self.manifest_volume_name: {"bind": "/manifest", "mode": "rw"},
            },
            environment=environment_vars,
            tty=True,
            shm_size="8g",
            ports={self.sde_args.port: self.sde_args.port},
            ulimits=[
                docker.types.Ulimit(name='memlock', soft=-1, hard=-1),
                docker.types.Ulimit(name='stack', soft=67108864, hard=67108864),
            ],
            device_requests=[docker.types.DeviceRequest(device_ids=["0,1"], capabilities=[['gpu']])],
            detach=True,
        )

        self.container.exec_run("python /workspace/speech_data_explorer/sde/paths.py")
        _, d = self.container.exec_run(
            f"python /workspace/speech_data_explorer/data_explorer.py {self.sde_args_line}", stream=True, detach=True
        )

        logging.info(f"Docker container {self.container_name} successfully started.")

    def run_docker_sde(self):
        self.build_docker_image()
        self.create_docker_volume()
        self.copy_manifest_to_volume()
        self.run_docker_container()

        print(f"http://0.0.0.0:{self.sde_args.port}")

        responce = ""
        while responce != "no":
            responce = input("To stop it enter 'no': ")
        else:
            self.container.stop()


def run_sde_inside_docker(sde_args):
    docker_sde_obj = DockerSDE(sde_args=sde_args)
    docker_sde_obj.run_docker_sde()
