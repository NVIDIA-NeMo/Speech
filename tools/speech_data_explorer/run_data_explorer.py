import argparse
import os
import subprocess

from sde.run_inside_docker import run_sde_inside_docker


def parse_args():
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest',
        help='path to JSON manifest file',
    )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument(
        '--disable-caching-metrics', action='store_true', help='disable caching metrics for errors analysis'
    )
    parser.add_argument(
        '--estimate-audio-metrics',
        '-a',
        action='store_true',
        help='estimate frequency bandwidth and signal level of audio recordings',
    )
    parser.add_argument(
        '--audio-base-path',
        default=None,
        type=str,
        help='A base path for the relative paths in manifest. It defaults to manifest path.',
    )
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')
    parser.add_argument(
        '--names_compared',
        '-nc',
        nargs=2,
        type=str,
        help='names of the two fields that will be compared, example: pred_text_contextnet pred_text_conformer. "pred_text_" prefix IS IMPORTANT!',
    )
    parser.add_argument(
        '--show_statistics',
        '-shst',
        type=str,
        help='field name for which you want to see statistics (optional). Example: pred_text_contextnet.',
    )
    parser.add_argument(
        '--gpu',
        '-gpu',
        action='store_true',
        help='use GPU-acceleration',
    )
    parser.add_argument(
        '--inside_docker',
        '-dckr',
        action='store_true',
        help='run SDE inside Docker container',
    )

    args = parser.parse_args()

    if args.inside_docker:
        run_sde_inside_docker(args)
    else:
        data_explorer_script_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_explorer.py")
        cmd = ["python", f"{data_explorer_script_filepath}"]
        sde_args = [args.manifest]

        for arg in ["vocab", "port", "audio-base-path", "names_compared", "show_statistics"]:
            attr_value = getattr(args, arg.replace("-", "_"))
            if attr_value:
                sde_args.append(f"--{arg}={attr_value}")

        for arg in ["disable-caching-metrics", "estimate-audio-metrics", "debug", "gpu"]:
            attr_value = getattr(args, arg.replace("-", "_"))
            if attr_value:
                sde_args.append(f"--{arg}")

        cmd.extend(sde_args)

        subprocess.run(cmd)


if __name__ == "__main__":
    parse_args()
