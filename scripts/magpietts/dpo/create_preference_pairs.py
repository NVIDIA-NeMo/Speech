# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import copy
import json
import math
import os
import random

from tqdm import tqdm


def main():
    """
    This script creates chosen-rejected pairs for DPO/RPO.
    We match the manifest records with the generated audio files and metrics.
    The script then creates a new manifest with chosen-rejected pairs.
    which is used for training and validation manifest for DPO training.

    Arguments:
        --input_manifest: Path to the input JSON manifest file containing text/context records.
        --generated_audio_dir: Directory containing generated audio files and associated metadata.
        --group_size: Number of records per group used for ranking.
        --cer_threshold: CER threshold for chosen records. Only records with CER <= threshold are retained.
        --val_size: Number of validation samples to retain.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", type=str)
    parser.add_argument(
        "--generated_audio_dir",
        type=str,
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--cer_threshold", type=float, default=0.02)
    parser.add_argument(
        "--min_length_threshold",
        type=float,
        default=1.5,
        help="Minimum length permitted. Set this shorter to allow very short sentences (which can be useful for DPO tuning.",
    )
    parser.add_argument("--val_size", type=int, default=64)
    args = parser.parse_args()

    records = read_manifest(args.input_manifest)
    audio_files, codec_files, metric_files = find_audio_files(args.generated_audio_dir)
    assert len(records) <= len(
        audio_files
    ), "Mismatch between number of records and number of generated audio files {} vs {}".format(
        len(records), len(audio_files)
    )

    for idx, record in tqdm(enumerate(records)):
        if idx % 100 == 0:
            print("At idx: ", idx, len(records))
        record['audio_filepath'] = audio_files[idx]
        record['target_audio_codes_path'] = codec_files[idx]
        with open(metric_files[idx], 'r') as f:
            metrics = json.load(f)
            record['duration'] = metrics['duration']
            record['cer_gts'] = metrics['cer_gt']
            record['wer_gts'] = metrics['wer_gt']
            record['pred_context_similarity'] = metrics['spk_similarity']
            record['pred_transcript'] = metrics['pred_transcript']
            record['gt_transcript'] = metrics['gt_transcript']

    out_manifest_dir = args.generated_audio_dir.replace("/audios", "/manifests")
    if not os.path.exists(out_manifest_dir):
        os.makedirs(out_manifest_dir)

    out_manifest = os.path.join(out_manifest_dir, "manifest_with_metrics.json")
    write_manifest(out_manifest, records)

    group_size = args.group_size
    val_size = args.val_size

    for num_chosen_per_group in [1, 2]:
        all_best_records, all_worst_records = create_chosen_rejected_records(records, group_size, num_chosen_per_group)
        print("Len all_best_records: ", len(all_best_records))
        print("Len all_worst_records: ", len(all_worst_records))
        best_records, worst_records = filter_best_and_worst_records(
            all_best_records, all_worst_records, args.cer_threshold, args.min_length_threshold
        )
        print("Len filtered best_records: ", len(best_records))
        print("Len filtered worst_records: ", len(worst_records))
        worst_records = normalize_rejected_rewards(worst_records)
        paired_records = [
            (best_record, worst_record) for best_record, worst_record in zip(best_records, worst_records)
        ]
        random.shuffle(paired_records)

        final_records = []
        for best_record, worst_record in paired_records:
            assert best_record['reward'] == 1
            assert worst_record['reward'] < 1
            final_records.append(best_record)
            final_records.append(worst_record)

        final_records_val = final_records[:val_size]
        final_records_train = final_records[val_size:]

        train_manifest = os.path.join(
            out_manifest_dir, "dpo_train_manifest_numchosen_per_group_{}.json".format(num_chosen_per_group)
        )
        val_manifest = os.path.join(
            out_manifest_dir, "dpo_val_manifest_numchosen_per_group_{}.json".format(num_chosen_per_group)
        )

        write_manifest(train_manifest, final_records_train)
        write_manifest(val_manifest, final_records_val)


def read_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        records = []
        for line in lines:
            records.append(json.loads(line.strip()))
    return records


def write_manifest(fp, records):
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} records to: {}".format(len(records), fp))


def find_audio_files(directory):
    audio_files = []
    unique_ranks = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                rank_num = int(file.split("Rank")[1].split("_")[0])
                unique_ranks[rank_num] = True
                audio_num = int(file.split(".wav")[0].split("_")[-1])
                audio_files.append((rank_num, audio_num, os.path.join(root, file)))
    ranked_audio_files = []
    for af in audio_files:
        rank, num, path = af
        audio_num = num * len(unique_ranks) + rank
        ranked_audio_files.append((audio_num, path))
    ranked_audio_files = sorted(ranked_audio_files, key=lambda x: x[0])
    ranked_audio_files = [x[1] for x in ranked_audio_files]
    ranked_codec_files = [f.replace(".wav", "_codes.pt") for f in ranked_audio_files]
    metric_files = [f.replace(".wav", "_metrics.json") for f in ranked_audio_files]

    return ranked_audio_files, ranked_codec_files, metric_files


def pareto_rank(items):
    """
    Given a list of (cer, ssim, item_idx), return the list of items
    sorted by their Pareto rank (rank 1 is best). Items in the same
    rank are sorted by ascending cer.

    :param items: List of tuples (cer, ssim, item_idx).
    :return: A list of tuples (rank, cer, ssim, item_idx), sorted first by rank,
             then by ascending cer within the same rank.
    """

    # A helper function to check if item A is dominated by item B
    # A: (cerA, ssimA), B: (cerB, ssimB)
    def is_dominated(A, B):
        assert len(A) == 2
        assert len(B) == 2
        return (B[0] <= A[0]) and (B[1] >= A[1]) and (B != A)
        # Equivalently, check at least one strict inequality:
        # (B[0] < A[0]) or (B[1] > A[1])
        # can be factored into the condition:
        # (B[0] <= A[0]) and (B[1] >= A[1]) and (B != A)

    # Make a working copy so we can remove items
    remaining = items[:]

    ranked_items = []  # Will hold tuples of (rank, cer, ssim, item_idx)
    current_rank = 1

    while remaining:
        # Find all non-dominated items in the current set 'remaining'
        non_dominated = []
        for i in range(len(remaining)):
            dominated = False
            for j in range(len(remaining)):
                if i != j:
                    if is_dominated(remaining[i][:2], remaining[j][:2]):
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(remaining[i])

        # Assign current_rank to all non-dominated items
        # and remove them from remaining
        for nd in non_dominated:
            ranked_items.append((current_rank, nd[0], nd[1], nd[2]))
            remaining.remove(nd)

        current_rank += 1

    # Now sort the ranked items by (rank asc, cer asc, ssim desc)
    ranked_items.sort(key=lambda x: (x[0], x[1], -x[2]))

    return ranked_items


def standard_normal_cdf(z):
    """
    Compute the standard normal cumulative distribution function (CDF) for a given z-score.
    """
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normalize_rejected_rewards(worst_records):
    cer_deltas = [record['cer_delta'] for record in worst_records]
    sim_deltas = [record['sim_delta'] for record in worst_records]
    cer_mean = sum(cer_deltas) / len(cer_deltas)
    cer_std = math.sqrt(sum([(d - cer_mean) ** 2 for d in cer_deltas]) / len(cer_deltas))
    sim_mean = sum(sim_deltas) / len(sim_deltas)
    sim_std = math.sqrt(sum([(d - sim_mean) ** 2 for d in sim_deltas]) / len(sim_deltas))

    for record in worst_records:
        cer_z_score = (record['cer_delta'] - cer_mean) / cer_std
        sim_z_score = (record['sim_delta'] - sim_mean) / sim_std
        record['reward'] = 1.0 - (standard_normal_cdf(cer_z_score) + standard_normal_cdf(sim_z_score))  # Range -1 to 1

    return worst_records


def create_chosen_rejected_records(records_orig, group_size=6, num_chosen_per_group=1):
    records = copy.deepcopy(records_orig)
    assert len(records) % group_size == 0
    num_groups = len(records) // group_size
    best_records = []
    worst_records = []
    num_skipped = 0

    if num_chosen_per_group == 1:
        chosen_group_indices = [0]
        rejected_group_indices = [group_size - 1]
    elif num_chosen_per_group == 2:
        chosen_group_indices = [0, 1]
        rejected_group_indices = [group_size - 1, group_size - 2]
    else:
        raise ValueError("num_chosen_per_group must be 1 or 2")

    for gidx in range(num_groups):
        gsi = gidx * group_size
        gei = (gidx + 1) * group_size
        group = records[gsi:gei]

        cer_sim_indices = []
        skip_group = False
        for sidx, record in enumerate(group):
            if record['pred_transcript'] == "<INVALID>":
                print(f"Skipping group starting at index {gsi} due to invalid entries.")
                num_skipped += len(group)
                skip_group = True
                break
            cer_sim_indices.append((record['cer_gts'], record['pred_context_similarity'], sidx))
        if skip_group:
            continue
        cer_sim_indices_orig = copy.deepcopy(cer_sim_indices)
        cer_sim_indices = pareto_rank(cer_sim_indices)

        for cgi in chosen_group_indices:
            for rji in rejected_group_indices:
                best_record = group[cer_sim_indices[cgi][3]]
                worst_record = group[cer_sim_indices[rji][3]]
                best_record['reward'] = 1
                reward_delta = (worst_record['cer_gts'] - best_record['cer_gts']) + (
                    best_record['pred_context_similarity'] - worst_record['pred_context_similarity']
                )
                if (
                    reward_delta <= 0
                    or worst_record['cer_gts'] < best_record['cer_gts']
                    or worst_record['pred_context_similarity'] > best_record['pred_context_similarity']
                ):
                    print(
                        "Warning reward_delta is not positive",
                        reward_delta,
                        best_record['cer_gts'],
                        worst_record['cer_gts'],
                        best_record['pred_context_similarity'],
                        worst_record['pred_context_similarity'],
                    )
                    print(cer_sim_indices_orig)
                    print(cer_sim_indices)
                else:
                    # Never add pairs in which rejected has better CER than chosen or better context similarity
                    reward_delta = max(0.001, reward_delta)
                    worst_record['reward'] = 1.0 - reward_delta
                    worst_record['cer_delta'] = worst_record['cer_gts'] - best_record['cer_gts']
                    worst_record['sim_delta'] = (
                        best_record['pred_context_similarity'] - worst_record['pred_context_similarity']
                    )
                    best_records.append(best_record)
                    worst_records.append(worst_record)

    print(f"Skipped {num_skipped} records due to invalid entries.")
    return best_records, worst_records


def filter_best_and_worst_records(best_records, worst_records, cer_threshold=0.02, min_length_threshold=1.5):
    ridx = 0
    filtered_best_records = []
    filtered_worst_records = []
    best_cer_avg = 0.0
    worst_cer_avg = 0.0
    skipped_records = 0
    while ridx < len(best_records):
        # print(ridx, len(best_records))
        best_record = best_records[ridx]
        if best_record['cer_gts'] < cer_threshold:
            worst_record = worst_records[ridx]
            if (worst_record['duration'] > 19.0 or best_record['duration'] > 19.0) or (
                worst_record['duration'] < min_length_threshold or best_record['duration'] < min_length_threshold
            ):
                skipped_records += 1
                ridx += 1
                continue
            assert best_record['cer_gts'] <= worst_record['cer_gts']
            if worst_record['cer_gts'] == best_record['cer_gts']:
                assert worst_record['pred_context_similarity'] <= best_record['pred_context_similarity']

            filtered_best_records.append(best_record)
            filtered_worst_records.append(worst_record)
            best_cer_avg += best_record['cer_gts']
            worst_cer_avg += worst_record['cer_gts']
        ridx += 1

    best_cer_avg /= len(filtered_best_records)
    worst_cer_avg /= len(filtered_worst_records)
    print(f"Best CER avg: {best_cer_avg}, Worst CER avg: {worst_cer_avg}")
    return filtered_best_records, filtered_worst_records


if __name__ == "__main__":
    main()
