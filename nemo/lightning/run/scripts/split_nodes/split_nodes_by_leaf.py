#!/usr/bin/env python3
"""Split nodes by leaf switch for workload distribution.

This script takes a list of allocated nodes and a cluster topology file,
and distributes the nodes between two workloads based on their position
in the network topology.

Example usage:
    python split_nodes_by_leaf.py allocated_nodes.txt topology.txt --workload-a-nodes 3 --workload-b-nodes 5
"""

import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import click
from node_allocation.parsers import group_nodes_by_switch, parse_allocated_nodes, parse_topology_file
from node_allocation.strategies import compact_split_nodes_between_workloads, evenly_split_nodes_between_workloads


def process_files(
    allocated_nodes_file: str,
    topology_file: str,
    output_file: str,
    strategy: str = 'compact',
    workload_a_nodes: Optional[int] = None,
    workload_b_nodes: Optional[int] = None,
    victim_nodes: int = 1,
) -> None:
    """Core processing logic that orchestrates node allocation.

    This function reads input files, validates arguments, and applies
    the appropriate allocation strategy.

    Args:
        allocated_nodes_file: Path to file containing list of allocated nodes
        topology_file: Path to file containing cluster topology information
        strategy: 'even' for balanced allocation, 'compact' for compact allocation
        workload_a_nodes: Required number of nodes for workload A
        workload_b_nodes: Required number of nodes for workload B

    Returns:
        None: Results are printed to standard output
    """
    # Parse input files
    allocated_nodes = parse_allocated_nodes(allocated_nodes_file)
    node_to_switch, switch_hierarchy = parse_topology_file(topology_file)

    # Group allocated nodes by switch
    switch_to_nodes = group_nodes_by_switch(allocated_nodes, node_to_switch)
    print(switch_to_nodes)

    total_nodes = len(allocated_nodes)

    # Validate node count requirements
    if workload_a_nodes is None or workload_b_nodes is None:
        print("Error: Both workload_a_nodes and workload_b_nodes must be specified", file=sys.stderr)
        sys.exit(1)

    # Validate the specified node counts
    if workload_a_nodes + workload_b_nodes > total_nodes:
        print(
            f"Error: Requested nodes ({workload_a_nodes + workload_b_nodes}) exceeds available nodes ({total_nodes})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply the selected allocation strategy
    if strategy == 'even':
        # Use the even split method
        workload_a, workload_b = evenly_split_nodes_between_workloads(
            switch_to_nodes, workload_a_nodes, workload_b_nodes
        )
    else:
        # Use the compact allocation strategy
        workload_a, workload_b = compact_split_nodes_between_workloads(
            switch_to_nodes, node_to_switch, switch_hierarchy, workload_a_nodes, workload_b_nodes, victim_nodes
        )

    if len(workload_a) != workload_a_nodes or len(workload_b) != workload_b_nodes:
        print(
            f"Error: Requested {workload_a_nodes} nodes for workload A and {workload_b_nodes} nodes for workload B, but got {len(workload_a)} and {len(workload_b)} nodes respectively",
            file=sys.stderr,
        )
        sys.exit(1)

    from collections import Counter

    # Get unique switches for each workload
    workload_a_switches = Counter(node_to_switch[node] for node in workload_a)
    workload_b_switches = Counter(node_to_switch[node] for node in workload_b)

    print("Switch Distribution:")
    print("Workload A switches:")
    for switch, count in workload_a_switches.items():
        print(f"{switch}: {count}")
    print("Workload B switches:")
    for switch, count in workload_b_switches.items():
        print(f"{switch}: {count}")

    # Output comma-separated nodelists
    with open(output_file, 'w') as f:
        print(",".join(workload_a), file=f)
        print(",".join(workload_b), file=f)


@click.command()
@click.option('--allocated-nodes-file', type=click.Path(exists=True), required=True)
@click.option('--topology-file', type=click.Path(exists=True), required=True)
@click.option(
    '--strategy',
    type=click.Choice(['even', 'compact']),
    default='compact',
    help='Node allocation strategy: "even" balances nodes across switches, '
    '"compact" places most of workload A nodes on same/nearby switches',
    required=True,
)
@click.option('--workload-a-nodes', type=int, required=True, help='Number of nodes required for workload A')
@click.option('--workload-b-nodes', type=int, required=True, help='Number of nodes required for workload B')
@click.option(
    '--victim-nodes',
    type=int,
    required=True,
    default=1,
    help='Number of victim nodes required in workload A in compact strategy',
)
@click.option('--output-file', type=click.Path(exists=False), required=False, help='File to write the output to')
def main(
    allocated_nodes_file: str,
    topology_file: str,
    strategy: str,
    workload_a_nodes: int,
    workload_b_nodes: int,
    output_file: str,
    victim_nodes: int,
) -> None:
    """Split nodes by leaf switch for workload distribution.

    ALLOCATED_NODES_FILE: File containing list of allocated nodes
    TOPOLOGY_FILE: File containing cluster topology information
    """
    process_files(
        allocated_nodes_file, topology_file, output_file, strategy, workload_a_nodes, workload_b_nodes, victim_nodes
    )


def test_main() -> None:
    """Test the main function with sample inputs."""
    # Print debug info first
    print(f"Current working directory: {os.getcwd()}")
    nodes_path = Path('allocated_nodes.txt')
    topo_path = Path('topology.txt')
    print(f"Nodes file exists: {nodes_path.exists()}")
    print(f"Topology file exists: {topo_path.exists()}")

    a_nodes = 16
    b_nodes = 60
    victim_nodes = 1
    # Test even strategy with specific node counts
    print(f"\n=== Testing EVEN strategy with {a_nodes} nodes for A, {b_nodes} for B ===")
    split_nodes_path = Path('split-nodes-even.txt')
    process_files(str(nodes_path), str(topo_path), str(split_nodes_path), 'even', a_nodes, b_nodes)

    with open(split_nodes_path, 'r') as f:
        split_nodes = f.read().split('\n')
    print(f"Split nodes: {split_nodes}")

    print(f"Workload A ({a_nodes} nodes):", split_nodes[0])
    print(f"Workload B ({b_nodes} nodes):", split_nodes[1])

    node_to_switch, _ = parse_topology_file(str(topo_path))

    # Print switch distribution for each workload
    workload_a_nodes = split_nodes[0].split(',') if split_nodes[0] else []
    workload_b_nodes = split_nodes[1].split(',') if split_nodes[1] else []
    print(f"Workload A nodes: '{workload_a_nodes}'")
    print(f"Workload B nodes: '{workload_b_nodes}'")

    # Test compact strategy with specific node counts
    print(f"\n=== Testing COMPACT strategy with {a_nodes} nodes for A, {b_nodes} for B ===")
    split_nodes_path = Path('split-nodes-compact.txt')
    process_files(str(nodes_path), str(topo_path), str(split_nodes_path), 'compact', a_nodes, b_nodes, victim_nodes)

    with open(split_nodes_path, 'r') as f:
        split_nodes = f.read().split('\n')

    print(f"Workload A ({a_nodes} nodes):", split_nodes[0])
    print(f"Workload B ({b_nodes} nodes):", split_nodes[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main()
    else:
        main()
