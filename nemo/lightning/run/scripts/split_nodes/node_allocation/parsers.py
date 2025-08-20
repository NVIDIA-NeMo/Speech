"""Parsers and utility functions for node allocation.

This module contains functions for parsing topology files, node lists,
and other utility functions needed for node allocation.
"""

import re
from typing import List, Dict, Optional, Tuple, Set


def expand_nodes(raw_string: str) -> List[str]:
    """Expand a string containing node specifications into a list of node names.
    
    Args:
        raw_string: String containing node specifications, e.g. "pool1-1195,pool1-[2110-2111]"
                    or "hgx-isr1-[001-008]"
    
    Returns:
        List of expanded node names
    """
    # Extract the nodes part after "Nodes="
    nodes_part = raw_string.split("Nodes=")[1].split()[0]
    
    expanded = []
    # Split on commas to handle each node specification
    for spec in nodes_part.split(','):
        # Handle range format: prefix-[start-end]
        if '[' in spec and ']' in spec:
            # Extract prefix (e.g., "pool1-" or "hgx-isr1-")
            prefix = spec.split('[')[0]
            range_part = spec.split('[')[1].split(']')[0]
            
            if '-' in range_part:
                start_str, end_str = range_part.split('-')
                start = int(start_str)
                end = int(end_str)
                # Determine padding based on start number's string representation
                padding = len(start_str)
                for num in range(start, end + 1):
                    expanded.append(f"{prefix}{num:0{padding}d}")
            else:
                # Single number in brackets
                num = int(range_part)
                # Determine padding based on the number's length
                padding = len(range_part)
                expanded.append(f"{prefix}{num:0{padding}d}")
        else:
            # Handle individual node format without ranges
            expanded.append(spec)
    
    return expanded


def parse_topology_file(topology_file: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Parse a topology file and return node-to-switch mapping and switch relationships.
    
    Args:
        topology_file: Path to the topology file
        
    Returns:
        Tuple containing:
        - node_to_switch: Dictionary mapping node names to their switch
        - switch_hierarchy: Dict containing switch parent-child relationships
    """
    with open(topology_file) as f:
        topo_output = f.read().strip().splitlines()
    
    # Parse topology to map nodes to switches
    node_to_switch: Dict[str, str] = {}
    switch_hierarchy: Dict[str, Dict[str, str]] = {
        'parents': {},  # Maps switches to their parents
        'children': {}  # Maps switches to their children
    }
    
    current_switch = None
    
    for line in topo_output:
        # Look for switch definitions - match any switch name after SwitchName=
        m = re.search(r'SwitchName=([^\s]+) Level=(\d+)', line)
        if m:
            current_switch = m.group(1)
            switch_level = int(m.group(2))
            
            # For leaf switches (Level 0), find their parents
            if switch_level == 0:
                parent_match = re.search(r'Switches=(.*?)$', line)
                if parent_match:
                    parents = parent_match.group(1).strip()
                    switch_hierarchy['parents'][current_switch] = parents
                    
                    # Add this switch as a child of its parents
                    for parent in parents.split(','):
                        if parent not in switch_hierarchy['children']:
                            switch_hierarchy['children'][parent] = []
                        switch_hierarchy['children'][parent].append(current_switch)
        
        # Look for node definitions and map them to the current switch
        if "Nodes=" in line and current_switch:
            expanded = expand_nodes(line)
            for node in expanded:
                node_to_switch[node] = current_switch
    
    return node_to_switch, switch_hierarchy


def parse_allocated_nodes(allocated_nodes_file: str) -> List[str]:
    """Parse a file containing allocated nodes.
    
    Args:
        allocated_nodes_file: Path to the file containing the list of allocated nodes
        
    Returns:
        List of node names
    """
    with open(allocated_nodes_file) as f:
        allocated_nodes = f.read().strip().split()
    return allocated_nodes


def parse_node_input(node_input: str, is_file: bool = False) -> List[str]:
    """Parse node input, either from a file or directly from a string.
    
    Args:
        node_input: Either a file path or a direct node list string
        is_file: Whether the input is a file path
        
    Returns:
        List of node names
    """
    if is_file:
        with open(node_input) as f:
            nodes = f.read().strip().split()
        return nodes
    else:
        # Direct input string, could be compressed, so don't split
        return [node_input]


def group_nodes_by_switch(allocated_nodes: List[str], node_to_switch: Dict[str, str]) -> Dict[str, List[str]]:
    """Group allocated nodes by their switch.
    
    Args:
        allocated_nodes: List of allocated node names
        node_to_switch: Dictionary mapping nodes to switches
        
    Returns:
        Dictionary mapping switches to their list of allocated nodes
    """
    switch_to_nodes: Dict[str, List[str]] = {}
    missing_nodes: List[str] = []
    
    for node in allocated_nodes:
        switch = node_to_switch.get(node)
        if switch:
            switch_to_nodes.setdefault(switch, []).append(node)
        else:
            missing_nodes.append(node)
    
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} node(s) not found in topology!")
    
    return switch_to_nodes


def calculate_switch_distance(switch1: str, switch2: str, switch_hierarchy: Dict[str, Dict[str, str]]) -> int:
    """Calculate the 'distance' between two switches based on topology.
    
    Distance is defined as:
    - 0 if switches are the same
    - 1 if they share a direct parent
    - 2 if they only share the core switch
    
    Args:
        switch1: First switch name
        switch2: Second switch name
        switch_hierarchy: Dict containing switch parent-child relationships
        
    Returns:
        Distance between the switches (0, 1, or 2)
    """
    if switch1 == switch2:
        return 0
        
    # Get parents
    parents = switch_hierarchy.get('parents', {})
    
    # If we don't have parent info, assume maximum distance
    if switch1 not in parents or switch2 not in parents:
        return 2
        
    parent1 = parents.get(switch1)
    parent2 = parents.get(switch2)
    
    # If they share a direct parent, they're close
    if parent1 and parent2 and parent1 == parent2:
        return 1
    
    # Otherwise, they meet at the core
    return 2 