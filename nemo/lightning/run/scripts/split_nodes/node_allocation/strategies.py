"""Node allocation strategies.

This module contains different strategies for allocating nodes between workloads.
"""

import sys
from typing import List, Dict, Tuple, Set, Optional

from .parsers import calculate_switch_distance


def _handle_single_switch_case(
    switch_to_nodes: Dict[str, List[str]],
    workload_a_nodes: int
) -> Tuple[List[str], List[str]]:
    """Handle the special case where all nodes are under a single leaf switch.
    
    In this case, we simply allocate the first N nodes to workload A and the rest to workload B.
    
    Args:
        switch_to_nodes: Dictionary mapping switches to their allocated nodes
        workload_a_nodes: Required number of nodes for workload A
        
    Returns:
        tuple: (workload_a, workload_b) lists of nodes
    """
    # Get all nodes and sort them for deterministic behavior
    all_nodes = []
    for nodes in switch_to_nodes.values():
        all_nodes.extend(sorted(nodes))
    
    # Allocate first N nodes to workload A, rest to workload B
    workload_a = all_nodes[:workload_a_nodes]
    workload_b = all_nodes[workload_a_nodes:]
    
    return workload_a, workload_b


def evenly_split_nodes_between_workloads(
    switch_to_nodes: Dict[str, List[str]], 
    workload_a_nodes: int, 
    workload_b_nodes: int
) -> Tuple[List[str], List[str]]:
    """Split nodes evenly between workloads, balancing across switches.
    
    This strategy distributes nodes between workloads A and B while trying to maintain
    balance across switches. For each switch, it allocates nodes proportionally based
    on the overall workload requirements.
    
    Args:
        switch_to_nodes: Dictionary mapping switches to their allocated nodes
        workload_a_nodes: Required number of nodes for workload A
        workload_b_nodes: Required number of nodes for workload B
        
    Returns:
        tuple: (workload_a, workload_b) lists of nodes
    """
    # Special case: If all nodes are under a single leaf switch
    if len(switch_to_nodes) == 1:
        return _handle_single_switch_case(switch_to_nodes, workload_a_nodes)
    
    # Prepare workload lists
    workload_a: List[str] = []
    workload_b: List[str] = []
    
    # Track the balance of nodes between workloads
    total_a: int = 0
    total_b: int = 0
    
    # Calculate total nodes to allocate
    total_nodes: int = workload_a_nodes + workload_b_nodes
    
    # Collect all nodes from all switches for allocation
    all_nodes: List[str] = []
    for switch in sorted(switch_to_nodes.keys()):
        nodes = sorted(switch_to_nodes[switch])
        all_nodes.extend(nodes)
    
    # Track allocated nodes to identify unallocated ones later
    allocated_nodes: Set[str] = set()
    
    # Process switches in a deterministic order
    for switch in sorted(switch_to_nodes.keys()):
        nodes = sorted(switch_to_nodes[switch])  # Sort for deterministic behavior
        num_nodes = len(nodes)
        
        # Skip if we don't need any more nodes for either workload
        if total_a >= workload_a_nodes and total_b >= workload_b_nodes:
            break
            
        # Calculate how many nodes should go to each workload
        nodes_left_a = max(0, workload_a_nodes - total_a)
        nodes_left_b = max(0, workload_b_nodes - total_b)
        
        # If we have enough nodes, allocate as needed
        if num_nodes <= nodes_left_a + nodes_left_b:
            # Figure out how to divide this switch's nodes
            if nodes_left_a == 0:
                # All to B
                a_count = 0
                b_count = min(num_nodes, nodes_left_b)
            elif nodes_left_b == 0:
                # All to A
                a_count = min(num_nodes, nodes_left_a)
                b_count = 0
            else:
                # Divide proportionally based on remaining needs
                a_ratio = nodes_left_a / (nodes_left_a + nodes_left_b)
                a_count = round(num_nodes * a_ratio)
                
                # Adjust to ensure we don't exceed limits
                a_count = min(a_count, nodes_left_a)
                b_count = min(num_nodes - a_count, nodes_left_b)
        else:
            # Not enough nodes left, proportionally allocate what we have
            if workload_a_nodes == 0:
                a_count = 0
                b_count = num_nodes
            elif workload_b_nodes == 0:
                a_count = num_nodes
                b_count = 0
            else:
                a_ratio = workload_a_nodes / (workload_a_nodes + workload_b_nodes)
                a_count = round(num_nodes * a_ratio)
                a_count = min(a_count, nodes_left_a)
                b_count = min(num_nodes - a_count, nodes_left_b)
        
        # Assign nodes to workloads
        workload_a.extend(nodes[:a_count])
        workload_b.extend(nodes[a_count:a_count+b_count])
        
        # Track which nodes have been allocated
        allocated_nodes.update(nodes[:a_count+b_count])
        
        # Update totals
        total_a += a_count
        total_b += b_count
    
    # Find unallocated nodes and assign them to workload B
    unallocated_nodes = sorted(set(all_nodes) - allocated_nodes)
    if unallocated_nodes:
        # print(f"Info: Assigning {len(unallocated_nodes)} unallocated nodes to workload B", file=sys.stderr)
        workload_b.extend(unallocated_nodes)
        
    return workload_a, workload_b


def compact_split_nodes_between_workloads(
    switch_to_nodes: Dict[str, List[str]], 
    node_to_switch: Dict[str, str], 
    switch_hierarchy: Dict[str, Dict[str, str]], 
    workload_a_nodes: int,
    workload_b_nodes: int
) -> Tuple[List[str], List[str]]:
    """Split nodes between workloads using a compact allocation strategy.
    
    This strategy:
    1. Finds the leaf switch with the most allocated nodes
    2. Places 1 node from this switch in workload A
    3. Places nodes from the largest switch in workload B up to the number of nodes needed
    4. Fills workload A with nodes from nearby switches (based on topology)
    5. If needed, uses more nodes from the largest switch to meet workload A requirements
    6. Places remaining nodes in workload B up to the number of nodes needed
    
    Args:
        switch_to_nodes: Dictionary mapping switches to their allocated nodes
        node_to_switch: Dictionary mapping nodes to their switches
        switch_hierarchy: Dict containing switch hierarchy information
        workload_a_nodes: Number of nodes required for workload A
        workload_b_nodes: Number of nodes required for workload B
    Returns:
        tuple: (workload_a, workload_b) lists of nodes
    """
    # Special case: If all nodes are under a single leaf switch
    if len(switch_to_nodes) == 1:
        return _handle_single_switch_case(switch_to_nodes, workload_a_nodes)
    
    workload_a: List[str] = []
    workload_b: List[str] = []
    
    # Collect all nodes for tracking unallocated ones
    all_nodes: List[str] = []
    for nodes in switch_to_nodes.values():
        all_nodes.extend(nodes)
    
    # Find the leaf switch with the most allocated nodes
    largest_switch: Optional[str] = None
    most_nodes: int = 0
    for switch, nodes in switch_to_nodes.items():
        if len(nodes) > most_nodes:
            most_nodes = len(nodes)
            largest_switch = switch
    
    # If we have no nodes, return empty workloads
    if not largest_switch:
        print(f"Info: Unable to find the largest switch, returning empty workloads", file=sys.stderr)
        return workload_a, workload_b
    
    if len(switch_to_nodes[largest_switch]) < 1:
        print(f"Info: Got only {len(switch_to_nodes[largest_switch])} nodes from the largest switch, returning empty workloads", file=sys.stderr)
        return workload_a, workload_b
    
    # Sort nodes for deterministic behavior
    sorted_nodes = sorted(switch_to_nodes[largest_switch])
    print(f"Info: Largest switch: {largest_switch}, node count: {len(sorted_nodes)}")
    
    # Place 1 node from the largest switch in workload A
    workload_a.append(sorted_nodes[0])
    
    # Keep a copy of the remaining nodes from the largest switch for potential later use
    largest_switch_remaining_nodes = sorted_nodes[1:]

    # Take nodes from the largest switch for workload B up to the number of nodes needed
    nodes_needed_for_b_from_largest_switch = min(workload_b_nodes, len(largest_switch_remaining_nodes))
    workload_b.extend(largest_switch_remaining_nodes[:nodes_needed_for_b_from_largest_switch])
    largest_switch_remaining_nodes = largest_switch_remaining_nodes[nodes_needed_for_b_from_largest_switch:]
    
    # Sort remaining switches by "distance" from the largest switch
    remaining_switches = [(switch, calculate_switch_distance(largest_switch, switch, switch_hierarchy)) 
                          for switch in switch_to_nodes.keys() 
                          if switch != largest_switch]
    remaining_switches.sort(key=lambda x: x[1])  # Sort by distance
    
    # Fill workload A with nodes from nearby switches
    nodes_needed = workload_a_nodes - len(workload_a)
    
    for switch, _ in remaining_switches:
        # Sort nodes in this switch for deterministic behavior
        switch_nodes = sorted(switch_to_nodes[switch])
        
        # If we can take all nodes from this switch
        if len(switch_nodes) <= nodes_needed:
            workload_a.extend(switch_nodes)
            nodes_needed -= len(switch_nodes)
        else:
            # Take only as many as we need
            workload_a.extend(switch_nodes[:nodes_needed])
            nodes_needed = 0
        
        # If we have enough nodes, stop
        if nodes_needed <= 0:
            break
    
    # If we still need more nodes for workload A, use nodes from the largest switch
    if nodes_needed > 0:
        # Take what we need from the largest switch's remaining nodes
        workload_a.extend(largest_switch_remaining_nodes[:nodes_needed])
    
    nodes_needed_for_b = workload_b_nodes - len(workload_b)
    if nodes_needed_for_b > 0:
        # Find unallocated nodes (nodes not yet assigned to either workload)
        allocated_nodes = set(workload_a + workload_b)
        unallocated_nodes = sorted(set(all_nodes) - allocated_nodes)
        # print(f"Info: Assigning {len(unallocated_nodes)} unallocated nodes to workload B", file=sys.stderr)
        workload_b.extend(unallocated_nodes[:nodes_needed_for_b])
    
    return workload_a, workload_b 