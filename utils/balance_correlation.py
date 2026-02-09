import networkx as nx
import numpy as np


def calculate_balance_correlation_dekker(signed_graph, triad_signature):
    """
    Calculates the balance correlation for a signed graph using NumPy matrices
    and matrix operations, based on Dekker et al. (2024).

    Args:
        signed_graph: A NetworkX graph where edges have a 'sign' attribute
                      (1 for positive, -1 for negative, 0 for neutral).
        triad_signature: A 3-tuple string (e.g., "p.pp", "n.pn", "z.nz") representing the
                        triad configuration.

    Returns:
        The balance correlation coefficient for the specified triad signature.
        Returns np.nan if there are not enough triads.
    """

    # Create a mapping from node labels to integer indices
    node_to_index = {node: i for i, node in enumerate(signed_graph.nodes())}
    num_nodes = len(signed_graph)
    p_matrix = np.zeros((num_nodes, num_nodes))
    n_matrix = np.zeros((num_nodes, num_nodes))
    z_matrix = np.zeros((num_nodes, num_nodes))

    for i in signed_graph.nodes():
        for j in signed_graph.nodes():
            if signed_graph.has_edge(i, j):
                sign = signed_graph[i][j].get('sign', 0)
                # Use the mapping to get integer indices
                row_index = node_to_index[i]
                col_index = node_to_index[j]
                if sign == 1:
                    p_matrix[row_index, col_index] = 1
                elif sign == -1:
                    n_matrix[row_index, col_index] = 1
                elif sign == 0:
                    z_matrix[row_index, col_index] = 1

    # Calculate 2-paths using matrix multiplication
    two_path_matrix = None
    rel_matrix = None
    if triad_signature[1] == 'p':
        rel_matrix = p_matrix
    elif triad_signature[1] == 'n':
        rel_matrix = n_matrix
    elif triad_signature[1] == 'z':
        rel_matrix = z_matrix

    two_path_matrix_1 = None
    if triad_signature[2] == 'p':
        two_path_matrix_1 = p_matrix
    elif triad_signature[2] == 'n':
        two_path_matrix_1 = n_matrix
    elif triad_signature[2] == 'z':
        two_path_matrix_1 = z_matrix

    two_path_matrix_2 = None
    if triad_signature[3] == 'p':
        two_path_matrix_2 = p_matrix
    elif triad_signature[3] == 'n':
        two_path_matrix_2 = n_matrix
    elif triad_signature[3] == 'z':
        two_path_matrix_2 = z_matrix

    rs_matrix = np.dot(two_path_matrix_1, two_path_matrix_2)

    # Extract relevant vectors and calculate correlation
    ego_alter_vector = None
    if triad_signature[0] == 'p':
        ego_alter_vector = p_matrix.flatten()
    elif triad_signature[0] == 'n':
        ego_alter_vector = n_matrix.flatten()
    elif triad_signature[0] == 'z':
        ego_alter_vector = z_matrix.flatten()

    two_path_vector = rs_matrix.flatten()

    # Remove diagonal elements (self-loops)
    mask = ~np.eye(num_nodes, dtype=bool)
    ego_alter_vector = ego_alter_vector[mask.flatten()]
    two_path_vector = two_path_vector[mask.flatten()]

    if len(set(two_path_vector)) > 1 and len(set(ego_alter_vector)) > 1:
        correlation = np.corrcoef(ego_alter_vector, two_path_vector)[0, 1]
        return correlation
    else:
        return np.nan


if __name__ == '__main__':
    # Example Usage:
    signed_graph = nx.Graph()
    signed_graph.add_edge('A', 'B', sign=1)  # p
    signed_graph.add_edge('A', 'C', sign=-1)  # n
    signed_graph.add_edge('B', 'C', sign=0)  # z
    signed_graph.add_edge('A', 'D', sign=1)  # p
    signed_graph.add_edge('B', 'D', sign=-1)  # n
    signed_graph.add_edge('C', 'D', sign=0)  # z

    # Calculate balance correlations for different triad signatures
    print(f"Balance Correlation for <p.pn>: {calculate_balance_correlation_dekker_matrix(signed_graph, 'p.pn')}")
    print(f"Balance Correlation for <z.pp>: {calculate_balance_correlation_dekker_matrix(signed_graph, 'z.pp')}")
    print(f"Balance Correlation for <n.nz>: {calculate_balance_correlation_dekker_matrix(signed_graph, 'n.nz')}")
    print(f"Balance Correlation for <p.pp>: {calculate_balance_correlation_dekker_matrix(signed_graph, 'p.pp')}")