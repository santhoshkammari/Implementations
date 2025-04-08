"""
Common Search Algorithms

This file provides stubs for Linear Search, Binary Search, BFS, and DFS algorithms.
Students can implement the logic within these functions.

For interviews, you should be most familiar with Linear Search, Binary Search, BFS, and DFS, as these are the most commonly asked about. BFS and DFS are typically used for graph traversal.

# Linear Search
# Binary Search
# Breadth-First Search (BFS)
# Depth-First Search (DFS)

Algorithm           | Best Time     | Average Time  | Worst Time    | Space
--------------------|---------------|---------------|---------------|-------
Linear Search       | O(1)          | O(n)          | O(n)          | O(1)
Binary Search       | O(1)          | O(log n)      | O(log n)      | O(1) (Iterative), O(log n) (Recursive)
BFS (Graph)         | O(V + E)      | O(V + E)      | O(V + E)      | O(V)
DFS (Graph)         | O(V + E)      | O(V + E)      | O(V + E)      | O(V)


Note: Time complexity for Binary search assumes the input array is sorted.
V = number of vertices, E = number of edges in the graph for BFS/DFS.
"""

def linear_search(arr, target):
    """Searches for a target value in an array using Linear Search.

    Args:
        arr: The list or array to search within.
        target: The value to search for.

    Returns:
        The index of the target if found, otherwise -1.
    """
    pass

def binary_search(arr, target):
    """Searches for a target value in a sorted array using Binary Search.

    Args:
        arr: The sorted list or array to search within.
        target: The value to search for.

    Returns:
        The index of the target if found, otherwise -1.
    """
    # Note: arr must be sorted for binary search to work correctly.
    pass

def bfs(graph, start_node):
    """Performs Breadth-First Search on a graph.

    Args:
        graph: The graph represented, e.g., as an adjacency list (dict).
        start_node: The node to start the search from.

    Returns:
        A list of nodes in the order they were visited, or potentially
        other results depending on the specific BFS application (e.g., path).
        Returns an empty list or None if the start node isn't in the graph.
    """
    # Note: Assumes graph is represented as an adjacency list (dictionary).
    pass

def dfs(graph, start_node, visited=None):
    """Performs Depth-First Search on a graph (recursive implementation).

    Args:
        graph: The graph represented, e.g., as an adjacency list (dict).
        start_node: The node to start the search from.
        visited: A set to keep track of visited nodes (used in recursion).
                 Should usually be None when initially called.

    Returns:
        A list of nodes in the order they were visited (pre-order traversal),
        or potentially other results. Returns an empty list or None if the
        start node isn't in the graph.
    """
    # Note: Assumes graph is represented as an adjacency list (dictionary).
    if visited is None:
        visited = set()
    pass


# Removed Jump Search, Interpolation Search, Exponential Search, Ternary Search

if __name__ == "__main__":
    # Example usage (students can modify this)

    # --- Array Search Examples ---
    my_array = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    target_value = 23

    # --- Test Linear Search ---
    # index = linear_search(my_array, target_value)
    # print(f"Linear Search: Target {target_value} found at index: {index}")

    # --- Test Binary Search ---
    # index = binary_search(my_array, target_value)
    # print(f"Binary Search: Target {target_value} found at index: {index}")


    # --- Graph Search Examples ---
    # Example graph represented as an adjacency list
    my_graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    start_node_graph = 'A'

    # --- Test BFS ---
    # visited_order_bfs = bfs(my_graph, start_node_graph)
    # print(f"BFS starting from {start_node_graph}: {visited_order_bfs}")

    # --- Test DFS ---
    # visited_order_dfs = dfs(my_graph, start_node_graph)
    # print(f"DFS starting from {start_node_graph}: {visited_order_dfs}")


    print("Search algorithm stubs created/updated. Implement the functions to test.")

