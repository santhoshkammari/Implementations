"""
Common Data Structures

# Arrays & Lists
# Linked Lists (Singly, Doubly, Circular)
# Stacks
# Queues (Simple, Priority, Deque)
# Hash Tables/Maps
# Binary Trees
# Binary Search Trees
# AVL Trees
# Heaps (Min, Max)
# Tries
# Graphs (Adjacency Matrix, Adjacency List)
# Disjoint Set (Union-Find)

Data Structure             | Access    | Search    | Insertion | Deletion  | Space
---------------------------|-----------|-----------|-----------|-----------|--------
Array                      | O(1)      | O(n)      | O(n)      | O(n)      | O(n)
Linked List                | O(n)      | O(n)      | O(1)      | O(1)      | O(n)
Stack                      | O(n)      | O(n)      | O(1)      | O(1)      | O(n)
Queue                      | O(n)      | O(n)      | O(1)      | O(1)      | O(n)
Hash Table                 | N/A       | O(1)*     | O(1)*     | O(1)*     | O(n)
Binary Search Tree         | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n)
AVL Tree                   | O(log n)  | O(log n)  | O(log n)  | O(log n)  | O(n)
Binary Heap                | O(1)**    | O(n)      | O(log n)  | O(log n)  | O(n)
Trie                       | O(k)      | O(k)      | O(k)      | O(k)      | O(n*k)
Graph (Adjacency Matrix)   | O(1)      | O(1)      | O(1)      | O(1)      | O(n²)
Graph (Adjacency List)     | O(n)      | O(n)      | O(1)      | O(n)      | O(n+e)
Disjoint Set               | O(α(n))   | O(α(n))   | O(α(n))   | N/A       | O(n)

* Average case, can be O(n) in worst case
** Only for the min/max element
k = length of key
n = number of nodes/elements
e = number of edges
α(n) = Inverse Ackermann function (practically constant)
"""

class Node:
    """Basic node for linked data structures."""
    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

class SinglyLinkedList:
    """Singly Linked List implementation.
    
    A linear data structure where elements are not stored in contiguous memory.
    Each element points to the next element.
    
    Operations:
    - append: add to end - O(n) or O(1) with tail pointer
    - prepend: add to beginning - O(1)
    - delete: remove element - O(n)
    - search: find element - O(n)
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, data):
        """Add element to end of the list."""
        pass
    
    def prepend(self, data):
        """Add element to beginning of the list."""
        pass
    
    def delete(self, data):
        """Remove element from the list."""
        pass
    
    def search(self, data):
        """Find element in the list."""
        pass

class DoublyLinkedList:
    """Doubly Linked List implementation.
    
    Similar to singly linked list but with pointers to both next and previous nodes.
    
    Operations:
    - append: add to end - O(1) with tail pointer
    - prepend: add to beginning - O(1)
    - delete: remove element - O(n) to find, O(1) to remove
    - search: find element - O(n)
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, data):
        """Add element to end of the list."""
        pass
    
    def prepend(self, data):
        """Add element to beginning of the list."""
        pass
    
    def delete(self, data):
        """Remove element from the list."""
        pass
    
    def search(self, data):
        """Find element in the list."""
        pass

class Stack:
    """Stack implementation (LIFO - Last In First Out).
    
    Operations:
    - push: add element to top - O(1)
    - pop: remove element from top - O(1)
    - peek: view top element - O(1)
    - is_empty: check if stack is empty - O(1)
    """
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add element to top of stack."""
        pass
    
    def pop(self):
        """Remove and return element from top of stack."""
        pass
    
    def peek(self):
        """Return top element without removing it."""
        pass
    
    def is_empty(self):
        """Check if stack is empty."""
        pass

class Queue:
    """Queue implementation (FIFO - First In First Out).
    
    Operations:
    - enqueue: add element to end - O(1)
    - dequeue: remove element from front - O(1)
    - peek: view front element - O(1)
    - is_empty: check if queue is empty - O(1)
    """
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add element to end of queue."""
        pass
    
    def dequeue(self):
        """Remove and return element from front of queue."""
        pass
    
    def peek(self):
        """Return front element without removing it."""
        pass
    
    def is_empty(self):
        """Check if queue is empty."""
        pass

class PriorityQueue:
    """Priority Queue implementation using binary heap.
    
    Elements with higher priority are served before elements with lower priority.
    
    Operations:
    - enqueue: add element with priority - O(log n)
    - dequeue: remove highest priority element - O(log n)
    - peek: view highest priority element - O(1)
    - is_empty: check if priority queue is empty - O(1)
    """
    def __init__(self):
        self.heap = []
    
    def enqueue(self, item, priority):
        """Add element with priority."""
        pass
    
    def dequeue(self):
        """Remove and return highest priority element."""
        pass
    
    def peek(self):
        """Return highest priority element without removing it."""
        pass
    
    def is_empty(self):
        """Check if priority queue is empty."""
        pass

class Deque:
    """Double-ended queue implementation.
    
    Elements can be added or removed from either end.
    
    Operations:
    - add_front: add element to front - O(1)
    - add_rear: add element to rear - O(1)
    - remove_front: remove element from front - O(1)
    - remove_rear: remove element from rear - O(1)
    - is_empty: check if deque is empty - O(1)
    """
    def __init__(self):
        self.items = []
    
    def add_front(self, item):
        """Add element to front of deque."""
        pass
    
    def add_rear(self, item):
        """Add element to rear of deque."""
        pass
    
    def remove_front(self):
        """Remove and return element from front of deque."""
        pass
    
    def remove_rear(self):
        """Remove and return element from rear of deque."""
        pass
    
    def is_empty(self):
        """Check if deque is empty."""
        pass

class HashTable:
    """Hash Table implementation.
    
    Maps keys to values using a hash function.
    Handles collisions using separate chaining.
    
    Operations:
    - put: insert/update key-value pair - O(1) average
    - get: retrieve value by key - O(1) average
    - remove: delete key-value pair - O(1) average
    - contains: check if key exists - O(1) average
    """
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Hash function to convert key to index."""
        pass
    
    def put(self, key, value):
        """Insert or update key-value pair."""
        pass
    
    def get(self, key):
        """Retrieve value by key."""
        pass
    
    def remove(self, key):
        """Delete key-value pair."""
        pass
    
    def contains(self, key):
        """Check if key exists."""
        pass

class TreeNode:
    """Basic node for tree data structures."""
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    """Binary Tree implementation.
    
    Each node has at most two children.
    
    Operations:
    - insert: add element - O(n)
    - search: find element - O(n)
    - traverse: visit all nodes - O(n)
    """
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        """Add element to the tree."""
        pass
    
    def search(self, data):
        """Find element in the tree."""
        pass
    
    def inorder_traversal(self, node=None):
        """Traverse tree in-order (left, root, right)."""
        pass
    
    def preorder_traversal(self, node=None):
        """Traverse tree pre-order (root, left, right)."""
        pass
    
    def postorder_traversal(self, node=None):
        """Traverse tree post-order (left, right, root)."""
        pass

class BinarySearchTree:
    """Binary Search Tree implementation.
    
    A binary tree where nodes are ordered: left < parent < right.
    
    Operations:
    - insert: add element - O(log n) average, O(n) worst
    - search: find element - O(log n) average, O(n) worst
    - delete: remove element - O(log n) average, O(n) worst
    - min/max: find min/max element - O(log n) average, O(n) worst
    """
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        """Add element to the BST."""
        pass
    
    def search(self, data):
        """Find element in the BST."""
        pass
    
    def delete(self, data):
        """Remove element from the BST."""
        pass
    
    def find_min(self):
        """Find minimum element in the BST."""
        pass
    
    def find_max(self):
        """Find maximum element in the BST."""
        pass

class AVLNode(TreeNode):
    """Node for AVL Tree with height attribute."""
    def __init__(self, data=None):
        super().__init__(data)
        self.height = 1

class AVLTree:
    """AVL Tree implementation (self-balancing BST).
    
    A balanced binary search tree where heights of left and right subtrees
    differ by at most 1.
    
    Operations:
    - insert: add element - O(log n)
    - search: find element - O(log n)
    - delete: remove element - O(log n)
    """
    def __init__(self):
        self.root = None
    
    def height(self, node):
        """Get height of node."""
        pass
    
    def balance_factor(self, node):
        """Get balance factor of node."""
        pass
    
    def right_rotate(self, y):
        """Right rotation for rebalancing."""
        pass
    
    def left_rotate(self, x):
        """Left rotation for rebalancing."""
        pass
    
    def insert(self, data):
        """Add element to the AVL tree."""
        pass
    
    def delete(self, data):
        """Remove element from the AVL tree."""
        pass
    
    def search(self, data):
        """Find element in the AVL tree."""
        pass

class MinHeap:
    """Min Heap implementation.
    
    A complete binary tree where parent is less than or equal to children.
    
    Operations:
    - insert: add element - O(log n)
    - extract_min: remove and return minimum element - O(log n)
    - peek: view minimum element - O(1)
    """
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        """Get parent index."""
        pass
    
    def left_child(self, i):
        """Get left child index."""
        pass
    
    def right_child(self, i):
        """Get right child index."""
        pass
    
    def swap(self, i, j):
        """Swap elements at indices i and j."""
        pass
    
    def insert(self, key):
        """Add element to the heap."""
        pass
    
    def heapify_up(self, i):
        """Restore heap property upward."""
        pass
    
    def heapify_down(self, i):
        """Restore heap property downward."""
        pass
    
    def extract_min(self):
        """Remove and return minimum element."""
        pass
    
    def peek(self):
        """Return minimum element without removing it."""
        pass

class TrieNode:
    """Node for Trie data structure."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    """Trie implementation (Prefix Tree).
    
    Tree-like data structure for efficient retrieval of keys in a dataset of strings.
    
    Operations:
    - insert: add string - O(k) where k is string length
    - search: find string - O(k) where k is string length
    - starts_with: check if prefix exists - O(k) where k is prefix length
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Add string to the trie."""
        pass
    
    def search(self, word):
        """Check if string exists in the trie."""
        pass
    
    def starts_with(self, prefix):
        """Check if any string with given prefix exists."""
        pass

class Graph:
    """Graph implementation using adjacency list.
    
    A collection of nodes (vertices) and connections (edges) between them.
    
    Operations:
    - add_vertex: add new vertex - O(1)
    - add_edge: add edge between vertices - O(1)
    - remove_vertex: delete vertex - O(V+E)
    - remove_edge: delete edge - O(E)
    - bfs: breadth-first search - O(V+E)
    - dfs: depth-first search - O(V+E)
    """
    def __init__(self, directed=False):
        self.adjacency_list = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        """Add new vertex to the graph."""
        pass
    
    def add_edge(self, v1, v2, weight=1):
        """Add edge between vertices."""
        pass
    
    def remove_vertex(self, vertex):
        """Delete vertex from the graph."""
        pass
    
    def remove_edge(self, v1, v2):
        """Delete edge between vertices."""
        pass
    
    def dfs(self, start):
        """Depth-first search traversal."""
        pass
    
    def bfs(self, start):
        """Breadth-first search traversal."""
        pass

class DisjointSet:
    """Disjoint Set (Union-Find) implementation.
    
    Keeps track of elements partitioned into non-overlapping subsets.
    
    Operations:
    - make_set: create a new set - O(1)
    - find: find which set an element belongs to - O(α(n))
    - union: merge two sets - O(α(n))
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        """Create a new set with single element x."""
        pass
    
    def find(self, x):
        """Find which set x belongs to."""
        pass
    
    def union(self, x, y):
        """Merge sets containing x and y."""
        pass


if __name__ == "__main__":
    # Example usage of data structures
    print("Testing data structures...")
    
    # Example: using a Binary Search Tree
    bst = BinarySearchTree()
    # Test your implementation by adding code here