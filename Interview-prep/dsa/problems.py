"""
Common DSA Interview Problems

This file contains implementations of frequently asked coding interview questions,
covering various problem-solving patterns:
- Array/String manipulation
- Two pointers technique
- Sliding window
- Binary search
- Dynamic programming
- Recursion
- Backtracking
- Tree/Graph traversals
- Greedy algorithms
- Hash table usage

Each problem includes:
- Problem statement
- Solution approach
- Time and space complexity
- Fully implemented code
"""


#-------------------------------------------------------------------------
# Math and Statistics
#-------------------------------------------------------------------------

def gcd(a, b):
    """
    Problem: Find greatest common divisor using Euclidean algorithm.
    
    Args:
        a, b: Two integers
        
    Returns:
        Greatest common divisor
        
    Time: O(log(min(a, b)))
    Space: O(1)
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Problem: Find least common multiple directly without using GCD.
    
    Args:
        a, b: Two integers
        
    Returns:
        Least common multiple
        
    Time: O(max(a, b)) in worst case
    Space: O(1)
    """
    # Find the larger of the two numbers
    larger = max(a, b)
    smaller = min(a, b)
    
    # Start with the larger number as a potential LCM
    result = larger
    
    # Keep incrementing by larger number until we find a multiple of both
    while result % smaller != 0:
        result += larger
    
    return result
#-------------------------------------------------------------------------
# Array and String Problems
#-------------------------------------------------------------------------

def two_sum(nums, target):
    """
    Problem: Find two numbers in an array that add up to a target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        Tuple of indices of the two numbers
        
    Time: O(n)
    Space: O(n)
    """
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


def max_subarray_sum(nums):
    """
    Problem: Find the contiguous subarray with the largest sum.
    (Kadane's Algorithm)
    
    Args:
        nums: List of integers
        
    Returns:
        Maximum sum of contiguous subarray
        
    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
        
    current_max = global_max = nums[0]
    
    for i in range(1, len(nums)):
        # Either take the current element alone or add it to previous subarray
        current_max = max(nums[i], current_max + nums[i])
        # Update global maximum
        global_max = max(global_max, current_max)
        
    return global_max


def merge_sorted_arrays(nums1, m, nums2, n):
    """
    Problem: Merge two sorted arrays where nums1 has enough space.
    
    Args:
        nums1: First sorted array with extra space
        m: Number of elements in nums1
        nums2: Second sorted array
        n: Number of elements in nums2
        
    Returns:
        nums1 modified in-place
        
    Time: O(m+n)
    Space: O(1)
    """
    # Start from the end to avoid overwriting elements
    p1, p2, p = m - 1, n - 1, m + n - 1
    
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # If there are elements left in nums2, copy them
    nums1[:p2+1] = nums2[:p2+1]
    
    return nums1


def rotate_array(nums, k):
    """
    Problem: Rotate array to the right by k steps.
    
    Args:
        nums: Array to rotate
        k: Number of steps to rotate
        
    Returns:
        nums rotated in-place
        
    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    k = k % n  # Handle k > n case
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    # Reverse entire array
    reverse(0, n - 1)
    # Reverse first k elements
    reverse(0, k - 1)
    # Reverse remaining elements
    reverse(k, n - 1)
    
    return nums


def is_palindrome(s):
    """
    Problem: Check if a string is a palindrome considering only alphanumeric chars.
    
    Args:
        s: Input string
        
    Returns:
        True if palindrome, False otherwise
        
    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


#-------------------------------------------------------------------------
# Two Pointers Technique
#-------------------------------------------------------------------------

def remove_duplicates(nums):
    """
    Problem: Remove duplicates from sorted array in-place.
    
    Args:
        nums: Sorted array
        
    Returns:
        Length of new array and modifies nums in-place
        
    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
        
    # Pointer to position where unique element should be placed
    unique_pos = 1
    
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[unique_pos] = nums[i]
            unique_pos += 1
    
    return unique_pos


def three_sum(nums):
    """
    Problem: Find all unique triplets that sum to zero.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of triplets that sum to zero
        
    Time: O(n²)
    Space: O(n) for the output
    """
    result = []
    nums.sort()
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        left, right = i + 1, n - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                # Found a triplet
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                    
                left += 1
                right -= 1
    
    return result


def container_with_most_water(height):
    """
    Problem: Find two lines that together with the x-axis form a container
    that holds the most water.
    
    Args:
        height: List of heights
        
    Returns:
        Maximum amount of water
        
    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        # Calculate area
        width = right - left
        area = width * min(height[left], height[right])
        max_area = max(max_area, area)
        
        # Move the pointer with lower height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area


#-------------------------------------------------------------------------
# Sliding Window
#-------------------------------------------------------------------------

def max_sliding_window(nums, k):
    """
    Problem: Find maximum in each sliding window of size k.
    
    Args:
        nums: Array of integers
        k: Window size
        
    Returns:
        Array of maximum elements
        
    Time: O(n)
    Space: O(k)
    """
    from collections import deque
    
    if not nums or k == 0:
        return []
    
    result = []
    window = deque()  # Store indices
    
    for i, num in enumerate(nums):
        # Remove elements outside current window
        while window and window[0] < i - k + 1:
            window.popleft()
        
        # Remove smaller elements
        while window and nums[window[-1]] < num:
            window.pop()
        
        window.append(i)
        
        # Add to result when first window is complete
        if i >= k - 1:
            result.append(nums[window[0]])
    
    return result


def longest_substring_without_repeating(s):
    """
    Problem: Find length of longest substring without repeating characters.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest substring
        
    Time: O(n)
    Space: O(min(m,n)) where m is the size of the character set
    """
    char_index = {}  # char -> last seen index
    start = 0
    max_length = 0
    
    for end, char in enumerate(s):
        # If char is in current window, update start pointer
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        else:
            max_length = max(max_length, end - start + 1)
        
        # Update last seen index
        char_index[char] = end
    
    return max_length


def min_window_substring(s, t):
    """
    Problem: Find the minimum window in s that contains all characters in t.
    
    Args:
        s: String to search in
        t: String to find
        
    Returns:
        Minimum window substring
        
    Time: O(n)
    Space: O(k) where k is the character set size
    """
    from collections import Counter
    
    if not s or not t:
        return ""
    
    # Initialize character frequency counters
    target_counter = Counter(t)
    window_counter = Counter()
    
    # Initialize variables
    required = len(target_counter)
    formed = 0
    start = 0
    min_len = float('inf')
    result_start = 0
    
    for end, char in enumerate(s):
        # Add character to window
        window_counter[char] += 1
        
        # Check if we've matched the frequency of a character
        if char in target_counter and window_counter[char] == target_counter[char]:
            formed += 1
        
        # Try to minimize window
        while formed == required:
            # Update result if current window is smaller
            if end - start + 1 < min_len:
                min_len = end - start + 1
                result_start = start
            
            # Remove character from window
            left_char = s[start]
            window_counter[left_char] -= 1
            
            # Check if we've broken the required frequency
            if left_char in target_counter and window_counter[left_char] < target_counter[left_char]:
                formed -= 1
            
            start += 1
    
    return "" if min_len == float('inf') else s[result_start:result_start + min_len]


#-------------------------------------------------------------------------
# Binary Search
#-------------------------------------------------------------------------

def binary_search(nums, target):
    """
    Problem: Find target in sorted array.
    
    Args:
        nums: Sorted array
        target: Element to find
        
    Returns:
        Index of target or -1 if not found
        
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def search_in_rotated_sorted_array(nums, target):
    """
    Problem: Find target in rotated sorted array.
    
    Args:
        nums: Rotated sorted array
        target: Element to find
        
    Returns:
        Index of target or -1 if not found
        
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            # Check if target is in left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            # Check if target is in right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def find_peak_element(nums):
    """
    Problem: Find a peak element (greater than neighbors).
    
    Args:
        nums: Array of integers
        
    Returns:
        Index of a peak element
        
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # If mid is a downward slope, search left
        if nums[mid] > nums[mid + 1]:
            right = mid
        # If mid is an upward slope, search right
        else:
            left = mid + 1
    
    return left  # left == right at this point


#-------------------------------------------------------------------------
# Dynamic Programming
#-------------------------------------------------------------------------

def fibonacci(n):
    """
    Problem: Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time: O(n)
    Space: O(1)
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def climb_stairs(n):
    """
    Problem: Count ways to climb n stairs taking 1 or 2 steps at a time.
    
    Args:
        n: Number of stairs
        
    Returns:
        Number of distinct ways
        
    Time: O(n)
    Space: O(1)
    """
    if n <= 2:
        return n
    
    one_step_before = 2
    two_steps_before = 1
    
    for _ in range(3, n + 1):
        curr = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = curr
    
    return one_step_before


def longest_increasing_subsequence(nums):
    """
    Problem: Find length of longest increasing subsequence.
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of longest increasing subsequence
        
    Time: O(n²) [O(n log n) with binary search optimization]
    Space: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS ending at nums[i]
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def coin_change(coins, amount):
    """
    Problem: Find fewest coins needed to make up amount.
    
    Args:
        coins: Array of coin denominations
        amount: Target amount
        
    Returns:
        Fewest number of coins or -1 if impossible
        
    Time: O(amount * len(coins))
    Space: O(amount)
    """
    # Initialize dp array with amount + 1 (impossible value)
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] <= amount else -1


def knapsack_0_1(values, weights, capacity):
    """
    Problem: 0/1 Knapsack - maximize value without exceeding capacity.
    
    Args:
        values: List of item values
        weights: List of item weights
        capacity: Knapsack capacity
        
    Returns:
        Maximum value
        
    Time: O(n * capacity)
    Space: O(capacity)
    """
    n = len(values)
    # Optimize space using 1D dp array
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]


#-------------------------------------------------------------------------
# Recursion and Backtracking
#-------------------------------------------------------------------------

def generate_parentheses(n):
    """
    Problem: Generate all valid combinations of n pairs of parentheses.
    
    Args:
        n: Number of pairs
        
    Returns:
        List of valid combinations
        
    Time: O(4^n / sqrt(n)) - Catalan number
    Space: O(n) for recursion stack
    """
    result = []
    
    def backtrack(s, open_count, close_count):
        # Base case
        if len(s) == 2 * n:
            result.append(s)
            return
        
        # Add open parenthesis if we haven't used all
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        
        # Add closing parenthesis if valid
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result


def permutations(nums):
    """
    Problem: Generate all permutations of an array.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of all permutations
        
    Time: O(n * n!)
    Space: O(n * n!)
    """
    result = []
    
    def backtrack(start):
        # Base case
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap to create a new permutation
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse on the next position
            backtrack(start + 1)
            
            # Backtrack (undo the swap)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


def subsets(nums):
    """
    Problem: Generate all possible subsets of a set.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of all subsets
        
    Time: O(n * 2^n)
    Space: O(n * 2^n)
    """
    result = []
    
    def backtrack(start, current):
        # Add current subset to result
        result.append(current[:])
        
        for i in range(start, len(nums)):
            # Include nums[i]
            current.append(nums[i])
            
            # Generate all subsets with nums[i]
            backtrack(i + 1, current)
            
            # Backtrack (exclude nums[i])
            current.pop()
    
    backtrack(0, [])
    return result


def sudoku_solver(board):
    """
    Problem: Solve a 9x9 Sudoku puzzle.
    
    Args:
        board: 9x9 matrix of characters
        
    Returns:
        True if solvable, otherwise False
        
    Time: O(9^(n*n)) in worst case
    Space: O(n*n) for recursion stack
    """
    # Find empty cell
    def find_empty():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    return i, j
        return None
    
    # Check if number is valid in given position
    def is_valid(row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    # Backtracking algorithm
    def solve():
        empty = find_empty()
        if not empty:
            return True  # Puzzle solved
        
        row, col = empty
        
        for num in '123456789':
            if is_valid(row, col, num):
                # Try this number
                board[row][col] = num
                
                # Recursively try to solve rest of the puzzle
                if solve():
                    return True
                
                # If failed, backtrack
                board[row][col] = '.'
        
        return False  # Trigger backtracking
    
    return solve()


#-------------------------------------------------------------------------
# Tree and Graph Problems
#-------------------------------------------------------------------------

class TreeNode:
    """Node for binary tree problems."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth_binary_tree(root):
    """
    Problem: Find maximum depth of binary tree.
    
    Args:
        root: Root of binary tree
        
    Returns:
        Maximum depth
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    if not root:
        return 0
    
    # Depth = 1 + max depth of subtrees
    return 1 + max(max_depth_binary_tree(root.left), max_depth_binary_tree(root.right))


def is_same_tree(p, q):
    """
    Problem: Check if two binary trees are identical.
    
    Args:
        p, q: Roots of binary trees
        
    Returns:
        True if identical, False otherwise
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    # Both empty
    if not p and not q:
        return True
    
    # One empty, one not
    if not p or not q:
        return False
    
    # Check current nodes and recursively check subtrees
    return (p.val == q.val and 
            is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))


def level_order_traversal(root):
    """
    Problem: Level order traversal of binary tree.
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of levels
        
    Time: O(n)
    Space: O(n)
    """
    from collections import deque
    
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result


def is_valid_bst(root):
    """
    Problem: Check if binary tree is a valid BST.
    
    Args:
        root: Root of binary tree
        
    Returns:
        True if valid BST, False otherwise
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    def validate(node, lower=float('-inf'), upper=float('inf')):
        if not node:
            return True
        
        # Check current node
        if node.val <= lower or node.val >= upper:
            return False
        
        # Check left subtree (all values must be < node.val)
        # Check right subtree (all values must be > node.val)
        return (validate(node.left, lower, node.val) and
                validate(node.right, node.val, upper))
    
    return validate(root)


def least_common_ancestor(root, p, q):
    """
    Problem: Find least common ancestor of two nodes in binary tree.
    
    Args:
        root: Root of binary tree
        p, q: Nodes to find LCA for
        
    Returns:
        LCA node
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    if not root or root == p or root == q:
        return root
    
    # Search in left and right subtrees
    left = least_common_ancestor(root.left, p, q)
    right = least_common_ancestor(root.right, p, q)
    
    # If both nodes found in different subtrees, current node is LCA
    if left and right:
        return root
    
    # Otherwise, return non-null result
    return left if left else right


class GraphNode:
    """Node for graph problems."""
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    """
    Problem: Clone an undirected graph.
    
    Args:
        node: Source graph node
        
    Returns:
        Copy of the graph
        
    Time: O(V+E)
    Space: O(V)
    """
    if not node:
        return None
    
    # Dictionary to map original nodes to their clones
    cloned = {}
    
    def dfs(original):
        if original in cloned:
            return cloned[original]
        
        # Create a clone of the current node
        copy = GraphNode(original.val)
        cloned[original] = copy
        
        # Clone all neighbors
        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))
        
        return copy
    
    return dfs(node)


def course_schedule(num_courses, prerequisites):
    """
    Problem: Check if it's possible to finish all courses.
    
    Args:
        num_courses: Number of courses
        prerequisites: List of prerequisite pairs
        
    Returns:
        True if possible, False if cycle exists
        
    Time: O(V+E)
    Space: O(V+E)
    """
    # Build adjacency list
    graph = [[] for _ in range(num_courses)]
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # 0 = unvisited, 1 = in progress, 2 = completed
    visited = [0] * num_courses
    
    def has_cycle(course):
        # Visiting a node in progress (cycle)
        if visited[course] == 1:
            return True
        
        # Already completed
        if visited[course] == 2:
            return False
        
        # Mark as in progress
        visited[course] = 1
        
        # Check all prerequisites
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        
        # Mark as completed
        visited[course] = 2
        return False
    
    for course in range(num_courses):
        if visited[course] == 0:
            if has_cycle(course):
                return False
    
    return True


def dijkstra(graph, start):
    """
    Problem: Find shortest paths from start vertex to all others.
    
    Args:
        graph: Dictionary of {vertex: {neighbor: weight}}
        start: Starting vertex
        
    Returns:
        Dictionary of shortest distances
        
    Time: O((V+E)logV) with binary heap
    Space: O(V)
    """
    import heapq
    
    # Initialize distances
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    
    # Priority queue for vertices to visit
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # Skip if we've found a better path
        if current_distance > distances[current_vertex]:
            continue
        
        # Check all neighbors
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            
            # If we found a better path, update and add to queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

"""
Common DSA Interview Problems

This file contains implementations of frequently asked coding interview questions,
covering various problem-solving patterns:
- Array/String manipulation
- Two pointers technique
- Sliding window
- Binary search
- Dynamic programming
- Recursion
- Backtracking
- Tree/Graph traversals
- Greedy algorithms
- Hash table usage

Each problem includes:
- Problem statement
- Solution approach
- Time and space complexity
- Fully implemented code
"""

#-------------------------------------------------------------------------
# Array and String Problems
#-------------------------------------------------------------------------

def two_sum(nums, target):
    """
    Problem: Find two numbers in an array that add up to a target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        Tuple of indices of the two numbers
        
    Time: O(n)
    Space: O(n)
    """
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


def max_subarray_sum(nums):
    """
    Problem: Find the contiguous subarray with the largest sum.
    (Kadane's Algorithm)
    
    Args:
        nums: List of integers
        
    Returns:
        Maximum sum of contiguous subarray
        
    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
        
    current_max = global_max = nums[0]
    
    for i in range(1, len(nums)):
        # Either take the current element alone or add it to previous subarray
        current_max = max(nums[i], current_max + nums[i])
        # Update global maximum
        global_max = max(global_max, current_max)
        
    return global_max


def merge_sorted_arrays(nums1, m, nums2, n):
    """
    Problem: Merge two sorted arrays where nums1 has enough space.
    
    Args:
        nums1: First sorted array with extra space
        m: Number of elements in nums1
        nums2: Second sorted array
        n: Number of elements in nums2
        
    Returns:
        nums1 modified in-place
        
    Time: O(m+n)
    Space: O(1)
    """
    # Start from the end to avoid overwriting elements
    p1, p2, p = m - 1, n - 1, m + n - 1
    
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # If there are elements left in nums2, copy them
    nums1[:p2+1] = nums2[:p2+1]
    
    return nums1


def rotate_array(nums, k):
    """
    Problem: Rotate array to the right by k steps.
    
    Args:
        nums: Array to rotate
        k: Number of steps to rotate
        
    Returns:
        nums rotated in-place
        
    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    k = k % n  # Handle k > n case
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    # Reverse entire array
    reverse(0, n - 1)
    # Reverse first k elements
    reverse(0, k - 1)
    # Reverse remaining elements
    reverse(k, n - 1)
    
    return nums


def is_palindrome(s):
    """
    Problem: Check if a string is a palindrome considering only alphanumeric chars.
    
    Args:
        s: Input string
        
    Returns:
        True if palindrome, False otherwise
        
    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


#-------------------------------------------------------------------------
# Two Pointers Technique
#-------------------------------------------------------------------------

def remove_duplicates(nums):
    """
    Problem: Remove duplicates from sorted array in-place.
    
    Args:
        nums: Sorted array
        
    Returns:
        Length of new array and modifies nums in-place
        
    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
        
    # Pointer to position where unique element should be placed
    unique_pos = 1
    
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[unique_pos] = nums[i]
            unique_pos += 1
    
    return unique_pos


def three_sum(nums):
    """
    Problem: Find all unique triplets that sum to zero.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of triplets that sum to zero
        
    Time: O(n²)
    Space: O(n) for the output
    """
    result = []
    nums.sort()
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        left, right = i + 1, n - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                # Found a triplet
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                    
                left += 1
                right -= 1
    
    return result


def container_with_most_water(height):
    """
    Problem: Find two lines that together with the x-axis form a container
    that holds the most water.
    
    Args:
        height: List of heights
        
    Returns:
        Maximum amount of water
        
    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        # Calculate area
        width = right - left
        area = width * min(height[left], height[right])
        max_area = max(max_area, area)
        
        # Move the pointer with lower height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area


#-------------------------------------------------------------------------
# Sliding Window
#-------------------------------------------------------------------------

def max_sliding_window(nums, k):
    """
    Problem: Find maximum in each sliding window of size k.
    
    Args:
        nums: Array of integers
        k: Window size
        
    Returns:
        Array of maximum elements
        
    Time: O(n)
    Space: O(k)
    """
    from collections import deque
    
    if not nums or k == 0:
        return []
    
    result = []
    window = deque()  # Store indices
    
    for i, num in enumerate(nums):
        # Remove elements outside current window
        while window and window[0] < i - k + 1:
            window.popleft()
        
        # Remove smaller elements
        while window and nums[window[-1]] < num:
            window.pop()
        
        window.append(i)
        
        # Add to result when first window is complete
        if i >= k - 1:
            result.append(nums[window[0]])
    
    return result


def longest_substring_without_repeating(s):
    """
    Problem: Find length of longest substring without repeating characters.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest substring
        
    Time: O(n)
    Space: O(min(m,n)) where m is the size of the character set
    """
    char_index = {}  # char -> last seen index
    start = 0
    max_length = 0
    
    for end, char in enumerate(s):
        # If char is in current window, update start pointer
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        else:
            max_length = max(max_length, end - start + 1)
        
        # Update last seen index
        char_index[char] = end
    
    return max_length


#-------------------------------------------------------------------------
# Binary Search
#-------------------------------------------------------------------------

def binary_search(nums, target):
    """
    Problem: Find target in sorted array.
    
    Args:
        nums: Sorted array
        target: Element to find
        
    Returns:
        Index of target or -1 if not found
        
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def search_in_rotated_sorted_array(nums, target):
    """
    Problem: Find target in rotated sorted array.
    
    Args:
        nums: Rotated sorted array
        target: Element to find
        
    Returns:
        Index of target or -1 if not found
        
    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            # Check if target is in left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            # Check if target is in right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


#-------------------------------------------------------------------------
# Dynamic Programming
#-------------------------------------------------------------------------

def fibonacci(n):
    """
    Problem: Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time: O(n)
    Space: O(1)
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def climb_stairs(n):
    """
    Problem: Count ways to climb n stairs taking 1 or 2 steps at a time.
    
    Args:
        n: Number of stairs
        
    Returns:
        Number of distinct ways
        
    Time: O(n)
    Space: O(1)
    """
    if n <= 2:
        return n
    
    one_step_before = 2
    two_steps_before = 1
    
    for _ in range(3, n + 1):
        curr = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = curr
    
    return one_step_before


def longest_increasing_subsequence(nums):
    """
    Problem: Find length of longest increasing subsequence.
    
    Args:
        nums: Array of integers
        
    Returns:
        Length of longest increasing subsequence
        
    Time: O(n²)
    Space: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS ending at nums[i]
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def coin_change(coins, amount):
    """
    Problem: Find fewest coins needed to make up amount.
    
    Args:
        coins: Array of coin denominations
        amount: Target amount
        
    Returns:
        Fewest number of coins or -1 if impossible
        
    Time: O(amount * len(coins))
    Space: O(amount)
    """
    # Initialize dp array with amount + 1 (impossible value)
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] <= amount else -1


#-------------------------------------------------------------------------
# Recursion and Backtracking
#-------------------------------------------------------------------------

def generate_parentheses(n):
    """
    Problem: Generate all valid combinations of n pairs of parentheses.
    
    Args:
        n: Number of pairs
        
    Returns:
        List of valid combinations
        
    Time: O(4^n / sqrt(n)) - Catalan number
    Space: O(n) for recursion stack
    """
    result = []
    
    def backtrack(s, open_count, close_count):
        # Base case
        if len(s) == 2 * n:
            result.append(s)
            return
        
        # Add open parenthesis if we haven't used all
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        
        # Add closing parenthesis if valid
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result


def permutations(nums):
    """
    Problem: Generate all permutations of an array.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of all permutations
        
    Time: O(n * n!)
    Space: O(n * n!)
    """
    result = []
    
    def backtrack(start):
        # Base case
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap to create a new permutation
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse on the next position
            backtrack(start + 1)
            
            # Backtrack (undo the swap)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


def subsets(nums):
    """
    Problem: Generate all possible subsets of a set.
    
    Args:
        nums: Array of integers
        
    Returns:
        List of all subsets
        
    Time: O(n * 2^n)
    Space: O(n * 2^n)
    """
    result = []
    
    def backtrack(start, current):
        # Add current subset to result
        result.append(current[:])
        
        for i in range(start, len(nums)):
            # Include nums[i]
            current.append(nums[i])
            
            # Generate all subsets with nums[i]
            backtrack(i + 1, current)
            
            # Backtrack (exclude nums[i])
            current.pop()
    
    backtrack(0, [])
    return result


#-------------------------------------------------------------------------
# Tree Traversal and Operations
#-------------------------------------------------------------------------

class TreeNode:
    """Node for binary tree problems."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    """
    Problem: Perform inorder traversal of binary tree (left-root-right).
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of values in inorder
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)
    
    traverse(root)
    return result


def inorder_traversal_iterative(root):
    """
    Problem: Perform inorder traversal iteratively.
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of values in inorder
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go all the way left
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Go right
        current = current.right
    
    return result


def preorder_traversal(root):
    """
    Problem: Perform preorder traversal of binary tree (root-left-right).
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of values in preorder
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)
    
    traverse(root)
    return result


def postorder_traversal(root):
    """
    Problem: Perform postorder traversal of binary tree (left-right-root).
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of values in postorder
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    result = []
    
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node.val)
    
    traverse(root)
    return result


def level_order_traversal(root):
    """
    Problem: Level order traversal of binary tree.
    
    Args:
        root: Root of binary tree
        
    Returns:
        List of levels
        
    Time: O(n)
    Space: O(n)
    """
    from collections import deque
    
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result


def max_depth_binary_tree(root):
    """
    Problem: Find maximum depth of binary tree.
    
    Args:
        root: Root of binary tree
        
    Returns:
        Maximum depth
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    if not root:
        return 0
    
    # Depth = 1 + max depth of subtrees
    return 1 + max(max_depth_binary_tree(root.left), max_depth_binary_tree(root.right))


def is_balanced_tree(root):
    """
    Problem: Check if binary tree is height-balanced.
    
    Args:
        root: Root of binary tree
        
    Returns:
        True if balanced, False otherwise
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    def height(node):
        if not node:
            return 0
        
        left_height = height(node.left)
        if left_height == -1:
            return -1
        
        right_height = height(node.right)
        if right_height == -1:
            return -1
        
        # Check if current node is balanced
        if abs(left_height - right_height) > 1:
            return -1
        
        return 1 + max(left_height, right_height)
    
    return height(root) != -1


def is_same_tree(p, q):
    """
    Problem: Check if two binary trees are identical.
    
    Args:
        p, q: Roots of binary trees
        
    Returns:
        True if identical, False otherwise
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    # Both empty
    if not p and not q:
        return True
    
    # One empty, one not
    if not p or not q:
        return False
    
    # Check current nodes and recursively check subtrees
    return (p.val == q.val and 
            is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))


def is_valid_bst(root):
    """
    Problem: Check if binary tree is a valid BST.
    
    Args:
        root: Root of binary tree
        
    Returns:
        True if valid BST, False otherwise
        
    Time: O(n)
    Space: O(h) where h is the height of the tree
    """
    def validate(node, lower=float('-inf'), upper=float('inf')):
        if not node:
            return True
        
        # Check current node
        if node.val <= lower or node.val >= upper:
            return False
        
        # Check left subtree (all values must be < node.val)
        # Check right subtree (all values must be > node.val)
        return (validate(node.left, lower, node.val) and
                validate(node.right, node.val, upper))
    
    return validate(root)


#-------------------------------------------------------------------------
# Graph Algorithms
#-------------------------------------------------------------------------

def bfs_shortest_path(graph, start, end):
    """
    Problem: Find shortest path between two nodes in unweighted graph.
    
    Args:
        graph: Dictionary of {node: [neighbors]}
        start: Starting node
        end: Target node
        
    Returns:
        Shortest path as a list, or None if no path exists
        
    Time: O(V+E)
    Space: O(V)
    """
    from collections import deque
    
    if start == end:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if neighbor == end:
                    return path + [neighbor]
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found


def dfs_traversal(graph, start):
    """
    Problem: Perform DFS traversal of graph.
    
    Args:
        graph: Dictionary of {node: [neighbors]}
        start: Starting node
        
    Returns:
        List of nodes in DFS order
        
    Time: O(V+E)
    Space: O(V)
    """
    visited = set()
    result = []
    
    def dfs(node):
        visited.add(node)
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)
    return result


def has_cycle_undirected(graph):
    """
    Problem: Detect cycle in undirected graph.
    
    Args:
        graph: Dictionary of {node: [neighbors]}
        
    Returns:
        True if cycle exists, False otherwise
        
    Time: O(V+E)
    Space: O(V)
    """
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            # Skip the edge to parent
            if neighbor == parent:
                continue
            
            # Found a back edge (cycle)
            if neighbor in visited:
                return True
            
            # Recursively check neighbors
            if dfs(neighbor, node):
                return True
        
        return False
    
    # Check all components
    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    
    return False


def topological_sort(graph):
    """
    Problem: Perform topological sort on directed acyclic graph.
    
    Args:
        graph: Dictionary of {node: [neighbors]}
        
    Returns:
        Topologically sorted list of nodes
        
    Time: O(V+E)
    Space: O(V)
    """
    # 0 = not visited, 1 = in progress, 2 = completed
    visited = {node: 0 for node in graph}
    result = []
    
    def dfs(node):
        # Detect cycle
        if visited[node] == 1:
            return False
        
        if visited[node] == 2:
            return True
        
        visited[node] = 1
        
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        visited[node] = 2
        result.append(node)
        return True
    
    for node in graph:
        if visited[node] == 0:
            if not dfs(node):
                return []  # Cycle detected
    
    return result[::-1]  # Reverse for correct order


#-------------------------------------------------------------------------
# Hash Table Problems
#-------------------------------------------------------------------------

def two_sum_map(nums, target):
    """
    Problem: Find indices of two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        Indices of two numbers
        
    Time: O(n)
    Space: O(n)
    """
    num_map = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return None


def group_anagrams(strs):
    """
    Problem: Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
        
    Returns:
        List of anagram groups
        
    Time: O(n * k) where n is length of array and k is max length of string
    Space: O(n * k)
    """
    from collections import defaultdict
    
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Sort characters as key
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())


def longest_consecutive_sequence(nums):
    """
    Problem: Find length of longest consecutive sequence in unsorted array.
    
    Args:
        nums: List of integers
        
    Returns:
        Length of longest consecutive sequence
        
    Time: O(n)
    Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start a sequence from its smallest element
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive elements
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length


#-------------------------------------------------------------------------
# Priority Queue (Heap) Problems
#-------------------------------------------------------------------------

def kth_largest(nums, k):
    """
    Problem: Find kth largest element in an array.
    
    Args:
        nums: List of integers
        k: Position (1-based)
        
    Returns:
        kth largest element
        
    Time: O(n log k)
    Space: O(k)
    """
    import heapq
    
    # Min heap of size k
    heap = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return heap[0]


def merge_k_sorted_lists(lists):
    """
    Problem: Merge k sorted linked lists.
    
    Args:
        lists: List of k sorted linked lists
        
    Returns:
        Merged sorted linked list
        
    Time: O(n log k) where n is total number of nodes
    Space: O(k)
    """
    import heapq
    
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    # Custom wrapper to make ListNode comparable
    class NodeWrapper:
        def __init__(self, node):
            self.node = node
        
        def __lt__(self, other):
            return self.node.val < other.node.val
    
    # Remove empty lists
    lists = [lst for lst in lists if lst]
    if not lists:
        return None
    
    # Initialize heap with first node from each list
    heap = [NodeWrapper(lst) for lst in lists]
    heapq.heapify(heap)
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        # Get smallest node
        node_wrapper = heapq.heappop(heap)
        node = node_wrapper.node
        
        # Add to result
        current.next = node
        current = current.next
        
        # If there are more nodes in this list, add to heap
        if node.next:
            heapq.heappush(heap, NodeWrapper(node.next))
    
    return dummy.next


#-------------------------------------------------------------------------
# Bit Manipulation
#-------------------------------------------------------------------------
def count_bits(n):
    """
    Problem: Count number of 1 bits in an integer.
    
    Args:
        n: Integer
        
    Returns:
        Number of 1 bits
        
    Time: O(log n) - number of bits
    Space: O(1)
    """
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def is_power_of_two(n):
    """
    Problem: Check if a number is a power of two.
    
    Args:
        n: Integer
        
    Returns:
        True if power of two, False otherwise
        
    Time: O(1)
    Space: O(1)
    """
    # A power of two has exactly one bit set
    return n > 0 and (n & (n - 1)) == 0


def single_number(nums):
    """
    Problem: Find number that appears only once in array where others appear twice.
    
    Args:
        nums: List of integers
        
    Returns:
        Number that appears only once
        
    Time: O(n)
    Space: O(1)
    """
    result = 0
    for num in nums:
        # XOR of a number with itself is 0
        # XOR of a number with 0 is the number itself
        result ^= num
    return result


def missing_number(nums):
    """
    Problem: Find missing number in array containing 0 to n except one number.
    
    Args:
        nums: List of integers
        
    Returns:
        Missing number
        
    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


#-------------------------------------------------------------------------
# Greedy Algorithms
#-------------------------------------------------------------------------

def jump_game(nums):
    """
    Problem: Determine if you can reach the last index of array.
    
    Args:
        nums: List of integers where nums[i] represents max jump length
        
    Returns:
        True if you can reach the last index, False otherwise
        
    Time: O(n)
    Space: O(1)
    """
    max_reach = 0
    
    for i in range(len(nums)):
        # If we can't reach current position
        if i > max_reach:
            return False
        
        # Update max reach
        max_reach = max(max_reach, i + nums[i])
        
        # If we can already reach the end
        if max_reach >= len(nums) - 1:
            return True
    
    return True


def min_meeting_rooms(intervals):
    """
    Problem: Find minimum number of meeting rooms required.
    
    Args:
        intervals: List of [start, end] intervals
        
    Returns:
        Minimum number of meeting rooms
        
    Time: O(n log n)
    Space: O(n)
    """
    import heapq
    
    if not intervals:
        return 0
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times of active meetings
    rooms = []
    
    for start, end in intervals:
        # If there's a room that finishes before this meeting starts
        if rooms and rooms[0] <= start:
            # Reuse that room
            heapq.heappop(rooms)
        
        # Add current meeting's end time
        heapq.heappush(rooms, end)
    
    # Number of rooms = size of heap
    return len(rooms)


def maximum_subarray(nums):
    """
    Problem: Find contiguous subarray with largest sum (Kadane's algorithm).
    
    Args:
        nums: List of integers
        
    Returns:
        Maximum sum
        
    Time: O(n)
    Space: O(1)
    """
    current_max = global_max = nums[0]
    
    for i in range(1, len(nums)):
        # Either take current element alone or add to previous subarray
        current_max = max(nums[i], current_max + nums[i])
        # Update global maximum
        global_max = max(global_max, current_max)
    
    return global_max


#-------------------------------------------------------------------------
# Trie Implementation
#-------------------------------------------------------------------------

class TrieNode:
    """Node for Trie data structure."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    """
    Problem: Implement a Trie (Prefix Tree).
    
    Operations:
    - insert: O(m) where m is word length
    - search: O(m) where m is word length
    - starts_with: O(m) where m is prefix length
    
    Space: O(n*m) where n is number of words, m is average word length
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the trie."""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
    
    def search(self, word):
        """Return True if the word is in the trie."""
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        """Return True if there is any word in the trie that starts with the given prefix."""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True


#-------------------------------------------------------------------------
# System Design Patterns (Object-Oriented Design)
#-------------------------------------------------------------------------

class LRUCache:
    """
    Problem: Design and implement an LRU (Least Recently Used) cache.
    
    Operations:
    - get: O(1)
    - put: O(1)
    
    Space: O(capacity)
    """
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        
        # Initialize doubly linked list with dummy head and tail
        self.head = self.Node()  # most recently used
        self.tail = self.Node()  # least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head (most recently used)."""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove node from linked list."""
        prev = node.prev
        next_node = node.next
        
        prev.next = next_node
        next_node.prev = prev
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)."""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove and return the least recently used node."""
        res = self.tail.prev
        self._remove_node(res)
        return res
    
    def get(self, key):
        """Get value by key and update usage."""
        if key not in self.cache:
            return -1
        
        # Update usage
        node = self.cache[key]
        self._move_to_head(node)
        
        return node.value
    
    def put(self, key, value):
        """Add or update key-value pair."""
        # Update existing
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
            return
        
        # Add new
        node = self.Node(key, value)
        self.cache[key] = node
        self._add_node(node)
        
        # Check capacity
        if len(self.cache) > self.capacity:
            # Remove least recently used
            tail = self._pop_tail()
            del self.cache[tail.key]


class MinStack:
    """
    Problem: Design a stack that supports push, pop, top, and getMin in O(1) time.
    
    Operations:
    - push: O(1)
    - pop: O(1)
    - top: O(1)
    - getMin: O(1)
    
    Space: O(n)
    """
    def __init__(self):
        self.stack = []  # (value, min_at_this_position)
    
    def push(self, val):
        """Push element onto stack and update min."""
        current_min = val
        if self.stack:
            current_min = min(current_min, self.stack[-1][1])
        self.stack.append((val, current_min))
    
    def pop(self):
        """Remove the top element."""
        if self.stack:
            self.stack.pop()
    
    def top(self):
        """Get the top element."""
        if self.stack:
            return self.stack[-1][0]
        return None
    
    def getMin(self):
        """Get the minimum element."""
        if self.stack:
            return self.stack[-1][1]
        return None


#-------------------------------------------------------------------------
# Additional Common Problems
#-------------------------------------------------------------------------

def is_valid_sudoku(board):
    """
    Problem: Check if a 9x9 Sudoku board is valid.
    
    Args:
        board: 9x9 matrix
        
    Returns:
        True if valid, False otherwise
        
    Time: O(1) - fixed size
    Space: O(1) - fixed size
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num == '.':
                continue
                
            # Check row
            if num in rows[i]:
                return False
            rows[i].add(num)
            
            # Check column
            if num in cols[j]:
                return False
            cols[j].add(num)
            
            # Check 3x3 box
            box_idx = (i // 3) * 3 + j // 3
            if num in boxes[box_idx]:
                return False
            boxes[box_idx].add(num)
    
    return True


def spiral_matrix(matrix):
    """
    Problem: Return all elements of the matrix in spiral order.
    
    Args:
        matrix: m x n matrix
        
    Returns:
        List of elements in spiral order
        
    Time: O(m*n)
    Space: O(m*n) for result
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        # Traverse left
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        # Traverse up
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return result


def rotate_matrix(matrix):
    """
    Problem: Rotate image by 90 degrees clockwise.
    
    Args:
        matrix: n x n matrix
        
    Returns:
        Matrix rotated in-place
        
    Time: O(n²)
    Space: O(1)
    """
    n = len(matrix)
    
    # Transpose matrix (swap rows with columns)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        left, right = 0, n - 1
        while left < right:
            matrix[i][left], matrix[i][right] = matrix[i][right], matrix[i][left]
            left += 1
            right -= 1
    
    return matrix


def product_except_self(nums):
    """
    Problem: Calculate product of array except self without division.
    
    Args:
        nums: List of integers
        
    Returns:
        List of products
        
    Time: O(n)
    Space: O(1) excluding result array
    """
    n = len(nums)
    result = [1] * n
    
    # Calculate products of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Multiply by products of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result


def next_permutation(nums):
    """
    Problem: Find next lexicographical permutation.
    
    Args:
        nums: List of integers
        
    Returns:
        Next permutation in-place
        
    Time: O(n)
    Space: O(1)
    """
    # Find first decreasing element from the right
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Find element just larger than nums[i]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        
        # Swap them
        nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse the portion after index i
    left = i + 1
    right = len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    
    return nums


def rain_water_trapped(height):
    """
    Problem: Calculate how much water can be trapped after raining.
    
    Args:
        height: List of heights
        
    Returns:
        Amount of water trapped
        
    Time: O(n)
    Space: O(1)
    """
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = height[left]
    right_max = height[right]
    result = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            result += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            result += right_max - height[right]
    
    return result


if __name__ == "__main__":
    # Example usage to test the implementations
    
    # Example 1: Two Sum
    print("Two Sum Example:")
    nums = [2, 7, 11, 15]
    target = 9
    print(two_sum(nums, target))  # Expected: (0, 1)
    
    # Example 2: Binary Search
    print("\nBinary Search Example:")
    nums = [1, 2, 3, 4, 5, 6, 7]
    target = 5
    print(binary_search(nums, target))  # Expected: 4
    
    # Example 3: Maximum Subarray Sum
    print("\nMaximum Subarray Sum Example:")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(maximum_subarray(nums))  # Expected: 6
    
    # Example 4: Linked List (using Python lists for simplicity)
    print("\nLinked List Example:")
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
    
    # Create a linked list: 1->2->3->4->5
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)
    
    # Example 5: Bit Manipulation
    print("\nBit Manipulation Example:")
    n = 5  # Binary: 101
    print(f"Number of 1 bits in {n}: {count_bits(n)}")  # Expected: 2
    
    # Example 6: Dynamic Programming
    print("\nDynamic Programming Example:")
    n = 5
    print(f"Number of ways to climb {n} stairs: {climb_stairs(n)}")  # Expected: 8
    
    print("\nAll examples executed successfully!")