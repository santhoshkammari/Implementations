"""
All Sorting Algorithms
# Bubble sort
# Selection sort
# Insertion sort
# Merge sort
# Quick sort
# Heap sort
# Counting sort
# Radix sort
# Bucket sort
# Tim sort

Algorithm       | Best Time  | Average Time | Worst Time  | Space
----------------|------------|--------------|-------------|-------
Bubble Sort     | O(n)       | O(n²)        | O(n²)       | O(1)
Selection Sort  | O(n²)      | O(n²)        | O(n²)       | O(1)
Insertion Sort  | O(n)       | O(n²)        | O(n²)       | O(1)
Merge Sort      | O(n log n) | O(n log n)   | O(n log n)  | O(n)
Quick Sort      | O(n log n) | O(n log n)   | O(n²)       | O(log n)
Heap Sort       | O(n log n) | O(n log n)   | O(n log n)  | O(1)
Counting Sort   | O(n+k)     | O(n+k)       | O(n+k)      | O(n+k)
Radix Sort      | O(nk)      | O(nk)        | O(nk)       | O(n+k)
Bucket Sort     | O(n+k)     | O(n+k)       | O(n²)       | O(n)
Tim Sort        | O(n)       | O(n log n)   | O(n log n)  | O(n)
"""

def bubble_sort(arr):
    """Sorts an array using Bubble Sort.

    It repeatedly steps through the list, compares adjacent elements
    and swaps them if they are in the wrong order. The pass through
    the list is repeated until the list is sorted.
    """
    for i in range(len(arr),0,-1):
        swapped = False
        for j in range(i-1):
            if arr[j]>arr[j+1]:
                temp = arr[j]
                arr[j]=arr[j+1]
                arr[j+1] = temp
                swapped = True 
        if not swapped:
            break

def selection_sort(arr):
    """Sorts an array using Selection Sort."""
    pass

def insertion_sort(arr):
    """Sorts an array using Insertion Sort."""
    pass

def merge_sort(arr):
    """Sorts an array using Merge Sort."""
    pass

def quick_sort(arr):
    """Sorts an array using Quick Sort."""
    pass

def heap_sort(arr):
    """Sorts an array using Heap Sort."""
    pass

def counting_sort(arr, max_val):
    """Sorts an array using Counting Sort."""
    pass

def radix_sort(arr):
    """Sorts an array using Radix Sort."""
    pass

def bucket_sort(arr):
    """Sorts an array using Bucket Sort."""
    pass

def tim_sort(arr):
    """Sorts an array using Tim Sort."""
    pass

# Note: Binary sort is typically a search algorithm, not a primary sorting algorithm.
# Insertion sort can use binary search to find the insertion point,
# but it's usually referred to as Binary Insertion Sort.
# If you meant Binary Insertion Sort:
def binary_insertion_sort(arr):
    """Sorts an array using Insertion Sort with binary search."""
    pass


if __name__=="__main__":
    a = [2,3,4,1]
    # Choose the sorting function you want to test, e.g., bubble_sort
    # res = binary_sort(a) # Original line had an undefined function
    bubble_sort(a) # Example call
    print(a) # Print the sorted array (most sorts modify in-place)