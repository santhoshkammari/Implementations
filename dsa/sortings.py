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

Fun Fact:
quick and merge sort are inspired from divide and conquer
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
    for i in range(len(arr)):
        min_i = i
        for j in range(i+1,len(arr)):
            if arr[j]<arr[min_i]:
                min_i=j
        arr[i],arr[min_i] = arr[min_i],arr[i]

def insertion_sort(arr):
    """Sorts an array using Insertion Sort."""
    for i in range(1,len(arr)):
        for j in range(i,0,-1):
            if arr[j-1]>arr[j]:
                arr[j],arr[j-1]=arr[j-1],arr[j]

def merge_sort(arr):
    """Sorts an array using Merge Sort."""
    if len(arr)==1:return arr
    mid = len(arr)//2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    
    res,i,j = [],0,0
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
            res.append(left[i])
            i+=1
        else:
            res.append(right[j])
            j+=1
    
    return res + left[i:] + right[j:]



def quick_sort(arr,low=0,high=None):
    if high is None:
        high = len(arr)-1

    def partition(arr,low,high):
        pivot = arr[high]
        i = low -1
        for j in range(low,high):
            if arr[j]<=pivot:
                i+=1
                arr[i],arr[j]=arr[j],arr[i]
        arr[i+1],arr[high]=arr[high],arr[i+1]
        return i+1
    
    if low<high:
        pivot = partition(arr,low,high)
        quick_sort(arr,low,pivot-1)
        quick_sort(arr,pivot+1,high)

    return arr


def heap_sort(arr):
    """
    Heap Sort uses a binary heap data structure and works in two phases:
    1. Build a max-heap from the input array (parent > children)
    2. swapping root with last element
    
    TC: O(n log n) 
    SC: O(1)
    
    The name "heap" comes from memory allocation, but in this context
    it refers to a complete binary tree with the heap property.
    
    Process:
    - Build max-heap: Start from last non-leaf node (n//2-1) and heapify up
    - Extract phase: Swap root (max) with last unsorted element, reduce heap size,
      then heapify to maintain heap property

    In a binary heap represented as an array:

    Nodes at indices greater than (n-1)//2 are leaf nodes
    Nodes at indices (n-1)//2 down to 0 are non-leaf nodes (they have at least one child)

    For example, in a heap with 10 elements:

    The last non-leaf node would be at index (10-1)//2 = 4
    Nodes at indices 5, 6, 7, 8, and 9 would be leaf nodes
    Nodes at indices 0, 1, 2, 3, and 4 would be non-leaf nodes
    """
    n = len(arr)

    def heapify(arr,j,i):
        p,l,r = i,2*i+1,2*i+2
        if l<j and arr[p]<arr[l]:
            p = l
        if r<j and arr[p]<arr[r]:
            p = r
        if p!=i:
            arr[i],arr[p] = arr[p],arr[i]
            heapify(arr,j,p)

    # max heap
    for node_idx in range((n - 1)//2, -1, -1):
        heapify(arr,len(arr),node_idx)

    #swap root to end and do max heap , total n-1 times.
    for last_node_idx in range(n-1,0,-1):
        arr[0],arr[last_node_idx] = arr[last_node_idx],arr[0]
        heapify(arr,last_node_idx,0)

    return arr
        


if __name__=="__main__":
    a = [5,2,4,6,1,3]
    # Choose the sorting function you want to test, e.g., bubble_sort
    # res = binary_sort(a) # Original line had an undefined function
    print(quick_sort(a))