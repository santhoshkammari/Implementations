def lru_cache():
    """
    doubly linked list[O(1) for Ins & Del] + hashmap[O(1) lookup] 
    for effiecinet caching by removing least used item when overflowed
    """
    class Node:
        def __init__(self,k,v):
            self.k=k
            self.v=v
            self.prev = None
            self.next = None

    class LRUCache:
        def __init__(self,size):
            self.size = size
            self.cache = {}

            self.head = Node(0,0)
            self.tail = Node(0,0)
            self.head.next = self.tail
            self.tail.prev = self.head

        def _add(self,node):
            p = self.head
            n = self.head.next
            p.next = node
            node.prev = p
            node.next = n
            n.prev = node

        def _remove(self,node):
            p = node.prev
            n = node.next
            p.next = n
            n.prev = p


        def put(self,k,v):
            if k in self.cache:
                self._remove(self.cache[k])

            node = Node(k,v)
            self._add(node)
            self.cache[k]=node

            if len(self.cache)>self.size:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.k]

        def get(self,k):
            if k in self.cache:
                node = self.cache[k]
                self._remove(node)
                self._add(node)
                return node.v
            return -1

        # Example usage
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))      # Output: 1
    cache.put(3, 3)          # evicts key 2
    print(cache.get(2))      # Output: -1 (not found)


def length_of_longest_substring(s):
    # Dictionary to store the last position of each character
    chars_pos = {}
    start = 0
    max_length = 0

    for i,char in enumerate(s):
        if char in chars_pos and chars_pos[char]>=start:
            start = chars_pos[char] + 1
        max_length = max(max_length,i-start+1)
        chars_pos[char]=i

    return max_length
    
def subarray_sum(nums, k):
    # Dictionary to store cumulative sum frequencies
    count = 0
    cum_sum = 0
    sum_map = {0:1}
    for value in nums:
        cum_sum+=value
        count+=sum_map.get(cum_sum-k,0)
        sum_map[cum_sum] = sum_map.get(cum_sum,0) + 1
    
    return count


if __name__=="__main__":
    lru_cache()
    # Example usage
    s = "abcabcbb"
    print(length_of_longest_substring(s))  # Output: 3 ("abc")
    # Example usage
    nums = [1, 1, 1]
    k = 2
    print(subarray_sum(nums, k))  # Output: 2 (subarrays: [1,1], [1,1])



            

            
            