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

if __name__=="__main__":
    lru_cache()

            

            
            