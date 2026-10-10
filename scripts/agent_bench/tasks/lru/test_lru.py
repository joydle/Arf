from lru import LRUCache

c = LRUCache(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") is None, "b was least recently used"
assert c.get("a") == 1 and c.get("c") == 3
c.put("a", 10)
assert c.get("a") == 10 and len(c) == 2
c.put("d", 4)
assert c.get("c") is None, "c evicted after a was refreshed"
assert len(c) == 2
print("OK")
