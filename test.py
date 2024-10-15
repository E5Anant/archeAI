import os

with open("cache.txt", 'r', encoding="utf-8") as f:
    content = f.read()
    print(content)