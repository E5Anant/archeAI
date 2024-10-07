import tiktoken

print(tiktoken.get_encoding("gpt-3.5-turbo").encode("hello"))
