# # First import the chunker you want from Chonkie
# from chonkie import RecursiveChunker

# # Initialize the chunker
# chunker = RecursiveChunker()

# # Chunk some text
# chunks = chunker("Chonkie is the goodest boi! My favorite chunking hippo hehe.")

# # Access chunks
# for chunk in chunks:
#     print(f"Chunk: {chunk.text}")
#     print(f"Tokens: {chunk.token_count}")

# First import the chunker you want from Chonkie 
from chonkie import TokenChunker

# Initialize the chunker
chunker = TokenChunker() # defaults to using GPT2 tokenizer

# Here's some text to chunk
text = """Woah! Chonkie, the chunking library is so cool!Woah! Chonkie, the chunking library is so cool!"""

# Chunk some text
chunks = chunker(text)

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")