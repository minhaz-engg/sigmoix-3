from chonkie import TokenChunker

# Create a chunker with specific parameters
chunker = TokenChunker(
    tokenizer="gpt2",
    chunk_size=1024,
    chunk_overlap=128
)

text = """Natural language processing has revolutionized how we interact with computers.
Machine learning models can now understand context, generate text, and even translate
between languages with remarkable accuracy. This transformation has enabled applications
ranging from virtual assistants to automated content generation."""

# Chunk the text
chunks = chunker.chunk(text)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Text: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
    print(f"Start index: {chunk.start_index}")
    print(f"End index: {chunk.end_index}")