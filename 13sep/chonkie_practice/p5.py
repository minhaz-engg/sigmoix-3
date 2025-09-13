from chonkie import TokenChunker

# Initialize chunker for batch processing
chunker = TokenChunker(
    tokenizer="gpt2",
    chunk_size=512,
    chunk_overlap=50
)

# Multiple documents to process
documents = [
    "First document about machine learning fundamentals",
    "Second document discussing neural networks",
    "Third document on natural language processing"
]

# Process all documents at once
batch_chunks = chunker.chunk_batch(documents)

# Iterate through results
for doc_idx, doc_chunks in enumerate(batch_chunks):
    print(f"\nDocument {doc_idx + 1}: {len(doc_chunks)} chunks")
    for chunk in doc_chunks:
        print(f"  - Chunk: {chunk.text}, ({chunk.token_count} tokens)")