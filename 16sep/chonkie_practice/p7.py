from chonkie import TokenChunker

# Initialize once
chunker = TokenChunker(
    tokenizer="gpt2",
    chunk_size=1024,
    chunk_overlap=10
)

# Use as a callable for single text
single_text = "This is a document that needs chunking..."
chunks = chunker(single_text)
print(f"Single text produced {len(chunks)} chunks")

# Use as a callable for multiple texts
multiple_texts = [
    "First document text...",
    "Second document text...",
    "Third document text..."
]
batch_results = chunker(multiple_texts)
print(f"Processed {len(batch_results)} documents")