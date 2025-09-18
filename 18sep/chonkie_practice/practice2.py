from chonkie import SemanticChunker

# Initialize chunker once
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",
    threshold=0.7,
    chunk_size=1024,
    min_sentences_per_chunk=2  # Ensure meaningful chunks
)

# Multiple documents with different topics
documents = [
    """Natural language processing has revolutionized how we interact with computers.
Machine learning models can now understand context, generate text, and even translate
between languages with remarkable accuracy. This transformation has enabled applications
ranging from virtual assistants to automated content generation.""",
    """Climate change is one of the most pressing issues of our time.
The impacts of global warming are being felt across the planet,
from rising sea levels to more frequent and severe weather events.
Addressing climate change requires urgent action from individuals,
businesses, and governments alike.""",
    """Quantum computing represents a new frontier in computational power.
By leveraging the principles of quantum mechanics,
these advanced systems have the potential to solve complex problems
that are currently intractable for classical computers."""
]

# Process all documents
batch_results = chunker.chunk_batch(documents)

# Analyze results
for doc_idx, chunks in enumerate(batch_results):
    print(f"\nDocument {doc_idx + 1}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total tokens: {sum(c.token_count for c in chunks)}")
    
    # Show semantic boundaries
    for i, chunk in enumerate(chunks):
        first_sentence = chunk.text.split('.')[0]
        print(f"  Chunk {i+1}: {first_sentence}")