from chonkie import SemanticChunker

text = """Neural networks process information through interconnected nodes.
The stock market experienced significant volatility this quarter.
Deep learning models require substantial training data for optimization.
Economic indicators point to potential recession risks ahead.
GPU acceleration has revolutionized machine learning computations.
Federal reserve policies impact global financial markets.
Transformer architectures dominate modern NLP applications.
Cryptocurrency markets show correlation with traditional assets."""

# Experiment with different thresholds
thresholds = [0.5, 0.7, 0.9]

for threshold in thresholds:
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",
        threshold=threshold,
        chunk_size=512,
        similarity_window=3  # Consider 3 sentences for similarity
    )
    
    chunks = chunker.chunk(text)
    print(f"\nThreshold {threshold}: {len(chunks)} chunks created")
    
    # Lower threshold = larger, more diverse chunks
    # Higher threshold = smaller, more focused chunks
    avg_size = sum(c.token_count for c in chunks) / len(chunks)
    print(f"Average chunk size: {avg_size:.1f} tokens")
    print("Chunks preview:")
    for chunk in chunks:
        print(f" - {chunk.text}")