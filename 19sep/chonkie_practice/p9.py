from chonkie import SemanticChunker

# Text with alternating topics
text = """Neural networks process information through interconnected nodes.
The stock market experienced significant volatility this quarter.
Deep learning models require substantial training data for optimization.
Economic indicators point to potential recession risks ahead.
GPU acceleration has revolutionized machine learning computations.
Federal reserve policies impact global financial markets.
Transformer architectures dominate modern NLP applications.
Cryptocurrency markets show correlation with traditional assets."""

# Enable skip-window to merge non-consecutive similar content
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",
    threshold=0.65,
    chunk_size=512,
    skip_window=2  # Look ahead 2 groups for similar content
)

chunks = chunker.chunk(text)

# AI-related content will be grouped together
# Financial content will be grouped separately
for i, chunk in enumerate(chunks):
    print(f"\nGroup {i+1}: {len(chunk.text.split('.'))} sentences")
    print(f"Preview: {chunk.text}")