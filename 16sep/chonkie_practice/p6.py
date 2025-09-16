from chonkie import TokenChunker

# Configure for large documents
chunker = TokenChunker(
    tokenizer="gpt2",
    chunk_size=4096,  # Larger chunks for efficiency
    chunk_overlap=30  # Maintain context between chunks
)

# Read a large document
# with open("large_document.txt", "r") as f:
#     large_text = f.read()

large_text = """
Born and raised in the vibrant city of Chittagong, my journey has taken me from its lively surroundings to the academic corridors of Sylhet, shaped by an enduring passion for engineering and discovery. I see myself not merely as an engineer, but as a lifelong learner committed to bridging the frontiers of Artificial Intelligence and Geoscience to address challenges that truly matter. My philosophy is simple: knowledge must serve a higher purpose, it should solve problems, protect lives, and build resilience for the future. I thrive at the intersection of curiosity and discipline, combining rigorous technical skills with a mindset grounded in humility, perseverance, and silent determination. Always seeking to grow, I aspire to contribute to research environments where ideas transform into solutions and personal effort aligns with collective progress. I am ready to join MS/PhD research opportunities.
"""

# Process efficiently
chunks = chunker.chunk(large_text)

print(f"Document statistics:")
print(f"  Original length: {len(large_text)} characters")
print(f"  Number of chunks: {len(chunks)}")
print(f"  Average chunk size: {sum(c.token_count for c in chunks) / len(chunks):.1f} tokens")

# Save chunks for further processing
for i, chunk in enumerate(chunks):
    with open(f"chunk_{i:03d}.txt", "w") as f:
        f.write(chunk.text)