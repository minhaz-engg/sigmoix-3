from chonkie import TokenChunker

# Initialize the chunker
chunker = TokenChunker(
    tokenizer="gpt2",
    chunk_size=512,
    chunk_overlap=50
)

# Chunk your text
text = """
Born and raised in the vibrant city of Chittagong, my journey has taken me from its lively surroundings to the academic corridors of Sylhet, shaped by an enduring passion for engineering and discovery. I see myself not merely as an engineer, but as a lifelong learner committed to bridging the frontiers of Artificial Intelligence and Geoscience to address challenges that truly matter. My philosophy is simple: knowledge must serve a higher purpose, it should solve problems, protect lives, and build resilience for the future. I thrive at the intersection of curiosity and discipline, combining rigorous technical skills with a mindset grounded in humility, perseverance, and silent determination. Always seeking to grow, I aspire to contribute to research environments where ideas transform into solutions and personal effort aligns with collective progress. I am ready to join MS/PhD research opportunities.
"""
chunks = chunker.chunk(text)

# Access chunk information
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")