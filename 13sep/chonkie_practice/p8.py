from chonkie import SemanticChunker

text = """Born and raised in the vibrant city of Chittagong, my journey has taken me from its lively surroundings to the academic corridors of Sylhet, shaped by an enduring passion for engineering and discovery. I see myself not merely as an engineer, but as a lifelong learner committed to bridging the frontiers of Artificial Intelligence and Geoscience to address challenges that truly matter. My philosophy is simple: knowledge must serve a higher purpose, it should solve problems, protect lives, and build resilience for the future. I thrive at the intersection of curiosity and discipline, combining rigorous technical skills with a mindset grounded in humility, perseverance, and silent determination. Always seeking to grow, I aspire to contribute to research environments where ideas transform into solutions and personal effort aligns with collective progress. I am ready to join MS/PhD research opportunities.
Alhamdulillah, just before my bachelor defence,
Our research paper titled "Comparative Analysis of Deep Learning Architectures for Multi-class Mineral Classification: A Study Using EfficientNet and ResNet Models" has officially been published in the journal Earth Science Informatics (Springer). 
Congratulation to my teammate Shreoshi Roy. Above all, I extend my deepest gratitude to our respected supervisor, Professor Dr. Md. Shofiqul Islam, Ph.D sir, whose constant support, wisdom, and guidance have been instrumental in shaping this work from to publication. JazakAllahu Khairan.
Kindly keep us in your prayers as we continue our journey in research and learning, insha’Allah.
"""

# Create semantic chunker
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",
    threshold=0.75,  # Higher threshold = more similar content grouped
    chunk_size=1024
)

chunks = chunker.chunk(text)

# Analyze semantic groupings
for i, chunk in enumerate(chunks):
    print(f"\n--- Semantic Group {i+1} ---")
    print(f"Content: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
    print(f"Theme: {chunk.text.split('.')[0]}")  # First sentence as theme indicator