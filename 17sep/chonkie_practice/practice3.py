from chonkie import SemanticChunker

# Customize sentence detection
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",
    threshold=0.7,
    chunk_size=1024,
    min_sentences_per_chunk=3,   # At least 3 sentences per chunk
    min_characters_per_sentence=30,  # Filter out short fragments
    delim=[". ", "! ", "? ", "\n\n"],  # Custom sentence delimiters
    include_delim="prev"  # Include delimiter with previous sentence
)

# Text with various sentence structures
text = """Short sentence. This is a much longer sentence with more detail.
Question here? Exclamation point! New paragraph starts here.

# another paragraph with different content: Machine learning models can now understand context, generate text, and even translate
between languages with remarkable accuracy. This transformation has enabled applications
ranging from virtual assistants to automated content generation.

Quantum computing represents a new frontier in computational power.By leveraging the principles of quantum mechanics,
these advanced systems have the potential to solve complex problems
that are currently intractable for classical computers.

Mining Engineering is a broad field that encompasses various disciplines.
It involves the extraction of minerals from the earth and the processing of these materials
to create valuable products. This field combines elements of geology, engineering, and environmental science
to ensure that mining operations are efficient, safe, and sustainable.
Environmental science plays a crucial role in modern mining practices.

aeronautical engineering focuses on the design, development, and testing of aircraft and spacecraft.
This field requires a deep understanding of aerodynamics, materials science, and propulsion systems.
Aeronautical engineers work on a variety of projects, from commercial airplanes to military jets and space
exploration vehicles. The goal is to create safe, efficient, and innovative designs that meet the
demands of modern aviation and space travel.

geophysics is the study of the physical properties and processes of the Earth.
This field encompasses a wide range of topics, including seismic activity, magnetic and gravitational fields,
and the internal structure of the planet. Geophysicists use various techniques, such as seismic surveys and remote sensing,
to gather data and develop models that help us understand the Earth's behavior and evolution.
"""

chunks = chunker.chunk(text)

for chunk in chunks:
    sentences = chunk.text.split('. ')
    print(f"Chunk with {len(sentences)} sentences")
    print(f"Preview: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
    print(f"Start index: {chunk.start_index}")
    print(f"End index: {chunk.end_index}\n")