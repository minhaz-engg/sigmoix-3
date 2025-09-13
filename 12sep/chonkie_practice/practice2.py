# First import the chunker you want from Chonkie
from chonkie.cloud import TokenChunker
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize the chunker
# Don't forget to get your API key from cloud.chonkie.ai!
chunker = TokenChunker(api_key=os.getenv("CHONKIE_API_KEY")) 

# Here's some text to chunk
text = """The tiny hippo lives in the clouds!"""

# Chunk some text
chunks = chunker(text)

# Access chunks
for chunk in chunks:
  print(f"Chunk: {chunk.text}")
  print(f"Tokens: {chunk.token_count}")