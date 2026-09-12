from app.services.embeddings.gemini import GeminiEmbeddingService


service = GeminiEmbeddingService()

text = """
AlphaGenome predicts how genomic variants can influence
gene regulation, gene expression, chromatin accessibility,
and RNA splicing.
"""

embedding = service.embed(text)

print("Embedding generated successfully")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])