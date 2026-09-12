from app.database.session import SessionLocal
from app.services.rag.service import RAGService


def main():
    db = SessionLocal()

    try:
        rag = RAGService()

        paper = rag.add_literature(
            db=db,
            title="Regulatory effects of genetic variation on gene expression",
            abstract=(
                "Genetic variants can alter transcription factor binding, "
                "chromatin accessibility, gene expression, and regulatory "
                "activity across human tissues."
            ),
            journal="Genome Biology",
            publication_year=2025,
        )

        print("Literature inserted successfully")
        print("ID:", paper.id)
        print("Title:", paper.title)

        results = rag.search(
            db=db,
            query="How can a genetic variant affect gene regulation?",
            limit=5,
        )

        print("\nSemantic search results:")

        for result in results:
            print("-", result.title)

    finally:
        db.close()


if __name__ == "__main__":
    main()
