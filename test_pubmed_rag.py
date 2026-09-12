from app.database.session import SessionLocal
from app.services.rag.pubmed import PubMedService
from app.services.rag.service import RAGService


def main():
    db = SessionLocal()

    try:
        pubmed = PubMedService()
        rag = RAGService()

        pmids = pubmed.search(
            query="genetic variants gene regulation",
            limit=5,
        )

        print(f"Found {len(pmids)} PubMed papers")

        papers = pubmed.fetch(pmids)

        print(f"Fetched {len(papers)} papers")

        inserted = 0
        skipped = 0

        for paper in papers:

            if not paper["abstract"].strip():
                print(
                    f"Skipped {paper['pmid']} - no abstract"
                )
                skipped += 1
                continue

            result, was_inserted = rag.add_literature(
                db=db,
                title=paper["title"],
                abstract=paper["abstract"],
                pmid=paper["pmid"],
                journal=paper["journal"],
                publication_year=paper["publication_year"],
            )

            if was_inserted:
                inserted += 1
                print(
                    f"Inserted: {paper['pmid']} - "
                    f"{paper['title'][:70]}"
                )
            else:
                skipped += 1
                print(
                    f"Skipped: {paper['pmid']} "
                    f"(already exists)"
                )

        print("\nIngestion summary:")
        print("Inserted:", inserted)
        print("Skipped:", skipped)

        query = "How do genetic variants affect gene regulation?"

        print(f"\nSemantic query:\n{query}")

        results = rag.search(
            db=db,
            query=query,
            limit=5,
        )

        print("\nTop semantic matches:")

        for i, result in enumerate(results, start=1):
            print(
                f"{i}. {result.title} "
                f"(PMID: {result.pmid})"
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()