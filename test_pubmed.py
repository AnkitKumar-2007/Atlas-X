from app.services.rag.pubmed import PubMedService


def main():
    pubmed = PubMedService()

    pmids = pubmed.search(
        query="genetic variants gene regulation",
        limit=3,
    )

    print("PMIDs found:")
    for pmid in pmids:
        print("-", pmid)

    papers = pubmed.fetch(pmids)

    print("\nPapers fetched:")

    for paper in papers:
        print("\nTitle:", paper["title"])
        print("PMID:", paper["pmid"])
        print("Journal:", paper["journal"])
        print("Year:", paper["publication_year"])
        print("Abstract:", paper["abstract"][:300], "...")


if __name__ == "__main__":
    main()
