from app.services.alphagenome.service import AlphaGenomeService


def main():
    service = AlphaGenomeService()

    result = service.predict_variant(
        chromosome="chr22",
        position=36201698,
        reference_bases="A",
        alternate_bases="C",
        start=35677410,
        end=36725986,
        ontology_terms=["UBERON:0001157"],
    )

    print("AlphaGenomeService SUCCESS")
    print()

    print("Variant:")
    print(result["variant"])

    print("\nInterval:")
    print(result["interval"])

    print("\nOutput type:")
    print(result["output_type"])

    print("\nReference summary:")
    print(result["reference"])

    print("\nAlternate summary:")
    print(result["alternate"])

    print("\nEffect summary:")
    print(result["effect"])

    print("\nMetadata:")
    print(result["metadata"])


if __name__ == "__main__":
    main()