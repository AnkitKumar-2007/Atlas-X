from app.config import settings

from alphagenome.data import genome
from alphagenome.models import dna_client


def main():
    if not settings.alphagenome_api_key:
        raise RuntimeError(
            "ALPHAGENOME_API_KEY is not configured"
        )

    print("Creating AlphaGenome client...")

    model = dna_client.create(
        settings.alphagenome_api_key
    )

    # Example human genomic interval
    interval = genome.Interval(
        chromosome="chr22",
        start=35677410,
        end=36725986,
    )

    # Example SNV
    variant = genome.Variant(
        chromosome="chr22",
        position=36201698,
        reference_bases="A",
        alternate_bases="C",
    )

    print("Running AlphaGenome variant prediction...")
    print(f"Variant: {variant}")

    outputs = model.predict_variant(
        interval=interval,
        variant=variant,
        ontology_terms=["UBERON:0001157"],
        requested_outputs=[
            dna_client.OutputType.RNA_SEQ
        ],
    )

    print("\nAlphaGenome prediction successful!")

    print("\nAvailable output object:")
    print(type(outputs))

    print("\nReference RNA-seq:")
    print(type(outputs.reference.rna_seq))

    print("\nAlternate RNA-seq:")
    print(type(outputs.alternate.rna_seq))


if __name__ == "__main__":
    main()