from app.config import settings

from alphagenome.data import genome
from alphagenome.models import dna_client


def main():
    model = dna_client.create(
        settings.alphagenome_api_key
    )

    interval = genome.Interval(
        chromosome="chr22",
        start=35677410,
        end=36725986,
    )

    variant = genome.Variant(
        chromosome="chr22",
        position=36201698,
        reference_bases="A",
        alternate_bases="C",
    )

    outputs = model.predict_variant(
        interval=interval,
        variant=variant,
        ontology_terms=["UBERON:0001157"],
        requested_outputs=[
            dna_client.OutputType.RNA_SEQ
        ],
    )

    ref = outputs.reference.rna_seq
    alt = outputs.alternate.rna_seq

    print("REFERENCE TRACK DATA")
    print("====================")

    print("Type:", type(ref))
    print("Shape:", getattr(ref, "shape", None))
    print("Values:", getattr(ref, "values", None))
    print("Coordinates:", getattr(ref, "coordinates", None))

    print("\nALTERNATE TRACK DATA")
    print("====================")

    print("Type:", type(alt))
    print("Shape:", getattr(alt, "shape", None))
    print("Values:", getattr(alt, "values", None))
    print("Coordinates:", getattr(alt, "coordinates", None))

    print("\nPUBLIC ATTRIBUTES")
    print("=================")

    print(
        [
            name
            for name in dir(ref)
            if not name.startswith("_")
        ]
    )


if __name__ == "__main__":
    main()
