from app.config import settings

from alphagenome.data import genome
from alphagenome.models import dna_client


def inspect_track(name, track):
    print(f"\n{name}")
    print("=" * len(name))

    print("Interval:", track.interval)
    print("Width:", track.width)
    print("Number of tracks:", track.num_tracks)
    print("Resolution:", track.resolution)
    print("Names:", track.names)
    print("Strands:", track.strands)
    print("Ontology terms:", track.ontology_terms)

    print("\nMetadata:")
    print(track.metadata)

    print("\nPositional axes:")
    print(track.positional_axes)

    values = track.values

    print("\nValues:")
    print("Type:", type(values))
    print("Rows:", len(values))
    print("Columns:", len(values[0]) if len(values) else 0)

    if len(values):
        print("First row:", values[0])
        print("Last row:", values[-1])


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

    inspect_track(
        "REFERENCE RNA-SEQ",
        outputs.reference.rna_seq,
    )

    inspect_track(
        "ALTERNATE RNA-SEQ",
        outputs.alternate.rna_seq,
    )


if __name__ == "__main__":
    main()