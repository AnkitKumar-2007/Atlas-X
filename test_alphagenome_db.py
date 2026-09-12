from app.database.session import SessionLocal
from app.services.alphagenome.prediction_service import (
    AlphaGenomePredictionService,
)


def main():
    db = SessionLocal()

    try:
        service = AlphaGenomePredictionService()

        prediction = service.predict_and_store(
            db=db,
            chromosome="chr22",
            position=36201698,
            reference="A",
            alternate="C",
            start=35677410,
            end=36725986,
            genome_build="GRCh38",
            ontology_terms=["UBERON:0001157"],
        )

        print("AlphaGenome database integration SUCCESS")
        print()

        print("Prediction ID:")
        print(prediction.id)

        print("\nVariant ID:")
        print(prediction.variant_id)

        print("\nOutput type:")
        print(prediction.output_type)

        print("\nOntology:")
        print(prediction.ontology_term)

        print("\nReference score:")
        print(prediction.reference_score)

        print("\nAlternate score:")
        print(prediction.alternate_score)

        print("\nEffect score:")
        print(prediction.effect_score)

        print("\nPrediction JSON keys:")
        print(list(prediction.prediction.keys()))

    finally:
        db.close()


if __name__ == "__main__":
    main()