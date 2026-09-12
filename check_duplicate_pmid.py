from sqlalchemy import text
from app.database.session import SessionLocal


db = SessionLocal()

try:
    result = db.execute(
        text(
            """
            SELECT pmid, COUNT(*)
            FROM literature
            WHERE pmid IS NOT NULL
            GROUP BY pmid
            HAVING COUNT(*) > 1
            """
        )
    )

    duplicates = list(result)

    print("Duplicate PMIDs:")
    print(duplicates)

finally:
    db.close()
