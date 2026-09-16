from app.database.setup import create_database, table_names


def main() -> None:
    database_url = create_database()
    tables = ", ".join(table_names()) or "(none)"
    print(f"Database ready: {database_url}")
    print(f"Tables: {tables}")


if __name__ == "__main__":
    main()
