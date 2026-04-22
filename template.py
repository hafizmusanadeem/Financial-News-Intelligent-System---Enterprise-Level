from pathlib import Path

SRC_PACKAGE = "fni"


def create_file(path: Path) -> None:
    """Create an empty file if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def create_structure():
    """Create project structure inside existing root directory."""

    base = Path.cwd()  # current project root

    structure = [
        # Root files
        base / ".env",
        base / "README.md",
        base / "pyproject.toml",

        # src package                                                               ✔
        base / "src" / SRC_PACKAGE / "__init__.py",

        # ingestion_source
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "__init__.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "alphavantage.py", # setup the connection to VA(vantage_api) and get data

        # ingestion
        base / "src" / SRC_PACKAGE / "ingestion" / "__init__.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "cleaner.py", # To clean the data collected
        base / "src" / SRC_PACKAGE / "ingestion" / "pipeline.py", # To run the process multiple times

        # database                                                                  ✔
        base / "src" / SRC_PACKAGE / "database" / "__init__.py",
        base / "src" / SRC_PACKAGE / "database" / "configuration.py", # To make connection with PostGreSQL
        base / "src" / SRC_PACKAGE / "database" / "load.py", # Load the Transformed Data
        base / "src" / SRC_PACKAGE / "database" / "migrations" / ".gitkeep", 


        # labelling
        base / "src" / SRC_PACKAGE / "labeling" / "__init__.py",
        base / "src" / SRC_PACKAGE / "labeling" / "event_labeler.py",

        # core                                                                      ✔ 
        base / "src" / SRC_PACKAGE / "core" / "__init__.py",
        base / "src" / SRC_PACKAGE / "core" / "config.py",
        base / "src" / SRC_PACKAGE / "core" / "constants.py", # Define all the constants being used in the codebase
        base / "src" / SRC_PACKAGE / "core" / "exceptions.py", # Define the exception module
        base / "src" / SRC_PACKAGE / "core" / "logging.py", # Define the logging module

        # API
        base / "src" / SRC_PACKAGE / "api" / "__init__.py",
        base / "src" / SRC_PACKAGE / "api" / "main.py", # Write the FastAPI endpoints

        # tests (outside src)
        base / "tests" / "__init__.py",
        base / "tests" / "test_ingestion.py",
    ]

    for path in structure:
        create_file(path)

    print("✅ Structure created inside current root directory")


if __name__ == "__main__":
    create_structure()