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
        base / "artifacts",
        base / "notebook"

        # src package                                                               ✔
        base / "src" / SRC_PACKAGE / "__init__.py",

        # etl_source (First the data is collected from the source API (AlphaVantage), more data_sources will be added in future, this is just a baseline)
        base / "src" / SRC_PACKAGE / "etl" / "__init__.py"
        base / "src" / SRC_PACKAGE / "etl" / "extract_from_sources" / "__init__.py",
        base / "src" / SRC_PACKAGE / "etl" / "extract_from_sources" / "alphavantage.py", # setup the connection to VA(vantage_api) and get data

        # etl (After the data is collected, then it is cleaned and labelled from LLM)
        base / "src" / SRC_PACKAGE / "etl" / "transform" / "__init__.py",
        base / "src" / SRC_PACKAGE / "etl" / "transform" / "labeller_LLM" / "__init__.py",
        base / "src" / SRC_PACKAGE / "etl" / "transform" / "labeller_LLM" / "label.py",
        base / "src" / SRC_PACKAGE / "etl" / "transform" / "clean.py", # To clean the data collected

        base / "src" / SRC_PACKAGE / "etl" / "load" / "__init__.py", 
        base / "src" / SRC_PACKAGE / "etl" / "load" / "configuration.py", # To Set Connection with PostGreSql Database
        base / "src" / SRC_PACKAGE / "etl" / "load" / "load.py", # To Set Connection with PostGreSql Database


        # database                                                                  ✔
        base / "src" / SRC_PACKAGE / "database" / "__init__.py",
        base / "src" / SRC_PACKAGE / "database" / "configuration.py", # To make connection with PostGreSQL
        base / "src" / SRC_PACKAGE / "database" / "load.py", # Load the Transformed Data
        base / "src" / SRC_PACKAGE / "database" / "migrations" / ".gitkeep", 

        # core                                                                      ✔ 
        base / "src" / SRC_PACKAGE / "core" / "__init__.py",
        base / "src" / SRC_PACKAGE / "core" / "config.py",
        base / "src" / SRC_PACKAGE / "core" / "constants.py", # Define all the constants being used in the codebase
        base / "src" / SRC_PACKAGE / "core" / "exceptions.py", # Define the exception module
        base / "src" / SRC_PACKAGE / "core" / "logger.py", # Define the logger module

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