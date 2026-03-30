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

        # src package
        base / "src" / SRC_PACKAGE / "__init__.py",

        # ingestion
        base / "src" / SRC_PACKAGE / "ingestion" / "__init__.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "cleaner.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "validator.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "pipeline.py",

        # ingestion sources
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "__init__.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "newsapi.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "rss.py",
        base / "src" / SRC_PACKAGE / "ingestion" / "sources" / "sec_edgar.py",

        # database
        base / "src" / SRC_PACKAGE / "database" / "__init__.py",
        base / "src" / SRC_PACKAGE / "database" / "models.py",
        base / "src" / SRC_PACKAGE / "database" / "session.py",
        base / "src" / SRC_PACKAGE / "database" / "migrations" / ".gitkeep",

        # monitoring
        base / "src" / SRC_PACKAGE / "monitoring" / "__init__.py",
        base / "src" / SRC_PACKAGE / "monitoring" / "metrics.py",
        base / "src" / SRC_PACKAGE / "monitoring" / "logging.py",

        # dags
        base / "src" / SRC_PACKAGE / "dags" / "__init__.py",
        base / "src" / SRC_PACKAGE / "dags" / "ingestion_dag.py",

        # core
        base / "src" / SRC_PACKAGE / "core" / "__init__.py",
        base / "src" / SRC_PACKAGE / "core" / "config.py",
        base / "src" / SRC_PACKAGE / "core" / "constants.py",
        base / "src" / SRC_PACKAGE / "core" / "exceptions.py",

        # services
        base / "src" / SRC_PACKAGE / "services" / "__init__.py",
        base / "src" / SRC_PACKAGE / "services" / "sentiment.py",
        base / "src" / SRC_PACKAGE / "services" / "event_detection.py",
        base / "src" / SRC_PACKAGE / "services" / "impact_scoring.py",
        base / "src" / SRC_PACKAGE / "services" / "ranking.py",

        # API
        base / "src" / SRC_PACKAGE / "api" / "__init__.py",
        base / "src" / SRC_PACKAGE / "api" / "main.py",

        # tests (outside src)
        base / "tests" / "__init__.py",
        base / "tests" / "test_ingestion.py",
    ]

    for path in structure:
        create_file(path)

    print("✅ Structure created inside current root directory")


if __name__ == "__main__":
    create_structure()