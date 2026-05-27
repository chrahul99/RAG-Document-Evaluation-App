from pathlib import Path

from utils.config import AppConfig


def _delete_children(folder: Path, keep_gitkeep: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if keep_gitkeep and item.name == ".gitkeep":
            continue
        if item.is_dir():
            for child in item.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(item.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            item.rmdir()
        else:
            item.unlink()


def reset_knowledge_base(config: AppConfig, remove_uploads: bool = True) -> None:
    """Clear ChromaDB, history, logs, and optionally uploaded PDFs."""
    _delete_children(config.vectorstore_dir)

    if config.history_db_path.exists():
        config.history_db_path.unlink()

    log_file = config.project_root / "data" / "logs" / "app.log"
    if log_file.exists():
        log_file.unlink()

    if remove_uploads:
        _delete_children(config.upload_dir)
