from __future__ import annotations

from pathlib import Path
import shutil
from typing import Union


def write_empty_file(path: Union[str, Path]) -> str | None:
    """
    Create or truncate empty file for ``path``.

    - Ensures parent directories exist.
    - If the file already exists, it is truncated to empty.

    Parameters
    ----------
    path:
        File path (absolute or relative), e.g., the values from an
        "artifacts" parameter in a flow.

    Returns
    -------
    str | None
        Absolute path of the created or truncated file.
    """
    if path is None:
        raise ValueError("path cannot be None")

    path = Path(path).expanduser()

    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid writing to a directory
    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Path points to a directory, not a file: {path}"
        )

    # Create or truncate to empty
    with path.open("w", encoding="utf-8"):
        pass

    return str(path.resolve())


def delete_file(path: Union[str, Path]) -> str | None:
    """
    Delete file for ``path``.

    - Skips non-existent paths.
    - Raises if a path points to a directory (to avoid accidental directory deletion).

    Parameters
    ----------
    path:
        File path (absolute or relative) to delete.

    Returns
    -------
    str | None
        Absolute path of the file that was deleted.
    """
    if path is None:
        raise ValueError("artifacts cannot be None")

    path = Path(path).expanduser()

    # Avoid deleting directories
    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Path points to a directory, not a file: {path}"
        )

    if path.exists():
        path.unlink()
        return str(path.resolve())

    return None


def create_directory(path: Union[str, Path]) -> str | None:
    """
    Ensure directory exist for ``path``.

    - Creates the directory (and parents) if it does not exist.
    - Raises if a path exists and points to a file (not a directory).

    Parameters
    ----------
    path:
        Directory path (absolute or relative) to create/ensure exist.

    Returns
    -------
    str | None
        Absolute path of the directory that was created.
    """
    if path is None:
        raise ValueError("path cannot be None")

    path = Path(path).expanduser()

    # Avoid creating a directory at a path that is a file
    if path.exists() and path.is_file():
        raise NotADirectoryError(
            f"Path points to a file, not a directory: {path}"
        )

    # Create directory (and parents) if needed
    path.mkdir(parents=True, exist_ok=True)

    return str(path.resolve())


def delete_directory(path: Union[str, Path]) -> str | None:
    """
    Delete directory for ``path``.

    - Skips non-existent paths.
    - Raises if a path points to a file (to avoid accidental file deletion).
    - Removes directories recursively.

    Parameters
    ----------
    path:
        Directory path (absolute or relative) to delete exist.

    Returns
    -------
    str | None
        Absolute path of the directory that was deleted.
    """
    if path is None:
        raise ValueError("artifacts cannot be None")

    path = Path(path).expanduser()

    # Avoid deleting files
    if path.exists() and path.is_file():
        raise NotADirectoryError(
            f"Path points to a file, not a directory: {path}"
        )

    if path.exists():
        shutil.rmtree(path)
        return str(path.resolve())

    return None


def rename_file(src: str, dst: str) -> str | None:
    """
    Rename file for ``(src, dst)`` pair.

    - Skips non-existent source path.
    - Raises if a source points to a directory (to avoid renaming directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.

    Parameters
    ----------
    src:
        Source file path (absolute or relative)
    dst:
        Destination file path (absolute or relative)
        (absolute or relative).

    Returns
    -------
    str | None
        Absolute destination path of the file that was renamed.
    """
    if src is None or dst is None:
        raise ValueError("src and dst cannot be None")

    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    # Avoid acting on directories
    if src.exists() and src.is_dir():
        raise IsADirectoryError(
            f"Source path points to a directory, not a file: {src}"
        )

    # Skip non-existent sources
    if not src.exists():
        return None

    if dst.exists() and dst.is_dir():
        raise IsADirectoryError(
            f"Destination path points to a directory, not a file: {dst}"
        )

    # Do not overwrite existing files
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Perform rename (will move across directories on the same filesystem)
    src.rename(dst)

    return str(dst.resolve())


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> str | None:
    """
    Move file for ``(src, dst)`` pair.

    - Skips non-existent source paths.
    - Raises if a source points to a directory (to avoid moving directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.
    - Uses ``shutil.move`` to support cross-filesystem moves.

    Parameters
    ----------
    src:
        Source file path (absolute or relative)
    dst:
        Destination file path (absolute or relative)
        (absolute or relative).

    Returns
    -------
    str | None
        Absolute destination path of the file that was moved.
    """
    if src is None or dst is None:
        raise ValueError("src and dst cannot be None")

    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    # Avoid acting on directories
    if src.exists() and src.is_dir():
        raise IsADirectoryError(
            f"Source path points to a directory, not a file: {src}"
        )

    # Skip non-existent sources
    if not src.exists():
        return None

    if dst.exists() and dst.is_dir():
        raise IsADirectoryError(
            f"Destination path points to a directory, not a file: {dst}"
        )

    # Do not overwrite existing files
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Perform move (supports cross-filesystem moves)
    shutil.move(str(src), str(dst))

    return str(dst.resolve())


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> str | None:
    """
    Copy file for ``(src, dst)`` pair.

    - Skips non-existent source paths.
    - Raises if a source points to a directory (to avoid copying directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.
    - Uses ``shutil.copy2`` to preserve metadata when possible.

    Parameters
    ----------
    src:
        Source file path (absolute or relative)
    dst:
        Destination file path (absolute or relative)
        (absolute or relative).

    Returns
    -------
    str | None
        Absolute destination path of the file that was copied.
    """
    if src is None or dst is None:
        raise ValueError("src and dst cannot be None")

    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    # Avoid acting on directories
    if src.exists() and src.is_dir():
        raise IsADirectoryError(
            f"Source path points to a directory, not a file: {src}"
        )

    # Skip non-existent sources
    if not src.exists():
        return None

    if dst.exists() and dst.is_dir():
        raise IsADirectoryError(
            f"Destination path points to a directory, not a file: {dst}"
        )

    # Do not overwrite existing files
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Perform copy (preserves metadata when possible)
    shutil.copy2(str(src), str(dst))

    return str(dst.resolve())
