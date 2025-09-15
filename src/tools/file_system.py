from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable, List, Union, Tuple


def write_empty_files(artifacts: Iterable[Union[str, Path]]) -> List[str]:
    """
    Create or truncate empty files for each path in ``artifacts``.

    - Ensures parent directories exist.
    - If the file already exists, it is truncated to empty.

    Parameters
    ----------
    artifacts:
        Iterable of file paths (absolute or relative), e.g., the values from an
        "artifacts" parameter in a flow.

    Returns
    -------
    List[str]
        Absolute paths of the created or truncated files.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    written_files: List[str] = []

    for item in artifacts:
        path = Path(item).expanduser()

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

        written_files.append(str(path.resolve()))

    return written_files



def delete_files(artifacts: Iterable[Union[str, Path]]) -> List[str]:
    """
    Delete files for each path in ``artifacts``.

    - Skips non-existent paths.
    - Raises if a path points to a directory (to avoid accidental directory deletion).

    Parameters
    ----------
    artifacts:
        Iterable of file paths (absolute or relative) to delete.

    Returns
    -------
    List[str]
        Absolute paths of the files that were deleted.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    deleted_files: List[str] = []

    for item in artifacts:
        path = Path(item).expanduser()

        # Avoid deleting directories
        if path.exists() and path.is_dir():
            raise IsADirectoryError(
                f"Path points to a directory, not a file: {path}"
            )

        if path.exists():
            path.unlink()
            deleted_files.append(str(path.resolve()))

    return deleted_files



def create_directories(artifacts: Iterable[Union[str, Path]]) -> List[str]:
    """
    Ensure directories exist for each path in ``artifacts``.

    - Creates the directory (and parents) if it does not exist.
    - Raises if a path exists and points to a file (not a directory).

    Parameters
    ----------
    artifacts:
        Iterable of directory paths (absolute or relative) to create/ensure.

    Returns
    -------
    List[str]
        Absolute paths of the directories that exist after this call.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    ensured_directories: List[str] = []

    for item in artifacts:
        path = Path(item).expanduser()

        # Avoid creating a directory at a path that is a file
        if path.exists() and path.is_file():
            raise NotADirectoryError(
                f"Path points to a file, not a directory: {path}"
            )

        # Create directory (and parents) if needed
        path.mkdir(parents=True, exist_ok=True)

        ensured_directories.append(str(path.resolve()))

    return ensured_directories


def delete_directories(artifacts: Iterable[Union[str, Path]]) -> List[str]:
    """
    Delete directories for each path in ``artifacts``.

    - Skips non-existent paths.
    - Raises if a path points to a file (to avoid accidental file deletion).
    - Removes directories recursively.

    Parameters
    ----------
    artifacts:
        Iterable of directory paths (absolute or relative) to delete.

    Returns
    -------
    List[str]
        Absolute paths of the directories that were deleted.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    deleted_directories: List[str] = []

    for item in artifacts:
        path = Path(item).expanduser()

        # Avoid deleting files
        if path.exists() and path.is_file():
            raise NotADirectoryError(
                f"Path points to a file, not a directory: {path}"
            )

        if path.exists():
            shutil.rmtree(path)
            deleted_directories.append(str(path.resolve()))

    return deleted_directories


def rename_files(artifacts: Iterable[Tuple[Union[str, Path], Union[str, Path]]]) -> List[str]:
    """
    Rename files for each ``(src, dst)`` pair in ``artifacts``.

    - Skips non-existent source paths.
    - Raises if a source points to a directory (to avoid renaming directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.

    Parameters
    ----------
    artifacts:
        Iterable of pairs ``(src, dst)`` where both entries are file paths
        (absolute or relative).

    Returns
    -------
    List[str]
        Absolute destination paths of the files that were renamed.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    renamed_files: List[str] = []

    for pair in artifacts:
        try:
            src_item, dst_item = pair  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError("Each artifact must be a pair (src, dst)") from exc

        src = Path(src_item).expanduser()
        dst = Path(dst_item).expanduser()

        # Avoid acting on directories
        if src.exists() and src.is_dir():
            raise IsADirectoryError(
                f"Source path points to a directory, not a file: {src}"
            )

        # Skip non-existent sources
        if not src.exists():
            continue

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

        renamed_files.append(str(dst.resolve()))

    return renamed_files


def move_files(artifacts: Iterable[Tuple[Union[str, Path], Union[str, Path]]]) -> List[str]:
    """
    Move files for each ``(src, dst)`` pair in ``artifacts``.

    - Skips non-existent source paths.
    - Raises if a source points to a directory (to avoid moving directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.
    - Uses ``shutil.move`` to support cross-filesystem moves.

    Parameters
    ----------
    artifacts:
        Iterable of pairs ``(src, dst)`` where both entries are file paths
        (absolute or relative).

    Returns
    -------
    List[str]
        Absolute destination paths of the files that were moved.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    moved_files: List[str] = []

    for pair in artifacts:
        try:
            src_item, dst_item = pair  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError("Each artifact must be a pair (src, dst)") from exc

        src = Path(src_item).expanduser()
        dst = Path(dst_item).expanduser()

        # Avoid acting on directories
        if src.exists() and src.is_dir():
            raise IsADirectoryError(
                f"Source path points to a directory, not a file: {src}"
            )

        # Skip non-existent sources
        if not src.exists():
            continue

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

        moved_files.append(str(dst.resolve()))

    return moved_files


def copy_files(artifacts: Iterable[Tuple[Union[str, Path], Union[str, Path]]]) -> List[str]:
    """
    Copy files for each ``(src, dst)`` pair in ``artifacts``.

    - Skips non-existent source paths.
    - Raises if a source points to a directory (to avoid copying directories).
    - Raises if destination exists (to avoid accidental overwrites).
    - Ensures destination parent directories exist.
    - Uses ``shutil.copy2`` to preserve metadata when possible.

    Parameters
    ----------
    artifacts:
        Iterable of pairs ``(src, dst)`` where both entries are file paths
        (absolute or relative).

    Returns
    -------
    List[str]
        Absolute destination paths of the files that were copied.
    """
    if artifacts is None:
        raise ValueError("artifacts cannot be None")

    copied_files: List[str] = []

    for pair in artifacts:
        try:
            src_item, dst_item = pair  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError("Each artifact must be a pair (src, dst)") from exc

        src = Path(src_item).expanduser()
        dst = Path(dst_item).expanduser()

        # Avoid acting on directories
        if src.exists() and src.is_dir():
            raise IsADirectoryError(
                f"Source path points to a directory, not a file: {src}"
            )

        # Skip non-existent sources
        if not src.exists():
            continue

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

        copied_files.append(str(dst.resolve()))

    return copied_files
