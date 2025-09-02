from __future__ import annotations
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import subprocess
from .. import settings
import os
import re
import json
from pathlib import Path


class RunPytestArgs(BaseModel):
    repo: str = Field(..., description="Repo root path")
    k: str | None = Field(None, description="Pytest -k expression")


class RunPytestTool(BaseTool):
    name: str = "run_pytest"
    description: str = "Run pytest and return report output with exit code."
    args_schema: Type[BaseModel] = RunPytestArgs

    def _run(self, repo: str, k: str | None = None) -> str:
        # Build command list for better security and reliability
        cmd = ["python", "-m", "pytest", "-q"]
        if k:
            cmd.extend(["-k", k])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=settings.PYTEST_TIMEOUT,
            )

            # Include exit code in output for agent logic
            status = "PASSED" if result.returncode == 0 else "FAILED"
            output = f"[pytest] Status: {status} (exit code: {result.returncode})\n"
            output += result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"

            return output

        except subprocess.TimeoutExpired:
            return (
                f"[run_pytest] Error: Timeout after {settings.PYTEST_TIMEOUT} seconds"
            )
        except Exception as e:
            return f"[run_pytest] Error: {e}"


class RunUnittestArgs(BaseModel):
    repo: str = Field(..., description="Repo root path")
    start_dir: str = Field("tests", description="Start directory for test discovery")
    pattern: str = Field("test*.py", description="Filename pattern for test discovery")
    top_level_dir: str | None = Field(
        None, description="Top-level project directory for imports (optional)"
    )


class RunUnittestTool(BaseTool):
    name: str = "run_unittest"
    description: str = "Run unittest discovery and return report output with exit code."
    args_schema: Type[BaseModel] = RunUnittestArgs

    def _run(
        self,
        repo: str,
        start_dir: str = "tests",
        pattern: str = "test*.py",
        top_level_dir: str | None = None,
    ) -> str:
        # Build command list for better security and reliability
        cmd = [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            start_dir,
            "-p",
            pattern,
        ]
        if top_level_dir:
            cmd.extend(["-t", top_level_dir])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=settings.PYTEST_TIMEOUT,
            )

            # Include exit code in output for agent logic
            status = "PASSED" if result.returncode == 0 else "FAILED"
            output = f"[unittest] Status: {status} (exit code: {result.returncode})\n"
            output += result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"

            return output

        except subprocess.TimeoutExpired:
            return (
                f"[run_unittest] Error: Timeout after {settings.PYTEST_TIMEOUT} seconds"
            )
        except Exception as e:
            return f"[run_unittest] Error: {e}"


class DetectTestPresenceArgs(BaseModel):
    repo: str = Field(..., description="Repo root path")


class DetectTestPresenceTool(BaseTool):
    name: str = "detect_test_presence"
    description: str = (
        "Quickly detect whether a repo appears to use pytest and/or unittest. "
        'Returns a minimal JSON like {"pytest": bool, "unittest": bool}.'
    )
    args_schema: Type[BaseModel] = DetectTestPresenceArgs

    def _run(self, repo: str) -> str:
        root = Path(repo)
        if not root.exists() or not root.is_dir():
            return json.dumps({"pytest": False, "unittest": False})

        # Initial states
        uses_pytest = False
        uses_unittest = False

        def safe_read_text(path: Path, max_bytes: int = 100_000) -> str:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fp:
                    return fp.read(max_bytes)
            except Exception:
                return ""

        # Precompile patterns for faster repeated checks
        pytest_import_pattern = re.compile(r"\b(?:import|from)\s+pytest\b")
        unittest_import_pattern = re.compile(
            r"\b(?:import|from)\s+unittest\b|\bunittest\.TestCase\b"
        )

        scanned = 0
        max_scan = 800

        def process_file(path: Path) -> None:
            """Process a single file for framework detection"""
            nonlocal uses_pytest, uses_unittest, scanned

            if scanned >= max_scan or (uses_pytest and uses_unittest):
                return

            lower_name = path.name.lower()

            # File-name based hints (conftest.py is a strong pytest indicator)
            if lower_name == "conftest.py":
                uses_pytest = True
                return  # No need to scan conftest.py content

            scanned += 1

            # Only read file content if we still need to detect something
            if not (uses_pytest and uses_unittest):
                text = safe_read_text(path, max_bytes=20_000)

                if not uses_pytest and pytest_import_pattern.search(text):
                    uses_pytest = True
                if not uses_unittest and unittest_import_pattern.search(text):
                    uses_unittest = True

        # Phase 1: Process high-priority files first using glob (faster for specific patterns)
        priority_patterns = [
            "**/conftest.py",  # Strong pytest indicator
            "**/test_*.py",  # Test files
            "**/tests/**/*.py",  # Files in test directories
            "**/test/**/*.py",  # Files in test directories (alternative naming)
        ]

        for pattern in priority_patterns:
            if (uses_pytest and uses_unittest) or scanned >= max_scan:
                break

            for path in root.glob(pattern):
                process_file(path)
                if (uses_pytest and uses_unittest) or scanned >= max_scan:
                    break

        # Phase 2: If we haven't found both frameworks yet, scan remaining Python files
        if not (uses_pytest and uses_unittest) and scanned < max_scan:
            # Use glob to get all Python files, but limit scanning
            processed_paths = set()  # Avoid processing same file twice

            for path in root.glob("**/*.py"):
                if (uses_pytest and uses_unittest) or scanned >= max_scan:
                    break

                if (
                    path in processed_paths
                    or path.name.lower().startswith("test_")  # Already processed
                    or path.name.lower() == "conftest.py"  # Already processed
                    or "test" in path.parts
                ):  # Already processed
                    continue

                processed_paths.add(path)
                process_file(path)

        return json.dumps(
            {"pytest": bool(uses_pytest), "unittest": bool(uses_unittest)}
        )
