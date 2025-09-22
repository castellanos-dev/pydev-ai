from __future__ import annotations
from typing import Dict, List

from ..crews.summaries.file_summaries_crew import FileSummariesCrew
from ..crews.summaries.module_summaries_crew import ModuleSummariesCrew
from ..crews.summaries.output_format.summaries import SUMMARIES_SCHEMA

from .utils import load_json_output, sanitize_generated_content


def generate_file_summaries_from_chunk(chunk: List[Dict[str, str]], summaries_dir: str) -> Dict[str, str]:
    """
    Generate per-file summaries for a given chunk of code items.

    The input is a list of dicts with keys: {"path": str, "content": str}.
    Returns a mapping of summary paths (relative to summaries root) to sanitized Markdown.
    """

    summaries: Dict[str, str] = {}

    for item in chunk:
        result = FileSummariesCrew().crew().kickoff(inputs={
            "code_chunk": item,
            "summaries_dir": summaries_dir,
        })
        file_summaries = load_json_output(result, SUMMARIES_SCHEMA, 0)

        for entry in file_summaries:
            summaries[entry["path"]] = sanitize_generated_content(entry["content"])

    return summaries


def generate_module_summaries_from_file_summaries(file_summaries: Dict[str, str], summaries_dir: str) -> Dict[str, str]:
    """
    Generate per-module summaries using ONLY the per-file summaries as context.

    The provided mapping should associate file identifiers (paths) with their
    corresponding Markdown summaries. No real code is included.
    """

    result = ModuleSummariesCrew().crew().kickoff(inputs={
        "invidual_summaries": file_summaries,
        "summaries_dir": summaries_dir,
    })
    module_summaries = load_json_output(result, SUMMARIES_SCHEMA, 0)

    summaries: Dict[str, str] = {}
    for entry in module_summaries:
        summaries[entry["path"]] = sanitize_generated_content(entry["content"])
    return summaries
