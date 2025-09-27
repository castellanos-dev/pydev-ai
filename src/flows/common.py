from __future__ import annotations
from typing import Dict, Any

from ..crews.summaries.file_summaries_crew import FileSummariesCrew
from ..crews.summaries.module_summaries_crew import ModuleSummariesCrew
from ..crews.summaries.output_format.summaries import MODULE_SUMMARIES_SCHEMA, FILE_SUMMARIES_SCHEMA

from .utils import load_json_output


def generate_file_summaries_from_chunk(item: str) -> Dict[str, Dict[str, Any]]:
    """
    Generate per-file summaries for a given chunk of code items.

    Returns a JSON object.
    """

    result = FileSummariesCrew().crew().kickoff(inputs={
        "code_chunk": item,
    })
    return load_json_output(result, FILE_SUMMARIES_SCHEMA)


def generate_module_summaries_from_file_summaries(file_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Generate per-module summaries using ONLY the per-file summaries as context.

    The provided mapping should associate file identifiers (paths) with their
    corresponding JSON summaries (objects). No real code is included.
    """

    result = ModuleSummariesCrew().crew().kickoff(inputs={
        "invidual_summaries": file_summaries,
    })
    module_summaries = load_json_output(result, MODULE_SUMMARIES_SCHEMA)

    return module_summaries
