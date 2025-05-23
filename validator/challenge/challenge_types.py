from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from textwrap import dedent
from datetime import datetime

import asyncio

'''
Helper that lets the validator set the scope of files they want to select for a challenge
'''
@dataclass
class IngestionHeuristics:
    min_files_to_consider_dir_for_problems: int
    min_file_content_len: int

'''
Helper class to handle and later embed a file for which we will ask a problem statement
'''
@dataclass
class File:
    path: Path
    contents: str


@dataclass
class EmbeddedFile:
    path: str
    contents: str
    embedding: List

    def __str__(self):
        return f"File: {self.path}, Length: {len(self.contents)}"

    def __repr__(self) -> str:
        return f"File: {self.path}, Length: {len(self.contents)}"
 

@dataclass
class FilePair:
    cosine_similarity: float
    files: List[EmbeddedFile]


@dataclass
class GeneratedCodegenProblem: 
    problem_statement: str
    dynamic_checklist: List[str]

@dataclass
class HyrdatedGeneratedCodegenProblem:
    problem_uuid: str
    prompt: str
    model: str
    problem_statement: str
    dynamic_checklist: List[str]
    context_files: List[str]

    def to_detailed_format(self) -> str:
        context_files_string = ""
        for i, file in enumerate(self.context_files):
            context_files_string += f"# File {i} used to solve the problem: {file}"
        return dedent(f"""
        Problem Statement: {self.problem_statement}
        Checklist of items to consider: {self.dynamic_checklist}
        {context_files_string}
        """)

class ChallengeTask:
    def __init__(self, node_id: int, task: asyncio.Task, timestamp: datetime, challenge: HyrdatedGeneratedCodegenProblem, miner_hotkey: str):
        self.node_id = node_id
        self.task = task
        self.timestamp = timestamp
        self.challenge = challenge
        self.miner_hotkey = miner_hotkey