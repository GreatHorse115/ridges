'''
For now Ridges has a codegen challenge. Very shortly we will be introducing more challenge types
'''

import random
from typing import List, Final, Dict, Optional
from textwrap import dedent
import pickle
import os
import json
from pathlib import Path
from dataclasses import asdict
import uuid

import tiktoken
import openai
from jinja2 import Template
import numpy as np
from fiber.logging_utils import get_logger

from validator.challenge.challenge_types import HyrdatedGeneratedCodegenProblem, GeneratedCodegenProblem, EmbeddedFile, FilePair
from validator.config import (
    OPENAI_API_KEY, PREFERRED_OPENAI_MODEL,
    MIN_FILE_CONTENT_LEN_CHARS, MIN_FILES_IN_DIR_TO_GENERATE_PROBLEM
)

logger = get_logger(__name__)

PROBLEM_STATEMENT_TEMPLATE: Final[Template] = Template(
    dedent("""
    You are a skilled software engineering assistant. You will be provided with multiple files as context. Each file will contain portions of code, documentation, or relevant information about a software system. Your task is to come up with a specific software engineering problem that requires a solution to involve at least two of these files. You will generate a list of these problems, in the generated_problems array response.

    Further, once you have a problem statement, generate a checklist of points to consider and things that should be present in the solution (for example, are the correct Github API calls made if its a function that interfaces with the api). Generate several of these into dynamic_checklist field.
    Some additional guidelines are:
    - Do not output anything other than software engineering problem
    - The problem description should be very detailed and meticulous. It should contain sufficient context such that someone equipped with the codebase and your problem statement will have enough information to implement
    - The problem should be solvable by an autonomous SWE which can do things like installing PyPi packages, but cannot do things like make cloud provider accounts and register for other services manually.
    - The problem should not be overly difficult to implement, and should be fairly easy and not take too many LLM calls. 
    - Do not disclose which files would need to be modified to solve the problem.

    Here are the files:
    {% for file in files %}
    Filename: {{ file.path }}
    ```python3
    {{ file.contents }}
    ```
    {% endfor %}
    ```
    """)
)

# Functions related to selecting files to generate a synthetic coding problem
def walk_repository(repo_path: Path) -> Dict:
    """
    Picks files to generate problem statements from
    """
    repo_map = {}

    for root, dirs, files in os.walk(str(repo_path)):
        # Convert absolute path to relative path from repo root
        rel_path = os.path.relpath(root, str(repo_path))
        if rel_path == '.':
            rel_path = ''

        # Filter out common files/directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        files = [f for f in files if not f.startswith(('.', '__')) and not f.endswith(('.pyc', '.pyo')) and f.endswith('.py')]

        # Add to map
        repo_map[rel_path] = {
            'dirs': dirs,
            'files': files
        }

    return repo_map

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def highest_cosine_filepair_selector(file_pairs: List[FilePair]) -> FilePair:
    selected_file_pair = sorted(
        file_pairs,
        key=lambda x: float(x.cosine_similarity),
        reverse=True
    )[random.randint(1, 10)]

    return selected_file_pair


def evaluate_for_context(
    dir_path, 
    repo_structure,
    openai_client: openai.Client
):
    def _retrieve_files_in_dir():
        # Get all files in the current directory
        files = []
        for file_name in repo_structure['files']:
            path = os.path.join(dir_path, file_name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    contents = f.read()
                files.append(
                    {
                        'path': path,
                        'contents': contents
                    }
                )
            except (UnicodeDecodeError, IOError):
                logger.exception(f"Warning: Could not read file {path}")
                continue

        return files

    def _embed_code(raw_codes: List[str]) -> List[List[float]]:
        encoding = tiktoken.get_encoding('cl100k_base')
        truncated_inputs = [encoding.encode(json.dumps(code))[:8191] for code in raw_codes]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=truncated_inputs
        )
        # Return list of embedding vectors
        return [data.embedding for data in response.data]

    def _find_most_similar_files(embedded_files: List[EmbeddedFile]) -> Optional[FilePair]:
        max_similarity = -1
        most_similar_pair = None

        # Compare each pair of files
        for i in range(len(embedded_files)):
            for j in range(i + 1, len(embedded_files)):
                similarity = cosine_similarity(
                    embedded_files[i].embedding,
                    embedded_files[j].embedding
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = [embedded_files[i], embedded_files[j]]

        if most_similar_pair is None:
            return None

        return FilePair(
            cosine_similarity=max_similarity,
            files=most_similar_pair
        )

    if len(repo_structure['files']) >= MIN_FILES_IN_DIR_TO_GENERATE_PROBLEM:
        files = _retrieve_files_in_dir()
        embeddings = _embed_code(list(map(lambda file: file['contents'], files)))
        embedded_files = [
            EmbeddedFile(
                path=file['path'],
                contents=file['contents'],
                embedding=embeddings[i]
            )
            for i, file in enumerate(files)
            if len(file['contents']) > MIN_FILE_CONTENT_LEN_CHARS
        ]

        most_similar_files = _find_most_similar_files(embedded_files)

        return most_similar_files
    else:
        return []

def save_filepairs_to_cache(filepairs: List[FilePair], cache_path: str) -> None:
    """Save list of FilePairs to local cache."""
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert FilePairs to serializable format
    serializable_pairs = [
        {
            'cosine_similarity': pair.cosine_similarity,
            'files': [asdict(f) for f in pair.files]
        }
        for pair in filepairs
    ]
    
    with open(cache_path, 'wb') as f:
        pickle.dump(serializable_pairs, f)

def load_filepairs_from_cache(cache_path: str) -> List[FilePair]:
    """Load FilePairs from cache, returns empty list if cache doesn't exist."""
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct FilePairs from serialized data
        return [
            FilePair(
                cosine_similarity=pair['cosine_similarity'],
                files=[EmbeddedFile(**f) for f in pair['files']]
            )
            for pair in data
        ]
    except (FileNotFoundError, EOFError):
        return []

def get_all_filepairs(
    local_repo_path: Path, 
    openai_client: openai.Client,
    refresh: bool = False,
) -> List[FilePair]:
    cache_path = f".cache/{local_repo_path}"

    filepairs_from_cache = load_filepairs_from_cache(cache_path=cache_path)

    if filepairs_from_cache and len(filepairs_from_cache) > 0 and refresh == False:
        logger.info("Retrieved relevant filepairs for repo from cache")
        return filepairs_from_cache
    
    logger.error("filepairs not found in cache")
    
    repo_structure = walk_repository(local_repo_path)

    file_pairs = []
    for dir_path, contents in repo_structure.items():
        full_path = os.path.join(local_repo_path, dir_path) if dir_path else local_repo_path
        if contents['files']:
            file_pairs.append(evaluate_for_context(full_path, contents, openai_client))

    valid_pairs = [pair for pair in file_pairs if pair and isinstance(pair, FilePair)]
    if not valid_pairs:
        raise ValueError("No valid file pairs found in the repository. Ensure there are directories with 5+ Python files.")
    
    # Save the filepairs to cache
    save_filepairs_to_cache(
        filepairs=valid_pairs,
        cache_path=cache_path
    )

    return valid_pairs

async def create_next_codegen_challenge(
    openai_client: openai.Client
) -> HyrdatedGeneratedCodegenProblem:
    ''' 
    Creates a Codegen challenge task. 
    To do this, validators clone a repo from the available repo set, find a random set of filepairs, 
    and generate a problem + checklist of things the problem must solve. 
    This is then returned to miners that specialize in codegen

    Args:
        validator_hotkey (str): The validator's hotkey.

    Returns:
        GeneratedProblemStatement or None if error
    '''

    # Select a supported repo at random
    repo_path: Path = None

    file_pairs = get_all_filepairs(local_repo_path=repo_path, openai_client=openai_client)
    selected_pair = random.choice(file_pairs)

    # Generate problem statement + dynamic checklist of issues to be solved
    prompt_with_filepair_context = PROBLEM_STATEMENT_TEMPLATE.render(
        dict(selected_pair.files)
    )

    completion = openai_client.beta.chat.completions.parse(
        model=PREFERRED_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt_with_filepair_context},
            {"role": "user", "content": f"Generate the list of problem statements. Generate exactly 1 problem statement statements, no more and no less"},
        ],
        response_format=GeneratedCodegenProblem,
    )

    generated_problem: GeneratedCodegenProblem = completion.choices[0]
    problem_id = str(uuid.uuid4())

    return HyrdatedGeneratedCodegenProblem(
        problem_uuid=problem_id,
        prompt=prompt_with_filepair_context,
        model=PREFERRED_OPENAI_MODEL,
        problem_statement=generated_problem.problem_statement,
        dynamic_checklist=generated_problem.dynamic_checklist,
        context_files=[file.contents for file in selected_pair.files]
    )