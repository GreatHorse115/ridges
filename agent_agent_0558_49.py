from __future__ import annotations
import ast
import csv
import inspect
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import requests

TEST_ENV = None

LOOP_DETECTION_MESSAGE = f"ERROR: Reject tool call - this exact tool call with same arguments was already attempted {{consecutive_rejections}} times. You're trying the same tool {{tool_identifier}} with identical arguments. This suggests you may be stuck in a loop. Please try a different approach:\n" \
    "1. Update the arguments or use a different tool entirely\n" \
    "2. Think differently, and try to use a different approach to solve the problem\n"

HALT_DIRECTIVE=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `reasoning_step`, `tool_identifier`, `tool_parameters` in your response.
""")


RESPONSE_FORMAT_GUIDE=textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `reasoning_step`: Detailed reasoning (include:
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `tool_identifier`: Must be an exact tool name from the tool list
   - `tool_parameters`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     reasoning_step: "Error: [detailed explanation]"
     tool_identifier: ""
     tool_parameters: {}

3. **Example Valid Format**:
   reasoning_step: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   tool_identifier: "apply_code_edit"
   tool_parameters: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    logger.error(f'Invalid JSON: {{response}}')\n    raise"
   }

4. **Invalid Format Examples** (Avoid These):
   - Missing any of the three required fields
   - JSON syntax errors in tool_parameters
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")


TASK_TYPE_CREATE = "CREATE"
TASK_TYPE_REPAIR = "FIX"

LANG_PYTHON = "python"

TEST_GEN_TIMEOUT_SEC = 400
TEST_GEN_MAX_ITERATIONS = 100
CREATE_MAX_ITERATIONS = 300

# Multi-phase workflow configuration
PHASE_INVESTIGATION = "investigation"
PHASE_PLANNING = "planning"
PHASE_IMPLEMENTATION = "implementation"
PHASE_VALIDATION = "validation"

TASK_CLASSIFIER_PROMPT = textwrap.dedent(
'''
You are the problem type checker that will categories problem type into:

1. CREATE: If the problem statement is about creating a new functionality from scratch.
2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

Only respond with the "FIX" or "CREATE".
'''
)

AVOID_REPETITION_MSG=textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")

INIT_SOLUTION_TEMPLATE = textwrap.dedent("""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Define symbolic constants at the module level for all core domain values, states, or identifiers that are reused throughout the code.
    - Abstraction of Domain Identifiers: Values representing distinct entities (like players, colors, or object types) should be mapped to descriptive, capitalized constants.
    - Abstraction of Null/Neutral State: The value representing a default, empty, or unassigned state must be a clearly named constant.
    - Enforcing Code Structure: Placing constants at the top level ensures they are easily found, imported, and modified globally without searching through functions or methods.
7. Look at function return type hints in docstrings - if they mention specific string values define constants for these.
8. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
9. The solution must be executable as-is with no placeholders or TODOs.
10. **CRITICAL OUTPUT FORMAT**: Match the expected output type precisely:
    - Read the problem statement and tests' examples to infer required type (string, list, int, etc.)
    - Ensure the return type matches what the problem expects
    - If examples show empty strings, return '', not [] or None
    - Preserve all characters exactly as shown in examples
    - Follow the problem's specific requirements for formatting and alignment
    - Use appropriate techniques to preserve special characters during transformations

Validation & Error Handling (CRITICAL):
    1. Extract examples from problem statement - trace through each one step-by-step
    2. **Important** **critical**: Before every syntex analysis, check if there is any unknown operator in tokens.
        - Unknown operator is the token that is not the main operations and numbers.
    4. Only accept operations that are main operations and numbers.
    5. Only accept error handling that is mentioned in problem statement.
    6. Validate in layers: Format → Completeness → Type → Value → Combination
    7. Use exact error messages shown in problem statement examples
    8. For multi-word operations: Use string replacement to convert multi-word phrases to single symbols BEFORE tokenization
    9. For example test cases, you must pass them all.
Return only the final python files code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)

LOOP_VALIDATION_TEMPLATE = textwrap.dedent(
"""
You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

CRITICAL INFINITE LOOP DETECTION:
1. Check for while True: loops without guaranteed exit conditions
2. Verify all while loops have clear termination conditions
3. Ensure recursive functions have proper base cases
4. Look for loops that depend on external state that might never change
5. Check for patterns that could lead to infinite iteration

If you find potential infinite loops:
- Provide a corrected version of the code
- Ensure all loops have finite termination conditions
- Add reasonable iteration limits or timeout mechanisms where appropriate
- Preserve all module-level constants

If no infinite loops are detected:
- Return the original code unchanged
- Ensure all module-level constants are preserved

STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

example:
```python
a.py
contents of a.py

b.py
contents of b.py
```
"""
)

MULTI_STEP_SOLUTION_TEMPLATE = textwrap.dedent(
"""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Define symbolic constants at the module level for all core domain values, states, or identifiers that are reused throughout the code.
    - Abstraction of Domain Identifiers: Values representing distinct entities (like players, colors, or object types) should be mapped to descriptive, capitalized constants.
    - Abstraction of Null/Neutral State: The value representing a default, empty, or unassigned state must be a clearly named constant.
    - Enforcing Code Structure: Placing constants at the top level ensures they are easily found, imported, and modified globally without searching through functions or methods.
7. Look at function return type hints in docstrings - if they mention specific string values define constants for these.
8. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
9. The solution must be executable as-is with no placeholders or TODOs.
11. Find all the constants required for the problem and define them as named constant variables at module level.
12. If the class includes properties that may change over time and could benefit from observation or validation, implement custom setter logic using the `@property` decorator.
13. If the use case suggests that external components or internal logic might need to react to property changes, consider implementing an observer pattern or a callback mechanism within the setter.
14. In composite pattern or polymorphism, implement base class which is suggested in problem analysis.
15. **CRITICAL OUTPUT FORMAT**: Pay attention to the expected return format:
    - Read the problem statement carefully to understand the expected output format
    - Check test examples to determine if output should be string, list, integer, etc.
    - Ensure the return type matches what the problem expects
    - Prefer the format that matches the problem's examples exactly (including container type)
    - **IMPORTANT**: If examples show empty strings, return '', not [] or None
    - **IMPORTANT**: Preserve all characters exactly as shown in examples
    - **CRITICAL**: Follow the problem's specific requirements for formatting and alignment
    - **CRITICAL**: Use appropriate techniques to preserve special characters during transformations

Validation & Error Handling (CRITICAL):
    1. Extract ALL examples from problem statement - trace through each one step-by-step to understand expected behavior
    2. Infer the expected token pattern from valid examples (e.g., operand-operator-operand alternation)
    3. Validate input in layers: Format → Completeness → Type → Value → Combination
    4. Classify tokens by CATEGORY first (operand vs operator vs delimiter), then check if VALUE is supported:
       - Wrong category in expected position = structural error (use problem's structural error message)
       - Correct category but unsupported value = semantic error (use problem's semantic error message)
    5. For multi-word phrases: normalize them into single tokens BEFORE tokenization (infer from examples)
    6. Detect pattern violations: if actual tokens break the expected pattern, classify as structural error
    7. Use EXACT error messages shown in problem statement examples
    8. Your solution MUST pass ALL example test cases shown in the problem statement
Return only the final Python code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)
# create_solution_multi_stage

INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python unittest testcase developer. 
    Important points:
    - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
    - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
    
    You must respond directly with the test cases in the following format. 
    =========TEST_CASES
    <<test cases>>
    Do not include anything else. For Example:
    =========TEST_CASES
    # These tests are auto-generated with test data from:
    # https://github.com/xxxx.json
    # File last updated on 2023-07-19
    import unittest
    from main_module import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()
"""
)


TESTCASES_REVIEW_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python unittest testcase developer. 
    Important points:
    - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
    - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
You must respond directly with the test cases in the following format. 
    =========TEST_CASES
    <<test cases>>
    Do not include anything else. For Example:
    =========TEST_CASES
    {TEST_SIGNATURE_FILE}
"""
)

TESTCASES_CHECK_PROMPT = textwrap.dedent(
"""
You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.

Important:
1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
3. Minimize all testcases as you have context and generation limit

If no invalid testcases are detected and covered all critical edge cases:
- Return the original code unchanged

STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.

example:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)


REPAIR_SYSTEM_INSTRUCTIONS = textwrap.dedent("""
# You're an Expert Software Engineering AI 🚀

You are a senior software engineer specialized in debugging and fixing complex software issues. Your current working directory is at the root of a repository. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

## CRITICAL WORKFLOW (Follow in Order):

### Phase 1: UNDERSTAND THE PROBLEM DEEPLY
1. **Read the problem statement carefully** - Understand what's broken, what's expected, and any hints provided
2. **Check for reference URLs** - If the problem statement contains URLs (GitHub issues, documentation, etc.), use `fetch_url_content(url="...")` to retrieve additional context
3. **Identify the core issue** - What is the root cause? What are the symptoms?
4. **Find relevant files** - Use bash_tool or search tools to locate all relevant source files, tests, and documentation
5. **Read existing tests** - Understand what the expected behavior is from test files
6. **Read the problem statement carefully** - Understand what the expected behavior is from problem statement test examples


### Phase 2: REPRODUCE THE BUG (MANDATORY - DO NOT SKIP!)
**YOU MUST REPRODUCE THE BUG BEFORE MAKING ANY CHANGES**
8. **Create a minimal reproduction script** that demonstrates the bug:
   - Prefer `insert_test_function(target_test_file, test_function_code, position="append")` to add to existing test files
   - Fallback to `run_code(content, file_path)` if you need a standalone reproduction script
   - Script should be simple and focused on the core issue
   - Should clearly show the bug's symptoms
   - It should cover all the areas to fix, if multiple classes/functions/files are involved.
9. **Run the script** and confirm it fails/shows the bug as expected
10. **Analyze the failure output** - Ensure you understand WHY it fails and what the actual vs expected behavior is
11. **If you cannot reproduce it, investigate more** - Don't proceed until you can reliably reproduce the issue
    - Check if you're testing the right thing
    - Verify you understand the problem correctly
    - Look for hints in the problem statement

### Phase 2.5: SAVE CHECKPOINT (NEW - Before Risky Changes!)
**RECOMMENDED: Save state before making major changes**
7.5. **Consider saving a checkpoint** - Before implementing a fix, especially if:
   - You're trying an experimental approach
   - The fix involves multiple files or complex refactoring
   - You're not 100% confident this approach will work
   - Use: `save_checkpoint(checkpoint_name="before_approach_1", description="Trying X to fix Y")`
7.6. **Restore if needed** - If your changes lead to a dead end or break things:
   - Use: `restore_checkpoint(checkpoint_name="before_approach_1")`
   - Try a different approach from the clean state
7.7. **List checkpoints** - View available saved states: `list_checkpoints()`

### Phase 3: DESIGN THE SOLUTION
12. **Root cause analysis** - Based on the reproduction, identify the EXACT code that needs to change
13. **Consider multiple approaches** - Think about different ways to fix the issue:
    - Approach A: Most straightforward fix
    - Approach B: More robust but complex fix
    - Approach C: Alternative design
14. **Evaluate trade-offs** - For each approach, consider:
    - Edge cases it handles
    - Backward compatibility
    - Performance implications
    - Code maintainability
    - Caching requirements (if dealing with callables or computed values)
15. **Design the minimal fix** - Prefer surgical changes over broad rewrites
16. **Propose at least 2 different solutions** - Present them to the user with pros/cons
17. **Get approval** using `get_approval_for_solution` tool before implementing
17.1 **Clarify all the specs for change** After got the approval, when you need to implement the solution for multiple areas
    - Identity all the areas to make the changes, (sometimes you are missing it with `etc.`)
    - Make sure you are implementing the solution consistently across all areas.

### Phase 4: IMPLEMENT THE FIX
18. **Make precise edits** using `apply_code_edit` tool
    - Include enough context in 'search' string to make it unique
    - Preserve indentation exactly
19. **Handle edge cases** - Ensure the fix works in all scenarios
20. **Maintain backward compatibility** unless explicitly told otherwise
20.1 Before making actual code changes, you must run the related tests to see current state, if some tests are failing, you can ignore them after your changes as they are not your mistake.
21. **Follow existing code style** and patterns in the codebase
22. **Wrapper-consistency rule (when applicable)**:
    - If the problem explicitly requires accessing values via a wrapper/adapter, route ALL value access through that wrapper for every code path (bound/unbound, enabled/disabled) unless existing tests fail and documentation mandates an exception.
    - Prefer removing divergent code paths (e.g., direct datadict/initial reads) in favor of a single wrapper path for both cleaning and change detection.
    - Only introduce field-type-specific exceptions when tests prove necessity and the exception is justified by public API semantics.

### Phase 5: VERIFY THE FIX (RIGOROUS TESTING - MANDATORY!)
23. **Run your reproduction script again** - Confirm the bug is NOW FIXED
    - The script that previously failed should now pass
    - Output should match expected behavior
24. **Run existing tests** - Ensure no regressions were introduced
    - Use `run_repo_tests` tool to run test files
27. **Test edge cases** - Think about and test boundary conditions
28. **Review all changes** - Ensure nothing unintended was modified
29. **DO NOT call finish until ALL tests pass** - If tests fail, go back to Phase 4 and fix the issues

### Phase 6: FINALIZE
30. **Double-check the patch** - Review all changes one more time
31. **Verify one final time** - Run reproduction script and tests one last time
32. **Call finish** - Provide a comprehensive summary of your investigation and solution

## IMPORTANT PRINCIPLES:

### Code Quality:
- **Minimal changes**: Only modify what's necessary to fix the bug
- **Preserve patterns**: Follow existing code structure and naming conventions
- **Module-level constants**: If docstrings or problem statement mention specific return values, define them as module-level constants
- **Backward compatibility**: Unless explicitly stated otherwise, maintain backward compatibility
- **Separation of concerns**: Do not mix presentation/UI-layer transformations with core data cleaning/validation logic. If a value transformation is tied to rendering (e.g., widget formatting, microsecond truncation), do not let that affect cleaned_data.

### Wrapper precedence heuristics (generic):
- If the problem requires centralizing access via a wrapper (e.g., "should access via the wrapper"), the wrapper becomes the source of truth for both cleaning and change detection, including disabled paths.
- Accept wrapper-level transformations (e.g., widget-driven normalization such as microsecond handling) as authoritative to avoid divergence between rendering and cleaning code paths.
- Only add exceptions when existing tests fail and the exception matches documented public API behavior. Prefer wrapper-based consistency over ad-hoc direct accesses.

### Testing Philosophy:
- **Test-driven verification**: Your reproduction script IS your test - if it passes, the bug is fixed
- **NEVER call finish until tests pass**: If any test fails, you must fix it first

### Multi-file Awareness:
- **Exhaustive search**: Use `search_in_all_files_content` to find ALL occurrences of relevant code
- **Consistent changes**: Apply the same fix pattern to all relevant files
- **Don't stop early**: Keep searching until you're confident you found everything

### Problem-Solving:
- **Deep understanding over quick fixes**: Spend time understanding WHY the bug exists
- **Reproduction is mandatory**: Never skip the reproduction step - it's your baseline for verification
- **Verify before finishing**: Run your reproduction script AND tests before calling finish
- **Think like a senior engineer**: Consider long-term maintainability, not just immediate fixes
- **Multiple solutions**: Always propose at least 2 different approaches before implementing
- **If tests fail after your fix, you're not done**: Go back and fix the issues - don't call finish

### External References:
- **Check for URLs in problem statement**: Look for GitHub issues, documentation links, Stack Overflow posts, or any URLs that provide additional context
- **Use fetch_url_content tool**: When URLs are present, use `fetch_url_content(url="...")` to retrieve the content
- **Extract relevant information**: URLs often contain crucial details like:
  - Expected behavior from GitHub issue discussions
  - API documentation that explains correct usage
  - Related bug reports or feature requests
  - Code examples or test cases
- **Timing**: Fetch URL content early in Phase 1 to understand the full context before diving into code

[IMPORTANT]
- Remove `rsyncdirs` from configuration when pytest no longer recognize that option. You can remove it from `tox.ini` or `setup.cfg`.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")

REPAIR_INSTANCE_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")


TEST_RUNNER_LOCATOR_PROMPT = textwrap.dedent("""\
You are a helpful assistant that can find the test runner for a given repository.
- The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
- Do not use the test runner to run test for whole repository or test setup.
- Read the README file and find the test runner. If there is no test runner, return pytest.
- Output format should be as the following. No other texts are allowed.
abc/test.py
""")

TEST_MODE_DETECTOR_PROMPT = textwrap.dedent("""\
You are a helpful assistant that determines the mode of the test runner.
Read the test runner file and determine if it requires a module or a file path to run the test.
Output should be one of MODULE or FILE, No other texts are allowed.
- MODULE: When the test runner requires a module path to run the test.
- FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
""")


PHASE_SPECIFIC_GUIDANCE = {
    PHASE_INVESTIGATION: textwrap.dedent("""
    ## 🔍 INVESTIGATION PHASE - Focus Areas:
    - Your primary goal is to UNDERSTAND the problem deeply before making changes
    - Use search tools extensively to locate all relevant code
    - Read and analyze the codebase structure
    - Identify all files and functions related to the issue
    - Document your findings about the root cause
    - DO NOT make code changes yet - only investigate and understand
    - Look for similar patterns or related issues in the codebase
    - Understand dependencies and relationships between components
    """),
    
    PHASE_PLANNING: textwrap.dedent("""
    ## 📋 PLANNING PHASE - Focus Areas:
    - Based on investigation, design a comprehensive solution approach
    - Propose at least 2-3 different solution strategies
    - Consider edge cases and potential side effects
    - Plan the sequence of changes needed
    - Identify which tests will validate your fix
    - Think about backward compatibility
    - Document your planned approach before implementation
    - Get approval for your solution strategy using get_approval_for_solution
    """),
    
    PHASE_IMPLEMENTATION: textwrap.dedent("""
    ## ⚙️ IMPLEMENTATION PHASE - Focus Areas:
    - Now you can apply the approved solution plan
    - Make precise, targeted code changes using apply_code_edit
    - Follow the plan from the planning phase
    - Make one logical change at a time
    - After each significant change, run relevant tests
    - If tests fail, analyze and adjust your approach
    - Ensure code quality and style consistency
    - Handle all identified edge cases
    """),
    
    PHASE_VALIDATION: textwrap.dedent("""
    ## ✅ VALIDATION PHASE - Focus Areas:
    - Thoroughly test all changes made
    - Run the full test suite to ensure no regressions
    - Verify all edge cases are handled correctly
    - Check that the original problem is fully resolved
    - Review code quality and documentation
    - Ensure backward compatibility is maintained
    - If any issues found, return to implementation phase
    - When confident, call finish with detailed summary
    """)
}

PROXY_SERVICE_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
EXECUTION_TIMEOUT_SEC = int(os.getenv("AGENT_TIMEOUT", "2000"))
PATCH_TIMEOUT_LIMIT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

MODEL_GLM = "zai-org/GLM-4.5-FP8"
MODEL_KIMI = "moonshotai/Kimi-K2-Instruct"
MODEL_DEEPSEEK = "deepseek-ai/DeepSeek-V3-0324"
MODEL_QWEN = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AVAILABLE_MODELS=[MODEL_GLM, MODEL_KIMI, MODEL_DEEPSEEK, MODEL_QWEN]
REPAIR_MAX_STEPS = 400

VERBOSE_LOGGING=True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('agent_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

run_id=None
  
class ChainOfThoughtProcessor:
    class Action:
            
        def __init__(self, reasoning_step: str, tool_identifier: str, tool_parameters: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,attempt_count:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.reasoning_step=reasoning_step
            self.tool_identifier=tool_identifier
            self.tool_parameters=tool_parameters
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.attempt_count=attempt_count
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=5):
        self.thoughts: list[ChainOfThoughtProcessor.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep
        
    def is_valid_tool_call(self, tool_identifier: str|list, tool_parameters: dict|list) -> bool:
        # Return True if no previous thoughts exist
        thought_count = len(self.thoughts)
        if thought_count == 0:
            return True
            
        # Get last action details
        previous_action = self.thoughts[-1]
        prev_tool = previous_action.tool_identifier
        prev_args = previous_action.tool_parameters
        
        # Check for exact duplicate - return False if found
        is_duplicate = (tool_identifier == prev_tool) and (tool_parameters == prev_args)
        return not is_duplicate

    def add_action(self, action: ChainOfThoughtProcessor.Action) -> bool: # don't add if thought is repeated
        # if not self.is_valid_tool_call(action.tool_identifier, action.tool_parameters):
        #     return False
        self.thoughts.append(action)
        return True
        
    def is_thought_repeated(self)->bool:
        # Need at least 2 thoughts to compare
        total_thoughts = len(self.thoughts)
        if total_thoughts < 2:
            return False
        
        # Extract last two actions
        recent_action = self.thoughts[-1]
        before_action = self.thoughts[-2]
        
        # Compare tool identifiers and parameters
        same_tool = (recent_action.tool_identifier == before_action.tool_identifier)
        same_params = (recent_action.tool_parameters == before_action.tool_parameters)
        
        return same_tool and same_params
    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i<len(self.thoughts)-self.latest_observations_to_keep:
                assistant_str = (
                    f"reasoning_step:{thought.reasoning_step}\n"
                    f"tool_identifier:{thought.tool_identifier}\n"
                    f"tool_parameters:{thought.tool_parameters}\n"
                )
                # Compute observation summary length safely for str/list/None
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str=( f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n")
                
            else:
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"reasoning_step:{thought.reasoning_step}\ntool_identifier:{thought.tool_identifier}\ntool_parameters:{thought.tool_parameters}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"reasoning_step:{thought.reasoning_step}\n"
                            f"tool_identifier:{thought.tool_identifier}\n"
                            f"tool_parameters:{thought.tool_parameters}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"reasoning_step:{thought.reasoning_step}\ntool_identifier:{thought.tool_identifier}\ntool_parameters:{thought.tool_parameters}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
            messages.append({"role":"assistant","content":assistant_str})
            messages.append({"role":"user","content":user_str})
        return messages
    
    def export_to_csv(self,file_path:str="./xray.csv"):
        with open(file_path, "w") as f:
            writer=csv.writer(f)
            writer.writerow(["reasoning_step","tool_identifier","tool_parameters","observation","is_error","raw_response","attempt_count","is_deleted"])
            if len(self.thoughts)>0:
                for thought in self.thoughts:
                    writer.writerow([thought.reasoning_step,thought.tool_identifier,thought.tool_parameters,thought.observation,thought.is_error,thought.raw_response,thought.attempt_count,str(thought.inference_error_counter),str(thought.request_data),len(str(thought.request_data)),thought.is_deleted])
                
                
    def get_tokens_used(self):
        # Heuristic: approximately 0.75 tokens per word
        message_list = self.to_str()
        
        # Build combined text from all messages
        combined_text = ""
        for msg in message_list:
            combined_text += msg["content"] + "\n"
        
        # Count words and estimate tokens
        words = combined_text.split()
        total_words = len(words)
        estimated_tokens = total_words * 0.75
        
        return int(estimated_tokens)

class HelperUtilities:
    @classmethod
    def get_available_modules(cls) -> set[str]:
        """Return the set of top-level module names that can be imported in the
        *current* Python environment.

        The result includes:
        • built-in/stdlib module names (`sys.builtin_module_names`)
        • every top-level name discoverable on `sys.path` via `pkgutil.iter_modules()`
        This is useful when we need to check whether a piece of code depends on a
        package that is *not* present in the environment.
        """
        import sys, pkgutil

        available: set[str] = set(sys.builtin_module_names)
        for module_info in pkgutil.iter_modules():
            # Only keep the top-level package name (before the first dot)
            top_level = module_info.name.split(".")[0]
            available.add(top_level)
        return available

    @classmethod
    def message_to_str(cls,messages:list[dict]): 
        final_str=""
        for message in messages:
            role=message["role"]
            content=message["content"]
            final_str+=f"{role}: {content}\n"
        return final_str
    
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        # Split into individual lines
        line_array = strings.split("\n")
        total_lines = len(line_array)
        
        # Check if truncation needed
        needs_truncation = total_lines > n
        
        if needs_truncation:
            # Keep first n lines
            kept_lines = line_array[:n]
            remaining = total_lines - n
            truncated_text = "\n".join(kept_lines)
            suffix = f"\n...({remaining} more lines)"
            return truncated_text + suffix
        
        return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json=NetworkRequestHandler.fix_json_string_with_llm(json_string)
                # if fixed_json == ""
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError(f"Invalid JSON: {json_string}")
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])

class FunctionNodeWalker(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class ClassNodeWalker(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

class NetworkRequestHandler:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,response_text:str)->bool:
        if type(response_text) is dict and response_text.get("error",None) is not None and response_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if not response_text.strip().endswith("}") and not response_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(response_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in response_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in response_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in response_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in response_text or 'Connection refused' in response_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=MODEL_DEEPSEEK)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0)->str:
        global run_id
        url = f"{PROXY_SERVICE_URL.rstrip('/')}/api/inference"
        print("[REQUEST] run_id:", run_id)

        # Cache miss - make the actual request
        request_data = {
                "run_id": run_id if run_id else str(uuid4()),
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        
        try:
            response = requests.post(url, json=request_data, timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after 120 seconds for model {model}")
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for model {model}: {e}")
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for model {model}: {e}")
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return f"ERROR: Request failed for model {model}"
        
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.error(f"Invalid JSON response for model {model}: {e}")
            logger.error(f"Response content: {response.text[:500]}...")
            return f"ERROR: Invalid JSON response for model {model}"
        
        try:
            is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
            if is_oai_interface:
                response_text=response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    response_text=response_json.strip("\n").strip()
                else:
                    response_text=response_json
            if type(response_text) is not dict:
                response_text=response_text.lstrip()
            return response_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response structure for model {model}: {e}")
            logger.error(f"Response JSON: {response_json}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error processing response for model {model}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: Unexpected error for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            model: str,
                            max_retries: int = 5, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        response_text='not defined'
        error_tracking=cls.get_error_counter()
        reasoning_step, tool_identifier, tool_parameters = None, None, None
        attempt_count=0
        for attempt in range(max_retries):
            try:
                attempt_count+=1
                index = AVAILABLE_MODELS.index(model) if model in AVAILABLE_MODELS else -1
                response_text=cls.make_request(messages,model=AVAILABLE_MODELS[(index + attempt)%len(AVAILABLE_MODELS)], temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(response_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                reasoning_step, tool_identifier, tool_parameters,error_msg = cls.parse_response(response_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {response_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_tracking[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_tracking[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_tracking[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_tracking[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_tracking[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_tracking[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_tracking[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":response_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(1.2*delay, 1.5*delay))
                    continue
                else:
                    error_tracking[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return reasoning_step, tool_identifier, tool_parameters,response_text,attempt_count,error_tracking,messages
    
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, tool_parameters: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''

        tool_parameters=tool_parameters.replace('```json','').strip('```')
        error_msg=''

        try:
            tool_parameters = HelperUtilities.load_json(tool_parameters.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {tool_parameters}"    
            try:
                tool_parameters = cls.parse_malformed_json(ToolExecutionManager.get_tool_args_for_tool(tool_name,required=True), tool_parameters)
            except ToolExecutionManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return tool_parameters

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()),return_json:bool=False, temperature:float=0.0) -> dict:
        """Prod inference with caching. Default temperature=0.0 for determinism in debugging."""
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")

            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        reasoning_step,tool_identifier,tool_parameters,response_text,attempt_count,error_tracking,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        
        return reasoning_step,tool_identifier,tool_parameters,response_text,attempt_count,error_tracking,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*reasoning_step[\'\"]*:","reasoning_step:",text_resp)
        text_resp=re.sub("[\'\"]*tool_identifier[\'\"]*:","tool_identifier:",text_resp)
        text_resp=re.sub("[\'\"]*tool_parameters[\'\"]*:","tool_parameters:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "reasoning_step" not in text_resp and "tool_identifier:" in text_resp and "tool_parameters:" in text_resp and text_resp.find("tool_identifier:")<text_resp.find("tool_parameters:") and text_resp.find("tool_identifier:")>10:
            logger.info(f"reasoning_step not found in {text_resp[:50]}, adding it")
            text_resp="reasoning_step: "+text_resp
        if "tool_identifier:" in text_resp and "tool_parameters:" in text_resp and text_resp.find("tool_identifier:")<text_resp.find("tool_parameters:"):
            # remove all leading and trailing quotes in tool_name
            tool_identifier=text_resp.split("tool_identifier:")[1].split("tool_parameters:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"tool_identifier:[\'\" ]*{tool_identifier}[\'\" ]*","tool_identifier: "+tool_identifier,text_resp)
        
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        
        # PRE-VALIDATION: Check for common issues with specific guidance
        if not text_resp:
            return None, None, None, "Empty response - model may have hit context limit. Try shortening your previous message or observations."
        
        if len(text_resp) < 20:
            return None, None, None, f"Response too short ({len(text_resp)} chars). You must provide all three fields: reasoning_step, tool_identifier, and tool_parameters."
        
        # Check if response was truncated (doesn't end properly)
        if not text_resp.rstrip().endswith(('}', '"]', '"', 'None', ']')):
            return None, None, None, "Response appears truncated (doesn't end with proper JSON). This may indicate context limit reached. Shorten your reasoning_step or use more concise tool_parameters."
        
        # FIELD PRESENCE CHECK with specific guidance
        required_fields = {
            "reasoning_step:": "Explain your thought process and what you're doing",
            "tool_identifier:": "Specify which tool to use from the available tools list", 
            "tool_parameters:": "Provide tool arguments as valid JSON"
        }
        
        missing_fields = []
        for field, description in required_fields.items():
            if field not in text_resp:
                missing_fields.append(f"'{field}' - {description}")
        
        if missing_fields:
            return None, None, None, f"Missing required fields:\n" + "\n".join(f"  • {field}" for field in missing_fields) + "\n\nYour response must include all three: reasoning_step, tool_identifier, and tool_parameters in that exact order."
        
        # ORDER CHECK with specific guidance
        pos_reasoning = text_resp.find("reasoning_step:")
        pos_tool = text_resp.find("tool_identifier:")
        pos_params = text_resp.find("tool_parameters:")
        
        if not (pos_reasoning < pos_tool < pos_params):
            actual_order = sorted([
                (pos_reasoning, "reasoning_step"),
                (pos_tool, "tool_identifier"),
                (pos_params, "tool_parameters")
            ])
            actual_sequence = " → ".join([name for _, name in actual_order])
            return None, None, None, f"Fields are out of order. Found: {actual_sequence}\nRequired order: reasoning_step → tool_identifier → tool_parameters"
        
        # PARSING (existing logic with enhanced error messages)
        if "reasoning_step:" in text_resp and "tool_identifier:" in text_resp and "tool_parameters:" in text_resp and text_resp.find("reasoning_step:")<text_resp.find("tool_identifier:") and text_resp.find("tool_identifier:")<text_resp.find("tool_parameters:"):
            reasoning_step=text_resp.split("reasoning_step:")[1].split("tool_identifier:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("tool_identifier:")[1].split("tool_parameters:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("tool_parameters:")[1].strip().split("reasoning_step:")[0].strip().strip("\n")
            
            # Validate non-empty fields
            if not reasoning_step:
                return None, None, None, "reasoning_step is empty. Please explain what you're doing and why."
            
            if not next_tool_name_raw:
                return None, None, None, "tool_identifier is empty. Specify which tool to use. Available tools are listed in the system prompt."
            
            if not next_tool_args_raw or next_tool_args_raw == "{}":
                logger.warning(f"tool_parameters is empty or {{}} for tool '{next_tool_name_raw}' - may be valid if tool has no required parameters")
            
            try:
                # Enforce arrays per new contract: if single string/object, wrap as arrays
                if next_tool_name_raw.startswith("["):
                    tool_identifier = HelperUtilities.load_json(next_tool_name_raw)
                else:
                    tool_identifier = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(tool_identifier, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    tool_parameters = parsed_args
                else:
                    tool_parameters = [parsed_args for _ in tool_identifier]
            except JSONDecodeError as e:
                error_msg=f"Invalid JSON in tool_parameters: {str(e)}\n\nEnsure you use:\n  • Proper double quotes (not single quotes)\n  • Escaped special characters (\\n, \\\", etc.)\n  • No trailing commas\n  • Valid JSON syntax\n\nYour tool_parameters: {next_tool_args_raw[:200]}"
                HelperUtilities.log_to_failed_messages(text_resp)
                return None, None, None, error_msg
                
        else:
            if "reasoning_step:" not in text_resp:
                error_msg="Missing 'reasoning_step:' field. Explain your thought process before calling tools."
            elif "tool_identifier:" not in text_resp:
                error_msg="Missing 'tool_identifier:' field. Specify which tool you want to use."
            elif "tool_parameters:" not in text_resp:
                error_msg="Missing 'tool_parameters:' field. Provide the tool arguments as JSON."
            elif text_resp.find("reasoning_step:")>text_resp.find("tool_identifier:"):
                error_msg="Fields out of order: reasoning_step must come before tool_identifier."
            elif text_resp.find("tool_identifier:")>text_resp.find("tool_parameters:"):
                error_msg="Fields out of order: tool_identifier must come before tool_parameters."
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
                error_msg="Response format is invalid. Use this exact format:\n\nreasoning_step: <your detailed reasoning>\ntool_identifier: <tool_name>\ntool_parameters: {<valid_json>}"
            HelperUtilities.log_to_failed_messages(text_resp)
            return None,None,None,error_msg

        if len(tool_identifier) == 1:
            return reasoning_step, tool_identifier[0], tool_parameters[0], error_msg
            
        return reasoning_step, tool_identifier, tool_parameters,error_msg

class ToolExecutionManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
            GIT_OPERATION_FAILED=15
            GIT_CONFIG_ERROR=16
            GIT_STATE_ERROR=17
            GIT_MERGE_CONFLICT=18
            GIT_BRANCH_ERROR=19
            TEST_COVERAGE_ERROR = 20
            DEPENDENCY_ANALYSIS_ERROR = 21
            CODE_SMELL_DETECTION_ERROR = 22
            GIT_HISTORY_ERROR = 23
            CODE_QUALITY_ERROR = 24
            SOLUTION_VALIDATION_ERROR = 25
            CODE_STYLE_ERROR = 26
            SOLUTION_COMPARISON_ERROR = 27
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except ToolExecutionManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        # Preserve original function metadata
       
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True

        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method
    
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            # self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)

    def _run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
    
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=ToolExecutionManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=ToolExecutionManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=ToolExecutionManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES

            raise ToolExecutionManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"

        return observation


    def _add_line_numbers_to_content(self, content: str, start_line: int = 1) -> str:
        """Helper method to add line numbers to content."""
        lines = content.splitlines()
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_lines.append(f"{line_num:6}|{line}")
        return '\n'.join(numbered_lines)
    
    def _add_context_to_similar_match(self, original_content: str, formatted_match: str, context_lines: int = 2) -> str:
        """Add context lines around a similar match for better understanding."""
        lines = original_content.split('\n')
        
        # Extract the actual content from the formatted match (remove the description part)
        match_lines = formatted_match.split('\n')
        if len(match_lines) < 2:
            return formatted_match
            
        # Skip the description line (e.g., "Lines 45-47: ..." or "Line 23: ...")
        actual_content_lines = match_lines[1:]
        actual_content = '\n'.join(actual_content_lines)
        
        # Find where this content appears in the original file
        best_match_start = -1
        best_similarity = 0
        
        # Search for the best matching position in the original content
        for i in range(len(lines) - len(actual_content_lines) + 1):
            candidate_lines = lines[i:i + len(actual_content_lines)]
            candidate_content = '\n'.join(candidate_lines)
            
            import difflib
            similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_start = i
        
        if best_match_start == -1:
            return formatted_match  # Fallback to original if can't find position
        
        # Calculate context boundaries
        start_line = max(0, best_match_start - context_lines)
        end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
        
        # Build the context with line numbers
        context_lines_list = []
        for i in range(start_line, end_line):
            line_num = i + 1
            prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
            context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
        
        # Extract original description
        description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
        
        return f"{description}\n" + "\n".join(context_lines_list)

    def _find_most_similar_content(self, original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
        """Find the most similar content chunks to the search string."""
        import difflib
        
        # Split content into meaningful chunks
        lines = original_content.split('\n')
        
        # Try different chunk sizes to find the best match
        chunks = []
        
        # Individual lines
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
        
        # Multi-line chunks (3-5 lines) for better context
        search_lines = search_string.split('\n')
        target_chunk_size = max(3, len(search_lines))
        
        for i in range(len(lines) - target_chunk_size + 1):
            chunk_lines = lines[i:i + target_chunk_size]
            chunk_content = '\n'.join(chunk_lines).strip()
            if chunk_content:
                chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
        
        # Calculate similarity scores
        similarities = []
        for chunk_desc, chunk_content in chunks:
            ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
            if ratio > 0.3:  # Only include reasonably similar content
                similarities.append((ratio, chunk_desc, chunk_content))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            # Stage all modified, deleted, and new files
            subprocess.run(["git", "add", "-A"], check=False, capture_output=True, text=True)

            # Exclude any generated test/repro files from the final patch
            # We track these paths in self.generated_test_files
            try:
                for p in getattr(self, "generated_test_files", []) or []:
                    try:
                        subprocess.run(["git", "reset", p], check=False, capture_output=True, text=True)
                    except Exception:
                        continue
            except Exception:
                # If anything goes wrong with exclusions, proceed with best-effort patch
                pass

            # Produce staged diff only
            diff = subprocess.run(["git", "diff", "--cached"], check=False, capture_output=True, text=True)
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"

class RepairTaskToolExecutor(ToolExecutionManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]
        self.is_test_code_update_approved=False
        self.test_code_approval_reason: Optional[str]=None
        self.approved_test_file_paths:set[str]=set()
        self.approved_test_directories:set[str]=set()

        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations={
          k:0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
            
        # check if start and end line are not between a function..
        func_ranges=self.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line=end
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return HelperUtilities.limit_strings(content, n=limit) if limit!=-1  else content
    
    @ToolExecutionManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @ToolExecutionManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        # Guard: test code writes are blocked unless explicitly approved for this path
        if self._is_editing_test_code(file_path=file_path) and not self._is_test_path_approved(file_path):
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: Editing/creating test files requires prior approval via 'approve_test_code' for this path ('{file_path}')."
            )
        return self._save(file_path, content)
    
    @ToolExecutionManager.tool
    def save_checkpoint(self, checkpoint_name: str, description: str = "") -> str:
        '''
        Saves current code state as a named checkpoint for potential rollback.
        Useful before making risky changes or trying experimental approaches.
        
        Arguments:
            checkpoint_name: unique name for this checkpoint (e.g., "before_refactor", "working_state_1")
            description: optional description of what's being saved
        
        Output:
            Confirmation with checkpoint name and instructions for restoration
        '''
        try:
            # Create stash with descriptive message
            message = f"{checkpoint_name}: {description}" if description else checkpoint_name
            result = subprocess.run(
                ["git", "stash", "push", "-u", "-m", message],
                capture_output=True, text=True, check=True, timeout=30
            )
            
            # Store checkpoint name for tracking
            if not hasattr(self, '_checkpoints'):
                self._checkpoints = []
            self._checkpoints.append(checkpoint_name)
            
            logger.info(f"Checkpoint '{checkpoint_name}' created successfully")
            return f"Checkpoint '{checkpoint_name}' saved successfully. Use restore_checkpoint('{checkpoint_name}') to rollback if needed."
            
        except subprocess.TimeoutExpired:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.GIT_OPERATION_FAILED,
                "Git stash operation timed out after 30 seconds"
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.GIT_OPERATION_FAILED,
                f"Failed to create checkpoint: {error_msg}"
            )

    @ToolExecutionManager.tool
    def restore_checkpoint(self, checkpoint_name: str) -> str:
        '''
        Restores code to a previously saved checkpoint, discarding all current changes.
        WARNING: This will permanently discard all uncommitted changes.
        
        Arguments:
            checkpoint_name: name of checkpoint to restore (from save_checkpoint)
        
        Output:
            Confirmation of restoration or error if checkpoint not found
        '''
        try:
            # List all stashes to find the matching checkpoint
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True, text=True, check=True, timeout=30
            )
            
            stash_list = result.stdout.strip().split('\n') if result.stdout.strip() else []
            stash_index = None
            
            # Find matching checkpoint in stash list
            for i, stash_entry in enumerate(stash_list):
                if checkpoint_name in stash_entry:
                    stash_index = i
                    break
            
            if stash_index is None:
                available = [s.split(':', 2)[2].strip() if ':' in s else s for s in stash_list]
                raise ToolExecutionManager.Error(
                    ToolExecutionManager.Error.ErrorType.GIT_STATE_ERROR,
                    f"Checkpoint '{checkpoint_name}' not found. Available checkpoints: {available}"
                )
            
            # Reset current changes
            subprocess.run(["git", "reset", "--hard"], check=True, timeout=30, capture_output=True)
            subprocess.run(["git", "clean", "-fdx"], check=True, timeout=30, capture_output=True)
            
            # Apply the stash
            subprocess.run(
                ["git", "stash", "apply", f"stash@{{{stash_index}}}"],
                check=True, timeout=30, capture_output=True, text=True
            )
            
            logger.info(f"Checkpoint '{checkpoint_name}' restored successfully")
            return f"Successfully restored checkpoint '{checkpoint_name}'. All changes reverted to that state."
            
        except subprocess.TimeoutExpired:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.GIT_OPERATION_FAILED,
                "Git restore operation timed out after 30 seconds"
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.GIT_OPERATION_FAILED,
                f"Failed to restore checkpoint: {error_msg}"
            )

    @ToolExecutionManager.tool
    def list_checkpoints(self) -> str:
        '''
        Lists all available checkpoints that can be restored.
        Useful to see what saved states are available before deciding which to restore.
        
        Output:
            List of checkpoint names with their descriptions, or message if none exist
        '''
        try:
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True, text=True, check=True, timeout=30
            )
            
            if not result.stdout.strip():
                return "No checkpoints available. Use save_checkpoint() to create one before making risky changes."
            
            checkpoints = []
            for line in result.stdout.strip().split('\n'):
                # Parse: stash@{0}: On branch: checkpoint_name: description
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    checkpoints.append(f"  - {parts[2].strip()}")
                else:
                    checkpoints.append(f"  - {line}")
            
            return "Available checkpoints:\n" + "\n".join(checkpoints)
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
            return f"Error listing checkpoints: {e}"
    
    @ToolExecutionManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        
        Each solution should include:
        1. **Root Cause Analysis**: What is causing the bug? Be specific about the code path
        2. **Proposed Changes**: Exactly what code will be modified and how
        3. **Why This Works**: Explain the logic of why this fix resolves the root cause
        4. **Edge Cases Handled**: What edge cases does this solution handle?
        5. **Trade-offs**: What are the pros and cons of this approach?
        6. **Test Strategy**: How will you verify this works?
        
        Guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case
        2. Solution should handle all edge cases mentioned in the problem statement
        3. Prefer minimal changes that preserve existing patterns
        4. Consider backward compatibility
        5. Think about caching/consistency if dealing with callable values or computed properties
        
        Arguments:
            solutions: list of solutions proposed by you. Each solution should be very detailed with all 6 components above.
            selected_solution: Index of the solution you think is the best (0-indexed).
            reason_for_selection: Detailed reason for selecting this solution over others, referencing specific advantages.
            
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        
        # Validate solutions is a list with at least 2 items
        if type(solutions) is not list or len(solutions) < 2:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: solutions must be a list with length at least 2. Each solution should include: Root Cause Analysis, Proposed Changes, Why This Works, Edge Cases Handled, Trade-offs, and Test Strategy.")

        # Validate that each solution is detailed enough (at least 100 characters)
        for i, sol in enumerate(solutions):
            if not isinstance(sol, str):
                raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: Solution {i} must be a string.")
            if len(sol.strip()) < 100:
                raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: Solution {i} is too brief (< 100 chars). Each solution should be detailed and include all 6 components: Root Cause Analysis, Proposed Changes, Why This Works, Edge Cases, Trade-offs, Test Strategy.")

        self.is_solution_approved = True
        return "Approved. You may now implement the selected solution using apply_code_edit."

    @ToolExecutionManager.tool
    def approve_test_code(self, reason: str, file_paths: List[str] = None, directories: List[str] = None) -> str:
        '''
        Approve narrowly scoped edits to test-related files when strictly necessary (e.g., config tweaks or minor harness updates).
        Arguments:
            reason: succinct explanation for why test changes are required
            file_paths: specific test file paths approved for edit (absolute or relative)
            directories: directories under which test files are approved (absolute or relative)
        Output:
            approval: approved/not approved summary with the approved scopes recorded
        '''
        # Basic validation: require at least one scope to keep approval narrow
        file_paths = file_paths or []
        directories = directories or []
        if not file_paths and not directories:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                "Error: Provide at least one file path or directory to scope test edit approval."
            )

        def _norm(p: str) -> str:
            try:
                return os.path.normpath(os.path.abspath(p))
            except Exception:
                return os.path.normpath(p)

        # Record approval state and normalized scopes
        self.is_test_code_update_approved = True
        self.test_code_approval_reason = (reason or "").strip()
        for p in file_paths:
            self.approved_test_file_paths.add(_norm(p))
        for d in directories:
            self.approved_test_directories.add(_norm(d))

        summary = ["Approved test code changes with the following scope:"]
        if self.test_code_approval_reason:
            summary.append(f"- reason: {self.test_code_approval_reason}")
        if self.approved_test_file_paths:
            summary.append("- files:")
            for p in sorted(self.approved_test_file_paths):
                summary.append(f"  - {p}")
        if self.approved_test_directories:
            summary.append("- directories:")
            for d in sorted(self.approved_test_directories):
                summary.append(f"  - {d}")
        return "\n".join(summary)
          
    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
 
    @ToolExecutionManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionNodeWalker(content)
                visitor.visit(tree)
                
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"

        return functions

    @ToolExecutionManager.tool
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassNodeWalker(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"

        return classes

    @ToolExecutionManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        If more context is needed (e.g., surrounding functions, classes, etc.), follow up with get_classes or get_functions.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Compile the pattern safely. If the provided pattern is not a valid
        # regex (e.g., contains glob characters like '*'), fall back to a
        # literal search using re.escape to avoid regex errors/prompt tricks.
        def _compile_safe(pattern: str, flags: int):
            try:
                return re.compile(pattern, flags)
            except re.error:
                return re.compile(re.escape(pattern), flags)

        filename_pattern = _compile_safe(search_term, search_flags)
        content_pattern = filename_pattern

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name (safe regex)
                    if filename_pattern.search(file_path):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not content_pattern.search(content):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionNodeWalker(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if content_pattern.search(body):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if content_pattern.search(line):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")

        output = HelperUtilities.limit_strings("\n".join(output), n=100)
        if not output:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase.")
        return output

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
            tree = None  # Fallback if file cannot be parsed.

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        
        # Only process if tree is valid
        if tree is None:
            return func_ranges
        
        # Walk through AST nodes
        ast_nodes = list(ast.walk(tree))
        for node in ast_nodes:
            # Check if node is function definition
            is_func = isinstance(node, ast.FunctionDef)
            is_async_func = isinstance(node, ast.AsyncFunctionDef)
            
            if is_func or is_async_func:
                # Extract line information
                start_line = getattr(node, 'lineno', None)
                end_line = getattr(node, 'end_lineno', None)
                
                # Only add if both lines are present
                has_both_lines = (start_line is not None) and (end_line is not None)
                if has_both_lines:
                    func_info = (start_line, end_line, node.name)
                    func_ranges.append(func_info)
        
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{file_path}'")

        func_ranges=self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return HelperUtilities.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @ToolExecutionManager.tool
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)

    # @tool
    def search_recurive_in_all_files_in_directory(self, directory_path: str, search_term: str)->str:
        '''
        Locates text patterns recursively within all files in a specific directory
        Arguments:
            directory_path: target directory for pattern matching
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: directory '{directory_path}' does not exist.")
        output=subprocess.run(["bash", "-c", f"grep -rn --include='*.py' {directory_path} -e '{search_term}'"], capture_output=True)
        output=output.stdout.decode("utf-8")
        output=HelperUtilities.limit_strings(output, n=100)
        if not output:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{directory_path}'")
        return output
    
    @ToolExecutionManager.tool
    def start_over(self,problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass

            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            # Log stderr separately so it never pollutes the patch.
            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"

    def create_new_file(self,file_path:str, content:str)->str:
        '''
        Generates new file with specified content at target location. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            file_path: destination path for new file
            content: text content for file creation
        '''
        # Guard: block creating test files unless approved for this path
        if self._is_editing_test_code(file_path=file_path) and not self._is_test_path_approved(file_path):
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: Creating test files requires prior approval via 'approve_test_code' for this path ('{file_path}')."
            )
        return self._save(file_path, content)

    @ToolExecutionManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        # Remember last tests invoked for post-patch verification
        try:
            self._last_tests = file_paths[:]
        except Exception:
            pass

        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            # Avoid shell=True; increase timeout for heavier suites
            result = subprocess.run(["pytest", *file_paths], shell=False, capture_output=True, text=True, timeout=180, env=TEST_ENV)
            output = (result.stdout or "") + (result.stderr or "")
        else:
            if self.test_runner_mode == "MODULE":
                modules = [convert_path_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180, env=TEST_ENV)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [sanitize_file_path(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180, env=TEST_ENV)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    @ToolExecutionManager.tool
    def run_code(self,content:str,file_path:str)->str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        # Parse the file's AST to collect import statements
        
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available;
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

               

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60, env=TEST_ENV)
        if result.returncode!=0:
            
            error_type=ToolExecutionManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=ToolExecutionManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=ToolExecutionManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise ToolExecutionManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
       

        return observation

    def _insert_after_imports(self, content: str, block: str) -> str:
        '''Helper to insert test function after import statements'''
        lines = content.split('\n')
        last_import_idx = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import_idx = i
            elif stripped and not stripped.startswith('#'):
                # Stop at first non-import, non-comment line
                break
        
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, '\n' + block)
            return '\n'.join(lines)
        else:
            # No imports found, insert at beginning after any header comments
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    lines.insert(i, block + '\n')
                    return '\n'.join(lines)
            # File is empty or only comments
            return block + '\n\n' + content

    def _insert_before_main(self, content: str, block: str) -> str:
        '''Helper to insert test function before if __name__ == "__main__"'''
        lines = content.split('\n')
        main_idx = -1
        
        for i, line in enumerate(lines):
            if 'if __name__' in line and '__main__' in line:
                main_idx = i
                break
        
        if main_idx >= 0:
            lines.insert(main_idx, block + '\n')
            return '\n'.join(lines)
        else:
            # No main block, append to end
            return content.rstrip() + '\n\n' + block + '\n'

    @ToolExecutionManager.tool
    def insert_test_function(self, target_test_file: str, test_function_code: str, position: str = "append") -> str:
        '''
        Inserts a test function into an existing test file at the specified position.
        More sophisticated than run_code - integrates cleanly with existing test files.
        Generated tests are automatically excluded from the final patch.
        
        Arguments:
            target_test_file: path to existing test file (will create if doesn't exist)
            test_function_code: complete test function code (should start with 'def test_')
            position: where to insert - "append" (end of file), "after_imports", or "before_main"
        
        Output:
            Success message with insertion location, or error if syntax issues found
        '''
        if not target_test_file.endswith('.py'):
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_FILE_PATH,
                f"Test file '{target_test_file}' must be a Python file (.py extension)"
            )
        
        # Validate test function code
        test_fn = test_function_code.strip()
        if not test_fn:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL,
                "test_function_code cannot be empty"
            )
        
        if 'def test_' not in test_fn:
            logger.warning(f"Test function code doesn't contain 'def test_' - unusual but proceeding")
        
        is_new_file = not os.path.exists(target_test_file)
        
        if is_new_file:
            # Create new file with the test function
            new_content = test_fn + '\n'
            is_error, error = self.check_syntax_error(new_content, target_test_file)
            if is_error:
                raise ToolExecutionManager.Error(
                    ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR,
                    f"Generated test function has syntax error: {error.message}"
                )
            
            # Ensure directory exists
            dir_name = os.path.dirname(target_test_file)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        else:
            # Read existing file
            original_content = self._get_file_content(target_test_file, limit=-1)
            
            # Check if this exact test function already exists
            if test_fn in original_content:
                if target_test_file not in self.generated_test_files:
                    self.generated_test_files.append(target_test_file)
                return f"Test function already exists in '{target_test_file}', no changes made."
            
            # Insert based on position
            if position == "append":
                new_content = original_content.rstrip() + '\n\n' + test_fn + '\n'
            elif position == "after_imports":
                new_content = self._insert_after_imports(original_content, test_fn)
            elif position == "before_main":
                new_content = self._insert_before_main(original_content, test_fn)
            else:
                raise ToolExecutionManager.Error(
                    ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL,
                    f"Invalid position '{position}'. Use 'append', 'after_imports', or 'before_main'"
                )
            
            # Validate syntax of modified content
            is_error, error = self.check_syntax_error(new_content, target_test_file)
            if is_error:
                raise ToolExecutionManager.Error(
                    ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR,
                    f"Inserting test caused syntax error: {error.message}"
                )
        
        # Save the file
        with open(target_test_file, 'w') as f:
            f.write(new_content)
        
        # Track for exclusion from final patch
        if target_test_file not in self.generated_test_files:
            self.generated_test_files.append(target_test_file)
        
        action = "created" if is_new_file else f"updated (inserted at position '{position}')"
        logger.info(f"Test file {action}: {target_test_file}")
        return f"Test function successfully {action} in '{target_test_file}'"

    @ToolExecutionManager.tool
    def bash_tool(self, command: str, timeout: int = 30) -> str:
        '''
        Executes bash commands for file system operations and utilities.
        Useful for commands like ls, find, grep, cat, etc.

        Arguments:
            command: bash command to execute (e.g., "ls -la", "find . -name '*.py'", "grep -r 'pattern' .")
            timeout: maximum execution time in seconds (default: 30)

        Output:
            Returns stdout from the command execution.
            Returns error message if command fails or times out.
        
        Examples:
            bash_tool(command="ls -la")
            bash_tool(command="find . -name 'test_*.py'")
            bash_tool(command="grep -r 'def main' .")
        '''
        try:
            # Security: whitelist safe commands
            dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', '> /dev/', 'curl', 'wget']
            for pattern in dangerous_patterns:
                if pattern in command:
                    raise ToolExecutionManager.Error(
                        ToolExecutionManager.Error.ErrorType.INVALID_ARGUMENT,
                        f"Command contains potentially dangerous pattern: {pattern}"
                    )
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                raise ToolExecutionManager.Error(
                    ToolExecutionManager.Error.ErrorType.RUNTIME_ERROR,
                    f"Command failed with exit code {result.returncode}:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                )
            
            output = result.stdout.strip()
            if not output and result.stderr:
                output = result.stderr.strip()
            
            # Limit output size
            if len(output) > 10000:
                output = output[:10000] + f"\n... (output truncated, {len(output)} total characters)"
            
            return output if output else "Command executed successfully with no output."
            
        except subprocess.TimeoutExpired:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.TIMEOUT,
                f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.RUNTIME_ERROR,
                f"Error executing bash command: {str(e)}"
            )

    @ToolExecutionManager.tool
    def fetch_url_content(self, url: str, extract_text: bool = True, max_chars: int = 15000) -> str:
        """
        Fetches content from a URL (documentation, GitHub issues, etc.).
        Extracts only the meaningful content from <body> tag, ignoring headers, navigation, footers.
        Uses smart chunking to keep the most relevant parts when content is large.
        
        Arguments:
            url: The URL to fetch (e.g., "https://github.com/user/repo/issues/123", "https://docs.python.org/...")
            extract_text: If True, extracts and returns clean text content. If False, returns raw HTML/content
            max_chars: Maximum characters to return (default: 15000)
        
        Output:
            Text content from the URL (body only), or error message if fetch fails
        
        Examples:
            fetch_url_content(url="https://github.com/django/django/issues/12345")
            fetch_url_content(url="https://docs.python.org/3/library/asyncio.html", extract_text=True)
        """
        try:
            import urllib.request
            import urllib.error
            from html.parser import HTMLParser
            import re
            
            class SmartTextExtractor(HTMLParser):
                """Enhanced HTML parser focusing on body content with smart filtering"""
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_tags = {'script', 'style', 'head', 'meta', 'link', 'noscript', 'iframe'}
                    # Tags that often contain navigation/footer content
                    self.nav_tags = {'nav', 'header', 'footer', 'aside'}
                    self.current_tag = None
                    self.in_body = False
                    self.in_nav = False
                    self.nav_depth = 0
                    
                def handle_starttag(self, tag, attrs):
                    if tag == 'body':
                        self.in_body = True
                    
                    # Track if we're in navigation/header/footer
                    if tag in self.nav_tags:
                        self.in_nav = True
                        self.nav_depth += 1
                    
                    self.current_tag = tag
                    
                def handle_endtag(self, tag):
                    if tag == 'body':
                        self.in_body = False
                    
                    if tag in self.nav_tags:
                        self.nav_depth -= 1
                        if self.nav_depth <= 0:
                            self.in_nav = False
                            self.nav_depth = 0
                    
                    self.current_tag = None
                    
                def handle_data(self, data):
                    # Only collect text if we're in body, not in skip tags, and not in navigation
                    if (self.in_body and 
                        self.current_tag not in self.skip_tags and 
                        not self.in_nav):
                        text = data.strip()
                        # Filter out very short fragments and common UI elements
                        if text and len(text) > 2:
                            # Skip common navigation/UI text
                            lower_text = text.lower()
                            skip_patterns = ['skip to', 'menu', 'search', 'login', 'sign in', 'cookie']
                            if not any(pattern in lower_text for pattern in skip_patterns) or len(text) > 30:
                                self.text_parts.append(text)
                            
                def get_text(self):
                    return '\n'.join(self.text_parts)
            
            def extract_body_content(html_str):
                """Extract only content between <body> tags using simple string manipulation"""
                # Try to find body content
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html_str, re.DOTALL | re.IGNORECASE)
                if body_match:
                    return body_match.group(1)
                return html_str
            
            def smart_truncate(text, max_length):
                """Intelligently truncate text keeping beginning and end, removing middle"""
                if len(text) <= max_length:
                    return text
                
                # Keep 60% from start, 30% from end, note the gap
                start_chars = int(max_length * 0.60)
                end_chars = int(max_length * 0.30)
                
                # Try to break at paragraph boundaries
                start_part = text[:start_chars]
                end_part = text[-end_chars:]
                
                # Find last double newline in start part
                last_para = start_part.rfind('\n\n')
                if last_para > start_chars * 0.8:  # If we can save at least 80% of start
                    start_part = start_part[:last_para]
                
                # Find first double newline in end part
                first_para = end_part.find('\n\n')
                if first_para > 0 and first_para < end_chars * 0.2:  # Within first 20%
                    end_part = end_part[first_para:]
                
                removed_chars = len(text) - len(start_part) - len(end_part)
                
                return (f"{start_part}\n\n"
                       f"... [Middle section removed: ~{removed_chars} characters] ...\n\n"
                       f"{end_part}")
            
            # Set up request with headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            request = urllib.request.Request(url, headers=headers)
            
            # Fetch the content with timeout
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
                
                # Try to decode content
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content_str = content.decode('latin-1')
                    except:
                        return f"Error: Unable to decode content from {url}"
                
                # If extract_text is True and content looks like HTML, parse it
                if extract_text and ('<html' in content_str.lower() or '<body' in content_str.lower()):
                    # First, extract only body content
                    # body_content = extract_body_content(content_str)
                    
                    parser = SmartTextExtractor()
                    try:
                        parser.feed(content_str)
                        extracted_text = parser.get_text()
                        
                        # Clean up excessive whitespace
                        extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)
                        
                        # Smart truncation if too long
                        if len(extracted_text) > max_chars:
                            extracted_text = smart_truncate(extracted_text, max_chars)
                        
                        return f"Content from {url}:\n\n{extracted_text}"
                    except Exception as e:
                        # If parsing fails, try simple body extraction
                        logger.warning(f"HTML parsing failed for {url}: {e}")
                        if len(body_content) > max_chars:
                            body_content = smart_truncate(body_content, max_chars)
                        return f"Content from {url}:\n\n{body_content}"
                
                # Return raw or non-HTML content
                if len(content_str) > max_chars:
                    content_str = smart_truncate(content_str, max_chars)
                
                return f"Content from {url}:\n\n{content_str}"
                
        except urllib.error.HTTPError as e:
            return f"Error: HTTP {e.code} - {e.reason} when fetching {url}"
        except urllib.error.URLError as e:
            return f"Error: Failed to reach {url} - {e.reason}"
        except TimeoutError:
            return f"Error: Request to {url} timed out after 30 seconds"
        except Exception as e:
            return f"Error fetching URL {url}: {str(e)}"

    def _is_editing_test_code(self, file_path: str) -> bool:
        lowered_path = (file_path or "").lower()
        if ("test" in lowered_path or "/tests/" in lowered_path or "reproduce" in lowered_path or lowered_path.endswith("tests.py")) and not ("/src/" in lowered_path or lowered_path.startswith("src/")):
            return True
        return False

    def _is_test_path_approved(self, file_path: str) -> bool:
        """Return True if test edits are approved and the given path is within the approved scope.
        Approval requires prior call to approve_test_code and that the file path matches either an approved file
        or lies under an approved directory.
        """
        if not (self.is_test_code_update_approved):
            return False
        try:
            norm_path = os.path.normpath(os.path.abspath(file_path))
        except Exception:
            norm_path = os.path.normpath(file_path)

        if norm_path in self.approved_test_file_paths:
            return True
        for d in self.approved_test_directories:
            # Ensure directory path ends properly when comparing commonpath
            try:
                common = os.path.commonpath([norm_path, d])
                if common == d:
                    return True
            except Exception:
                # On path errors, do a safe prefix check as fallback
                if norm_path.startswith(d.rstrip("/")):
                    return True
        return False

    @ToolExecutionManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
        '''
        Performs targeted text replacement within source files. **Important**: If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace (must match EXACTLY with proper indentation and whitespace)
        replace: new text content to substitute
            
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        # Disallow modifying test files unless explicitly approved for this path
        if self._is_editing_test_code(file_path=file_path) and not self._is_test_path_approved(file_path):
            raise ToolExecutionManager.Error(
                ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: Editing test files is not allowed without prior 'approve_test_code' for this path ('{file_path}')."
            )
        if not self.is_solution_approved:
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: file '{file_path}' does not exist.")
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace (including proper indentation).")
                # Try to find similar content to help the agent
                similar_matches = self._find_most_similar_content(original, search, max_results=3)
                if similar_matches:
                    similar_info = "\n\nDid you mean one of these? (showing top matches):\n"
                    for ratio, match_info in similar_matches:
                        similar_info += f"\n[Similarity: {ratio:.2%}]\n{match_info}\n"
                    raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace (match indentation precisely).{similar_info}")
                raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=self.check_syntax_error(new_content)
                        if not is_error:
                            self.save_file(file_path, new_content)
                                
                            return "ok, code edit applied successfully"
                        else:
                            error.message="code edit failed. "+error.message
                            raise error
                except ToolExecutionManager.Error as e:
                    raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease include more context in your search string to make it unique (currently found {num_hits} times).")
    
    @ToolExecutionManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            raise ToolExecutionManager.Error(ToolExecutionManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

class StrategicPlanner:
    """Generates high-level solution strategies"""
    
    STRATEGY_PROMPT = textwrap.dedent("""
    Analyze this problem and generate 3 distinct solution strategies. Each strategy should include:
    - Name and approach description
    - Key steps (high-level)
    - Complexity (low/medium/high)
    - Risk level (low/medium/high)
    - Confidence score (0-1)
    
    Problem: {problem_statement}
    
    Respond in JSON format:
    {{
        "strategies": [
            {{
                "name": "strategy_name",
                "description": "approach description",
                "steps": ["step1", "step2", "step3"],
                "complexity": "low/medium/high",
                "risk": "low/medium/high",
                "confidence": 0.8
            }}
        ]
    }}
    """)
    
    def __init__(self, model_name: str = MODEL_DEEPSEEK):
        self.model_name = model_name
    
    def generate_strategies(self, problem_statement: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": "You are a strategic planning expert."},
                {"role": "user", "content": self.STRATEGY_PROMPT.format(problem_statement=problem_statement)}
            ]
            
            response = NetworkRequestHandler.make_request(messages, model=self.model_name)
            
            if response.strip().startswith('```json'):
                response = response.strip()[7:]
            if response.strip().startswith('```'):
                response = response.strip()[3:]
            if response.strip().endswith('```'):
                response = response.strip()[:-3]
            response = response.strip()
            
            parsed_response = json.loads(response)
            
            if parsed_response and "strategies" in parsed_response:
                return parsed_response
            
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
        
        return {
            "strategies": [
                {
                    "name": "Conservative Fix",
                    "description": "Minimal targeted changes",
                    "steps": ["Locate issue", "Apply minimal fix", "Test"],
                    "complexity": "low",
                    "risk": "low",
                    "confidence": 0.7
                },
                {
                    "name": "Comprehensive Solution",
                    "description": "Root cause analysis and fix",
                    "steps": ["Analyze root cause", "Design solution", "Implement", "Verify"],
                    "complexity": "high",
                    "risk": "medium",
                    "confidence": 0.6
                }
            ]
        }
    
    def select_best_strategy(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best strategy based on scoring"""
        def score_strategy(s):
            confidence = s.get("confidence", 0.5)
            risk_score = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(s.get("risk", "medium"), 0.7)
            complexity_score = {"low": 1.0, "medium": 0.8, "high": 0.6}.get(s.get("complexity", "medium"), 0.8)
            return confidence * 0.5 + risk_score * 0.3 + complexity_score * 0.2
        
        return max(strategies, key=score_strategy)

class PhaseManager:
    """Manages multi-phase workflow for complex problem solving"""
    
    def __init__(self, problem_statement: str, total_steps: int):
        self.problem_statement = problem_statement
        self.total_steps = total_steps
        self.current_phase = PHASE_INVESTIGATION
        self.phase_history = []
        self.complexity = self._assess_complexity()
        self.step_allocation = self._allocate_steps()
        self.phase_start_step = 0
        self.phase_checkpoints = {}
        
        logger.info(f"[PHASE_MANAGER] Problem complexity: {self.complexity['level']}")
        logger.info(f"[PHASE_MANAGER] Complexity score: {self.complexity['score']}")
        logger.info(f"[PHASE_MANAGER] Step allocation: {self.step_allocation}")
    
    def _assess_complexity(self) -> dict:
        """Assess problem complexity using multiple indicators"""
        
        problem_lower = self.problem_statement.lower()
        
        indicators = {
            "multi_file": len(re.findall(r'\bfile[s]?\b', problem_lower)) > 2,
            "algorithm": any(kw in problem_lower for kw in 
                           ['algorithm', 'optimization', 'performance', 'complexity', 'efficient']),
            "edge_cases": any(kw in problem_lower for kw in 
                            ['edge case', 'boundary', 'corner case', 'special case']),
            "refactor": any(kw in problem_lower for kw in 
                          ['refactor', 'redesign', 'restructure', 'rewrite']),
            "debugging": any(kw in problem_lower for kw in 
                           ['bug', 'error', 'crash', 'fail', 'incorrect', 'fix']),
            "multiple_components": len(re.findall(r'\bclass\b|\bfunction\b|\bmethod\b', problem_lower)) > 3,
            "integration": any(kw in problem_lower for kw in 
                             ['integrate', 'interaction', 'between', 'across']),
            "backward_compat": any(kw in problem_lower for kw in 
                                  ['backward', 'compatibility', 'breaking', 'legacy'])
        }
        
        score = sum(indicators.values())
        
        # Determine complexity level
        if score >= 5:
            level = "HIGH"
        elif score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return {
            "level": level,
            "score": score,
            "indicators": indicators
        }
    
    def _allocate_steps(self) -> dict:
        """Allocate steps to each phase based on complexity"""
        
        if self.complexity["level"] == "HIGH":
            # High complexity: thorough investigation and validation
            allocation = {
                PHASE_INVESTIGATION: 0.30,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.40,
                PHASE_VALIDATION: 0.15
            }
        elif self.complexity["level"] == "MEDIUM":
            # Medium complexity: balanced approach
            allocation = {
                PHASE_INVESTIGATION: 0.25,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.45,
                PHASE_VALIDATION: 0.15
            }
        else:
            # Low complexity: streamlined workflow
            allocation = {
                PHASE_INVESTIGATION: 0.20,
                PHASE_PLANNING: 0.10,
                PHASE_IMPLEMENTATION: 0.55,
                PHASE_VALIDATION: 0.15
            }
        
        # Adjust based on specific indicators
        if self.complexity["indicators"].get("algorithm"):
            allocation[PHASE_PLANNING] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05
        
        if self.complexity["indicators"].get("edge_cases"):
            allocation[PHASE_VALIDATION] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05
        
        # Convert to actual step counts
        return {
            phase: max(int(ratio * self.total_steps), 10)  # Minimum 10 steps per phase
            for phase, ratio in allocation.items()
        }
    
    def should_transition(self, current_step: int, cot: 'EnhancedCOT') -> tuple[bool, str]:
        """Determine if phase should transition"""
        
        steps_in_phase = current_step - self.phase_start_step
        allocated_steps = self.step_allocation[self.current_phase]
        
        # Check if allocated steps for this phase are exhausted
        if steps_in_phase >= allocated_steps:
            next_phase = self._get_next_phase()
            if next_phase:
                return True, next_phase
        
        # Early transition conditions based on phase goals
        if self.current_phase == PHASE_INVESTIGATION:
            # Transition if we've done sufficient investigation
            if steps_in_phase >= 10 and len(cot.thoughts) >= 10:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-10:]]
                search_count = sum(1 for t in recent_tools if 'search' in t or 'get_file' in t)
                
                # If investigation tools used heavily and we have findings
                if search_count >= 6:
                    logger.info(f"[PHASE_MANAGER] Investigation sufficient ({search_count} search ops)")
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase
        
        elif self.current_phase == PHASE_PLANNING:
            # Transition when solution is approved
            if len(cot.thoughts) >= 2:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-5:]]
                if 'get_approval_for_solution' in recent_tools:
                    logger.info(f"[PHASE_MANAGER] Solution approved, transitioning to implementation")
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase
        
        elif self.current_phase == PHASE_IMPLEMENTATION:
            # Check if significant changes made and tests passing
            if steps_in_phase >= 15 and len(cot.thoughts) >= 15:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-15:]]
                edit_count = sum(1 for t in recent_tools if 'edit' in t or 'save' in t)
                test_count = sum(1 for t in recent_tools if 'test' in t or 'run' in t)
                
                # If we've made changes and run tests
                if edit_count >= 3 and test_count >= 2:
                    logger.info(f"[PHASE_MANAGER] Implementation complete ({edit_count} edits, {test_count} test runs)")
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase
        
        return False, self.current_phase
    
    def _get_next_phase(self) -> str:
        """Get the next phase in sequence"""
        phase_sequence = [
            PHASE_INVESTIGATION,
            PHASE_PLANNING,
            PHASE_IMPLEMENTATION,
            PHASE_VALIDATION
        ]
        
        try:
            current_index = phase_sequence.index(self.current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def transition_to_phase(self, new_phase: str, current_step: int):
        """Transition to a new phase"""
        old_phase = self.current_phase
        self.phase_history.append({
            "phase": old_phase,
            "start_step": self.phase_start_step,
            "end_step": current_step,
            "steps_used": current_step - self.phase_start_step
        })
        
        self.current_phase = new_phase
        self.phase_start_step = current_step
        
        logger.info("="*80)
        logger.info(f"[PHASE_MANAGER] PHASE TRANSITION: {old_phase} → {new_phase}")
        logger.info(f"[PHASE_MANAGER] {old_phase} used {current_step - self.phase_start_step} steps")
        logger.info(f"[PHASE_MANAGER] {new_phase} allocated {self.step_allocation[new_phase]} steps")
        logger.info("="*80)
    
    def get_phase_guidance(self) -> str:
        """Get guidance for current phase"""
        return PHASE_SPECIFIC_GUIDANCE.get(self.current_phase, "")
    
    def create_checkpoint(self, step: int, test_results: dict = None):
        """Save checkpoint for current phase"""
        self.phase_checkpoints[self.current_phase] = {
            "step": step,
            "test_results": test_results,
            "timestamp": time.time()
        }
    
    def get_progress_summary(self, current_step: int) -> str:
        """Get summary of progress across phases"""
        steps_in_phase = current_step - self.phase_start_step
        allocated = self.step_allocation[self.current_phase]
        progress_pct = (steps_in_phase / allocated * 100) if allocated > 0 else 0
        
        summary = f"""
        [PHASE: {self.current_phase}] 
        Progress: {steps_in_phase}/{allocated} steps ({progress_pct:.1f}%)
        Overall: Step {current_step}/{self.total_steps}
        """
        return summary.strip()
    
    def use_multi_phase_workflow(self) -> bool:
        """Determine if multi-phase workflow should be used"""
        return self.complexity["level"] in ["HIGH", "MEDIUM"]


def initialize_git_repository():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)

def configure_environment_variables():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", test_mode: bool = False, test_env = None):
    """Legacy interface wrapper for backwards compatibility."""
    global PROXY_SERVICE_URL, REPO_DIR, EXECUTION_TIMEOUT_SEC, PATCH_TIMEOUT_LIMIT, TEST_ENV, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    REPO_DIR = repo_dir
    if test_mode:
        EXECUTION_TIMEOUT_SEC = 1000
        PATCH_TIMEOUT_LIMIT = 400
        TEST_ENV = test_env


    sys.path.insert(0, repo_dir)


    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    initialize_git_repository()

    configure_environment_variables()

    try:
        problem_type = determine_problem_category(input_dict.get("problem_statement"))

        if problem_type == TASK_TYPE_REPAIR:
            result = handle_repair_task(input_dict)
        else:
            result = handle_creation_task(input_dict)
    except Exception as e:
        result = handle_repair_task(input_dict)

    os.system("git reset --hard")

    return result

def determine_problem_category(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": TASK_CLASSIFIER_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{build_directory_structure()}"}
            ]
            
            response = NetworkRequestHandler.make_request(messages, model=MODEL_QWEN)

            if response not in [TASK_TYPE_CREATE, TASK_TYPE_REPAIR]:
                retry += 1
            else:
                break
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1
        
        time.sleep(2)

    return response

def transform_instruction_text(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] not in '.,;:!?':
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    # Pattern to match ```text...``` blocks
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        
        return f'```text\n{processed_content}\n```'
    
    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def create_solution_multi_stage(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": MULTI_STEP_SOLUTION_TEMPLATE
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nIMPORTANT: If the problem shows examples with strings containing newlines, return a single string with '\\n' separators, not a list of strings.\n\nCRITICAL: Carefully analyze the examples to understand formatting and character preservation requirements. Preserve all characters exactly as shown in the problem examples.\n\nCRITICAL: When transformations are required, use appropriate techniques to preserve special characters:\n1. Use temporary placeholders for special characters during transformation if needed\n2. Follow the problem's specific requirements for formatting and alignment\n3. Perform the required transformation\n4. Restore the original characters and clean up as needed\n\nThis ensures all characters are preserved in the correct positions.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]
    while retry < 10:
        try:
            code_response = NetworkRequestHandler.make_request(code_generation_messages, model=MODEL_QWEN)
            logger.info("Step 1 - Code Generation completed")
            
            # Step 5: Infinite Loop Check and Validation
            loop_check_messages = [
                {
                    "role": "system",
                    "content": LOOP_VALIDATION_TEMPLATE
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nOriginal stub with docstrings:\n{code_skeleton}\n\nAnalyze this code for potential issues and provide a corrected version if any issues are found. Ensure module-level constants (if any) are preserved or added based on return values in docstrings. Return ONLY the final Python code."
                }   
            ]
            
            loop_check_response = NetworkRequestHandler.make_request(loop_check_messages, model=MODEL_QWEN)
            logger.info("Step 2 - Infinite Loop Check completed")

            # Clean up the final response (use loop check response as it's the final validated version)
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()
            
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in create_solution_multi_stage: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def create_initial_implementation(problem_statement: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation")
            
            solution = create_solution_multi_stage(problem_statement, code_skeleton)
            
            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": INIT_SOLUTION_TEMPLATE
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\n"""
                    }
                ]
                
                response = NetworkRequestHandler.make_request(messages, model=MODEL_QWEN)
                
                # Clean up the response
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                logger.info("Generated initial solution successfully using fallback approach")
                return solution
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def generate_test_signature_file(testcode_response: str) -> str:
    """
    Generate a signature file by replacing all test function bodies with 'pass' statements.
    Preserves imports, class definitions, and function signatures.
    
    Args:
        testcode_response: The full test code with implementations
    
    Returns:
        Test signature file with function bodies replaced by 'pass'
    """
    import re
    
    try:
        # Clean up the test code response
        test_code = testcode_response.strip()
        if test_code.startswith('```python'):
            test_code = test_code[9:]
        if test_code.startswith('```'):
            test_code = test_code[3:]
        if test_code.endswith('```'):
            test_code = test_code[:-3]
        test_code = test_code.strip()
        
        # Parse the test code to replace function bodies
        lines = test_code.split('\n')
        result_lines = []
        test_method_count = 0
        max_test_methods = 5
        function_indent = 0
        skip_until_next_def = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip filename markers (exclude them from output)
            if stripped.endswith('.py') and ' ' not in stripped and len(stripped) > 3 and not stripped.startswith('#'):
                i += 1
                continue
            
            # Check if this is a function definition (test_ or helper functions)
            if stripped.startswith('def '):
                # Check if this is a test method
                is_test_method = 'def test_' in stripped
                
                # If we've already added 5 test methods, skip remaining test methods
                if is_test_method:
                    if test_method_count >= max_test_methods:
                        # Skip this test method entirely
                        function_indent = len(line) - len(line.lstrip())
                        skip_until_next_def = True
                        i += 1
                        continue
                    else:
                        test_method_count += 1
                
                # Add the function definition line
                result_lines.append(line)
                
                # Calculate indentation
                function_indent = len(line) - len(line.lstrip())
                
                # Add pass statement on next line with proper indentation
                result_lines.append(' ' * (function_indent + 4) + 'pass')
                
                # Skip all lines until we're back at the same or lower indentation level
                skip_until_next_def = True
                i += 1
                continue
            
            # If we're skipping function body
            if skip_until_next_def:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else function_indent + 4
                
                # If we're back at function level or less, stop skipping
                if stripped and current_indent <= function_indent:
                    skip_until_next_def = False
                    # Process this line normally (might be another def, class, import, etc.)
                    continue
                else:
                    # Skip this line (it's part of the function body)
                    i += 1
                    continue
            
            # Keep imports, class definitions, comments, blank lines, etc.
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or 
                stripped.startswith('class ') or
                stripped.startswith('#') or
                stripped.startswith('@') or  # decorators
                not stripped):  # blank lines
                result_lines.append(line)
            
            i += 1
        
        signature_code = '\n'.join(result_lines)
        logger.info(f"Generated test signature file successfully (included {test_method_count} test methods)")
        return signature_code
        
    except Exception as e:
        logger.error(f"Error generating test signature file: {e}")
        return testcode_response  # Return original if signature generation fails

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = NetworkRequestHandler.make_request(test_generation_messages, model=MODEL_QWEN)
            logger.info("Step 1 - Testcase Generation completed")
            
            # Generate signature file from testcode_response
            signature_file = generate_test_signature_file(testcode_response)
            logger.info("Generated test signature file", extra={"signature_file": signature_file})

            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_REVIEW_WITH_MULTI_STEP_REASONING_PROMPT.replace("{TEST_SIGNATURE_FILE}", signature_file)
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
                }   
            ]
            
            testcode_checked_response = NetworkRequestHandler.make_request(testcases_check_messages, model=MODEL_QWEN)
            logger.info("Step 2 - Testcase check completed")

            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    logger.error("Multi-step reasoning testcase generation failed after 10 retries")
    return ""

def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")
            
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {
                        "role": "system",
                        "content": INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                
                response = NetworkRequestHandler.make_request(messages, model=MODEL_QWEN)
                
                # Clean up the response
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
            
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            retry += 1
            time.sleep(2)
    
    logger.error("Failed to generate test cases after 10 retries")
    return ""

def parse_and_save_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    import re
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()
        
        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):  # Not a comment
            
            # Write the previous file if we have one
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                # Create directory if needed (for subdirectories)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Join content and remove empty lines at start/end
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            # Start new file
            current_filename = stripped_line
            current_content = []
        else:
            # This line is content for the current file
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files

def handle_creation_task(input_dict):
    global REPAIR_MAX_STEPS

    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = transform_instruction_text(problem_statement)
    print(problem_statement)

    code_skeleton = retrieve_code_structure()
    start_time = time.time()
    initial_solution = create_initial_implementation(problem_statement, code_skeleton)
    print(initial_solution)

    # Extract and write files from the initial solution
    created_files = parse_and_save_files(initial_solution)
    print(f"Created or Updated {len(created_files)} files: {created_files}")


    
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    print(test_cases)
    # Extract and write files from test cases
    test_files = parse_and_save_files(test_cases)
    print(f"Created or Updated {len(test_files)} files: {test_files}")

    timeout = EXECUTION_TIMEOUT_SEC - (time.time()-start_time) - 60
    
    REPAIR_MAX_STEPS = 120
    patch = repair_task_execution_flow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"pytest",
        test_runner_mode="FILE",
        is_of_create_task=True,
    )

    if patch is None:
        parse_and_save_files(initial_solution)

    tool_executor = ToolExecutionManager()

    # remove pycache directories
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
    patch = tool_executor.get_final_git_patch()
    return patch

def retrieve_code_structure() -> str:
    # Initialize the result string
    result = ""
    
    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"
    
    return result

def build_directory_structure(start_path: str = '.') -> str:

    tree_lines = []
    
    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            
            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            
            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                
                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                
                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                
                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
            except PermissionError:
                # Handle directories we can't read
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")
                
        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")
    
    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def locate_readme_file(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)
    
    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None

def locate_test_executor(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()
        
        response = NetworkRequestHandler.make_request([
            {"role": "system", "content": TEST_RUNNER_LOCATOR_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=MODEL_DEEPSEEK)
        return response.strip() or "pytest"
    except Exception as e:
        logger.error(f"Error finding test runner: {e}")
        return "pytest"

def convert_path_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')

def sanitize_file_path(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path

def determine_test_execution_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()
        
        response = NetworkRequestHandler.make_request([
            {"role": "system", "content": TEST_MODE_DETECTOR_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=MODEL_DEEPSEEK)
        return response.strip() or "FILE"
    except Exception as e:
        logger.error(f"Error determining test runner mode: {e}")
        return "FILE"

def enumerate_test_functions(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    # Default count for errors
    default_count = 0
    
    try:
        # Read file content
        file_handle = open(file_path, 'r', encoding='utf-8')
        file_content = file_handle.read()
        file_handle.close()
        
        # Find test function pattern
        import re
        pattern = r'^\s*def\s+test_\w+'
        flags = re.MULTILINE
        matches = re.findall(pattern, file_content, flags)
        
        # Return count of matches
        test_count = len(matches)
        return test_count
    
    except FileNotFoundError:
        return default_count
    except UnicodeDecodeError:
        return default_count

def fetch_test_configuration():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    test_files.sort(key=len)

    for path in test_files:
        if enumerate_test_functions(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        print(f"no test file found")
        return "pytest", "FILE"

    print(f"test_file_path: {test_file_path}")
    readme_file_path = locate_readme_file(test_file_path, '.')
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = locate_test_executor(readme_file_path)
        test_runner_mode = determine_test_execution_mode(test_runner)
    else:
        print("No README found, using default pytest")

    return test_runner, test_runner_mode

def handle_repair_task(input_dict: Dict[str, Any]):
    """Main entry point for task processing and code modification.

    Parameters
    ----------
    input_dict : dict
        Configuration dictionary containing the task specification.
        Required key: 'problem_statement' with task details.
        Optional keys: 'run_id', 'instance_id' for tracking purposes.
    """
    global run_id
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(EXECUTION_TIMEOUT_SEC)))
    
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split('/')[-1]
    repod_path = repo_path[:-len(repod_dir)-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)

    configure_environment_variables()
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd} and environ:{os.environ}")
    
    test_runner, test_runner_mode = fetch_test_configuration()
    print(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")

    try:
        logger.info(f"current files:{os.listdir()}")
        logger.info(f"packages installed:{subprocess.check_output(['pip','list']).decode('utf-8')}")
        logger.info(f"About to execute workflow...")
        patch_text= repair_task_execution_flow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            instance_id="",
            test_runner=test_runner,
            test_runner_mode=test_runner_mode
        )
        logger.info(f"workflow execution completed, patch length: {len(patch_text)}")

        os.system("git reset --hard")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"[CRITICAL] Exception in task processing: {error_info}")
        logs.append(error_info)
    finally:
        os.chdir(cwd)

    print(f"[CRITICAL] task processor returning patch length: {len(patch_text)}")
    print(f"[CRITICAL] patch: {patch_text}")
    return patch_text

def repair_task_execution_flow(problem_statement: str, *, timeout: int, run_id_1: str, instance_id: str = "", \
    test_runner: str = "pytest", test_runner_mode: str = "FILE", is_of_create_task: bool = False) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id=run_id_1
    cot=ChainOfThoughtProcessor()
    tool_executor=RepairTaskToolExecutor(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "approve_test_code",
            "get_functions",
            "get_classes",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_repo_tests",
            "run_code",
            "insert_test_function",  # NEW: Position-aware test insertion
            "save_checkpoint",       # NEW: Save state for rollback
            "restore_checkpoint",    # NEW: Restore previous state
            "list_checkpoints",      # NEW: View available checkpoints
            "bash_tool",
            "fetch_url_content",
            "apply_code_edit",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )

    strategic_planner = StrategicPlanner()
    logger.info("========== PHASE 1: STRATEGIC PLANNING ==========")
    strategies = strategic_planner.generate_strategies(problem_statement)
    selected_strategy = strategic_planner.select_best_strategy(strategies["strategies"])
    logger.info(f"[PLANNER] Selected strategy: {selected_strategy['name']}")
    strategy_guidance = f"\n\nStrategic Plan: {selected_strategy.get('name', 'Default')} - {selected_strategy.get('description', 'Standard approach')}\n\n"

    # Initialize phase manager for complex problems
    phase_manager = PhaseManager(problem_statement, REPAIR_MAX_STEPS)
    use_multi_phase = phase_manager.use_multi_phase_workflow()
    
    if use_multi_phase:
        logger.info("="*80)
        logger.info("[MULTI-PHASE WORKFLOW] Complex problem detected - using phased approach")
        logger.info(f"[MULTI-PHASE WORKFLOW] Complexity indicators: {phase_manager.complexity['indicators']}")
        logger.info("="*80)
    else:
        logger.info("[WORKFLOW] Using standard single-phase workflow for simple problem")
    

    logger.info(f"Starting main agent execution...")
    system_prompt = REPAIR_SYSTEM_INSTRUCTIONS.format(tools_docs=tool_executor.get_tool_docs(),format_prompt=RESPONSE_FORMAT_GUIDE)
    instance_prompt = REPAIR_INSTANCE_TEMPLATE.format(problem_statement=problem_statement) + strategy_guidance

    if is_of_create_task:
        instance_prompt += f"""There are initial implementation and test cases already created for the problem statement.
Your task is to fix any issues in the implementation and/or test cases so that all test cases pass successfully by following the strategic plan provided.
"""
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {REPAIR_MAX_STEPS} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
    for step in range(REPAIR_MAX_STEPS):
        logger.info(f"Execution step {step + 1}/{REPAIR_MAX_STEPS}")
        
        if use_multi_phase and step > 0:
            should_transition, new_phase = phase_manager.should_transition(step, cot)
            if should_transition:
                phase_manager.transition_to_phase(new_phase, step)
        
        # Log phase progress
        if use_multi_phase and step % 10 == 0:
            logger.info(phase_manager.get_progress_summary(step))
        
        if time.time() - start_time > timeout:
            cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step="global timeout reached",tool_identifier="",tool_parameters={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())

        # Add phase-specific guidance if using multi-phase workflow
        if use_multi_phase:
            phase_guidance = phase_manager.get_phase_guidance()
            messages.append({"role": "system", "content": phase_guidance})

        messages.append({"role": "system", "content": HALT_DIRECTIVE})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": AVOID_REPETITION_MSG.format(previous_response=f"tool_identifier:{last_thought.tool_identifier}\n tool_parameters:{last_thought.tool_parameters}")})
    
        try:
            # Use temperature=0.0 for deterministic reasoning in debugging tasks
            reasoning_step, tool_identifier, tool_parameters,response_text,attempt_count,error_tracking,messages = NetworkRequestHandler.inference(messages, model=MODEL_GLM, run_id=run_id, temperature=0.0)
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step=error_msg,tool_identifier="",tool_parameters={},observation="",is_error=True,raw_response=response_text,attempt_count=attempt_count),inference_error_counter=error_tracking,request_data=messages)
            break
        
        logger.info(f"About to execute operation: {tool_identifier}")
       
        try:
            logger.info(f"reasoning_step: {reasoning_step}\ntool_identifier: {tool_identifier}\ntool_parameters: {tool_parameters}\n")
            if '"' in tool_identifier or "'" in tool_identifier:
                tool_identifier=tool_identifier.replace('"','')
                tool_identifier=tool_identifier.replace("'","")
                
            next_observation = tool_executor.get_tool(tool_identifier)(**tool_parameters) if tool_parameters else tool_executor.get_tool(tool_identifier)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step=reasoning_step,tool_identifier=tool_identifier,tool_parameters=tool_parameters,observation=next_observation,is_error=False,raw_response=response_text,attempt_count=attempt_count,inference_error_counter=error_tracking,request_data=messages))

            # Create checkpoint after key successful actions
            if use_multi_phase and tool_identifier in ['run_repo_tests', 'apply_code_edit', 'get_approval_for_solution']:
                # Extract test results if available
                test_results = {}
                if 'passed' in str(next_observation).lower() or 'failed' in str(next_observation).lower():
                    # Simple parsing of test results
                    obs_str = str(next_observation)
                    test_results['observation'] = obs_str[:200]  # First 200 chars
                
                phase_manager.create_checkpoint(step, test_results)
                logger.debug(f"[PHASE_MANAGER] Checkpoint created at step {step} after {tool_identifier}")

        except ToolExecutionManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step=reasoning_step,tool_identifier=tool_identifier,tool_parameters=tool_parameters,observation=error_msg,is_error=True,raw_response=response_text,attempt_count=attempt_count,inference_error_counter=error_tracking,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step=reasoning_step,tool_identifier=tool_identifier,tool_parameters=tool_parameters,observation=error_msg,is_error=True,raw_response=response_text,attempt_count=attempt_count,inference_error_counter=error_tracking,request_data=messages))
            continue
        
        if tool_identifier == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        # This happens if we exit the loop without breaking (reached MAX_STEPS)
        cot.add_action(ChainOfThoughtProcessor.Action(reasoning_step="global timeout reached",tool_identifier="",tool_parameters={},observation="",is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({REPAIR_MAX_STEPS})")
        # If we were in the context of creating a task, we should return None
        if is_of_create_task:
            logger.info(f"[CRITICAL] create task - couldn't fix within max steps")
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_executor.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    # Generic verification gate: apply to a clean repo and re-run tests before finishing
    # try:
    #     # Reset to clean state and apply patch in-memory to validate
    #     subprocess.run(["git", "reset", "--hard"], check=False, capture_output=True, text=True)
    #     subprocess.run(["git", "clean", "-fdx"], check=False, capture_output=True, text=True)

    #     if patch.strip():
    #         # git diffs typically use a/ and b/ prefixes -> strip 1 leading component
    #         apply = subprocess.run(["git", "apply", "-p1", "-"], input=patch, text=True, capture_output=True)
    #         if apply.returncode != 0:
    #             logger.error("git apply failed: %s", apply.stderr.strip())
    #             # If apply fails, return the patch anyway for visibility
    #             return patch

    #     # Prefer running the repository's canonical tests over a narrow subset
    #     last_tests = []
    #     if hasattr(tool_executor, "_last_tests") and tool_executor._last_tests:
    #         last_tests = tool_executor._last_tests
    #     else:
    #         # Fallback: run full test tree when available
    #         last_tests = ["./tests"]

    #     # Filter out non-existent test targets after clean reset to avoid import errors
    #     try:
    #         if last_tests:
    #             valid_tests = []
    #             for t in last_tests:
    #                 if test_runner == "pytest":
    #                     candidate = t
    #                 else:
    #                     # Convert to a relative file path and ensure it exists
    #                     candidate = sanitize_file_path(t, os.getcwd(), test_runner) + ".py"
    #                 if os.path.exists(candidate):
    #                     valid_tests.append(t)
    #             last_tests = valid_tests if valid_tests else ["./tests"]
    #     except Exception:
    #         # On any error, fall back to running the repository's tests
    #         last_tests = ["./tests"]

    #     # First run: targeted or full tests
    #     verification_output = tool_executor.run_repo_tests(last_tests)
    #     print(verification_output)
    #     if "FAILED" in verification_output or "errors=" in verification_output or "ERROR" in verification_output:
    #         # Try full suite once more if we only ran a subset
    #         try:
    #             full_targets = ["./tests"]
    #             if last_tests != full_targets:
    #                 verification_output2 = tool_executor.run_repo_tests(full_targets)
    #                 print(verification_output2)
    #                 if ("FAILED" in verification_output2 or "errors=" in verification_output2 or "ERROR" in verification_output2):
    #                     logger.error("Verification after full-suite run failed. Suppressing patch output.")
    #                     return ""
    #             else:
    #                 logger.error("Verification after clean apply failed. Suppressing patch output.")
    #                 return ""
    #         except Exception:
    #             logger.error("Verification retry failed. Suppressing patch output.")
    #             return ""

    #     # Second pass to guard against flakiness
    #     # verification_output_2 = tool_executor.run_repo_tests(last_tests)
    #     # print(verification_output_2)
    #     # if "FAILED" in verification_output_2 or "errors=" in verification_output_2 or "ERROR" in verification_output_2:
    #     #     logger.error("Second verification run failed (flaky). Not finishing as success.")
    #     #     return patch
    # except Exception as e:
    #     logger.error("Post-patch verification error: %s", str(e))
    #     return patch

    return patch