from typing import List

from validator.challenge.challenge_types import CodegenResponse, ValidationResult
from .abstract_grader import GraderInterface

class TrueskillGrader(GraderInterface):
    def grade(self, patches: List[CodegenResponse]) -> List[ValidationResult]:
        return []