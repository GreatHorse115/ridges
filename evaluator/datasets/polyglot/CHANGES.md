# Changes to the Polyglot dataset

The Ridges team has made some changes to the original Polyglot problem set, which was sourced originally from [this repository](https://github.com/Aider-AI/polyglot-benchmark/tree/main/python/exercises/practice). The reason for these changes were because some of the questions are believed (by us) to be practically unsolvable due to errors in the problem statement or tests that do not correlate exactly with the instructions. All changes made to the problem set are documented here for transparency.

<br>
<br>

## affine-cipher

No changes. There are some links that are impossible for the agent to follow. This will be resolved in a future version of our sandbox, where we provide restricted Internet access.

## beer-song

Added typing to `recite()`, as it is unclear whether the function should return a list of strings, or a single string, seperated by newlines.

## book-store

Added typing to `total()`, as it is unclear what format of input the function should expect. Added a comment (`# in cents`) to indicate the format of the output, as the instructions do not specify whether the solution should return an answer as a decimal number of dollars, or a decimal number of cents. In fact, the instructions lean towards decimal dollars, whereas the tests use integer cents.

## bottle-song

Added typing to `recite()`, as it is unclear whether the function should return a list of strings, or a single string, seperated by newlines.

## bowling

No changes. The instructions provide enough typing information, as such, it is not necessary to include it in the Python file.

## connect

No changes.

## dominoes

Added typing to `can_chain()`.