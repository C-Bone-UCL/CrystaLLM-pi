# AI Usage Disclosure

This document records how generative AI was used to build and maintain this codebase. Assistance ranged from code completion early on to supervised agent-driven edits later. I recheck and validate all the code: every change, AI-assisted or not, was reviewed, tested and validated by me before it entered the repository, and the architectural and design decisions are mine.

## 1. Tools Used

Used in sequence: Copilot early on, Codex for about two months, Claude Code since.

*   **GitHub Copilot:** in-editor code completion and agents.
*   **Codex (OpenAI):** agent-driven edits with GPT-5.5 and 5.6.
*   **Claude Code (Anthropic):** agent-driven edits, using Claude 4- and 5-family models (Sonnet, Opus, Fable).

## 2. Where AI Was Applied and What It Did

*   **Code:** scaffolding boilerplate and repetitive patterns, refactoring and parallelisation help, and improving syntax and variable names. Later passes used Claude Code for larger supervised jobs, such as standardising docstrings and type annotations across the package.
*   **Documentation:** drafting docstrings from existing logic, the MkDocs site pages, and the README restructure. I edited and fact-checked the output.
*   **Tests:** suggesting edge cases and test scenarios, and drafting some test scaffolding. I decide what gets tested, review every test, and run the suites myself.
*   **Planning:** untracked working documents (plans, audits, proposals) used to organise the work.

## 3. Verification and Accountability

Responsibility for the correctness, scientific validity and stability of this project is mine, the author's.

*   **Review:** I read and validate every AI-assisted diff before it is committed. Nothing merges unreviewed.
*   **Testing:** the local and containerised test suites run against every change, by me and in CI.
*   **Output Validation:** model outputs and pipeline behaviour are rechecked against expected behaviour.
*   **Design:** the model families, the registry, the API surface and the testing strategy are my decisions.

## 4. Boundaries

*   Agent-driven edits run only under my prompts and review; no AI-generated change lands without my validation.
*   No reported result, metric or benchmark number is AI-generated. All come from runs I executed and checked.
