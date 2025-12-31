"""Jinja2-based prompt template loader for Oracle and Librarian agents.

This service loads prompt templates from the backend/prompts/ directory and renders
them with context variables. It supports hot-reload (no caching) so prompts can be
edited without restarting the server.

Fallback inline prompts are provided for bootstrapping when the prompts directory
doesn't exist yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2

logger = logging.getLogger(__name__)

# Default prompts directory relative to this file
# backend/src/services/prompt_loader.py -> backend/prompts/
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PromptLoaderError(Exception):
    """Raised when a prompt cannot be loaded."""

    pass


class PromptLoader:
    """Load and render Jinja2 prompt templates.

    Supports:
    - Loading templates from filesystem (backend/prompts/)
    - Fallback to inline prompts when directory doesn't exist
    - Hot-reload: templates are reloaded on every call (no caching)
    - Jinja2 rendering with context variables

    Example:
        >>> loader = PromptLoader()
        >>> system_prompt = loader.load("oracle/system.md", {"project_id": "my-project"})
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates.
                        Defaults to backend/prompts/ relative to this file.
        """
        self.prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

        # Check if prompts directory exists
        if self.prompts_dir.is_dir():
            # Create Jinja2 environment without caching for hot-reload
            # auto_reload=True ensures templates are reloaded on each access
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
                autoescape=False,  # Prompts are markdown, not HTML
                auto_reload=True,  # Always reload templates (no caching)
                keep_trailing_newline=True,  # Preserve trailing newlines in templates
            )
            logger.debug(
                "PromptLoader initialized with filesystem templates",
                extra={"prompts_dir": str(self.prompts_dir)},
            )
        else:
            self.env = None
            logger.warning(
                "Prompts directory not found, using inline fallbacks",
                extra={"prompts_dir": str(self.prompts_dir)},
            )

    def load(self, path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Load and render a prompt template.

        Args:
            path: Relative path to the template file (e.g., "oracle/system.md").
            context: Dictionary of variables to render into the template.

        Returns:
            The rendered prompt string.

        Raises:
            PromptLoaderError: If the template cannot be loaded or rendered.
        """
        context = context or {}

        # Try filesystem first
        if self.env is not None:
            try:
                template = self.env.get_template(path)
                rendered = template.render(**context)
                logger.debug(
                    "Loaded prompt from filesystem",
                    extra={"path": path, "context_keys": list(context.keys())},
                )
                return rendered
            except jinja2.TemplateNotFound:
                logger.debug(
                    "Template not found in filesystem, trying inline fallback",
                    extra={"path": path},
                )
            except jinja2.TemplateError as e:
                logger.error(
                    "Failed to render template",
                    extra={"path": path, "error": str(e)},
                )
                raise PromptLoaderError(f"Failed to render template {path}: {e}") from e

        # Fallback to inline prompts
        return self._get_inline_prompt(path, context)

    def _get_inline_prompt(self, path: str, context: Dict[str, Any]) -> str:
        """Get inline fallback prompt for bootstrapping.

        These minimal prompts are used when the prompts/ directory doesn't exist yet.
        They provide basic functionality until the full prompt files are created.

        Args:
            path: Relative path to the template (e.g., "oracle/system.md").
            context: Dictionary of variables to render into the template.

        Returns:
            The rendered inline prompt string.

        Raises:
            PromptLoaderError: If no inline fallback exists for the path.
        """
        inline_prompts: Dict[str, str] = {
            "oracle/system.md": """# Oracle System Prompt

You are the **Oracle**, an AI project manager that helps developers understand and navigate their codebase.

## Your Role

You help developers by:
- **Code Understanding**: Analyzing code structure, patterns, and relationships
- **Documentation Navigation**: Finding and synthesizing information from project documentation
- **Development Memory**: Recalling past decisions and context from development threads
- **Research**: Gathering external information when internal sources are insufficient

## Available Tools

You have access to tools for:
- Searching code and finding definitions
- Reading and writing documentation in the vault
- Recording decisions to memory threads
- Searching the web for external information

## Citation Requirements

Always cite your sources using these formats:
- **Code**: `[filename:line_number]`
- **Documentation**: `[note-path]`
- **Threads**: `[thread:thread_id]`
- **Web**: `[url]`

## Response Guidelines

1. Use tools proactively to gather information before answering
2. Be honest when you cannot find relevant context
3. Acknowledge uncertainty - do not fabricate citations
4. Suggest follow-up questions when appropriate

## Current Context

- **Project**: {{ project_id or 'Not specified' }}
- **User**: {{ user_id or 'Unknown' }}
""",
            "librarian/system.md": """# Librarian System Prompt

You are the **Librarian**, a specialized documentation organization agent.

## Your Role

You specialize in organizing, restructuring, and maintaining the documentation vault:

1. **File Organization**: Moving notes to appropriate folders
2. **Index Creation**: Generating index pages that summarize and link to related notes
3. **Wikilink Maintenance**: Ensuring links between notes remain valid
4. **Structure Recommendations**: Suggesting folder structures that improve navigability

## Available Tools

- **vault_read**: Read a markdown note
- **vault_write**: Create or update markdown notes
- **vault_search**: Search vault for related content
- **vault_list**: List notes in a folder
- **vault_move**: Move or rename notes (updates wikilinks automatically)
- **vault_create_index**: Create an index.md file for a folder

## Organization Principles

1. Related notes should be in the same folder
2. Create index files for folders with more than 3 notes
3. Follow existing naming conventions
4. Prefer small, incremental changes

## Task Focus

Complete only vault organization tasks. Report your actions clearly back to the Oracle.

## Project Context

- **Project**: {{ project_id or 'Not specified' }}
""",
            "oracle/synthesis.md": """# Synthesis Prompt

Synthesize the gathered context into a clear, well-structured answer.

## Available Context

{{ context_summary or 'No context gathered.' }}

## Guidelines

1. Lead with the direct answer to the user's question
2. Cite all sources using the appropriate format
3. Be concise but thorough
4. Acknowledge any gaps in the available information
5. Suggest follow-up questions if appropriate
""",
            "oracle/compression.md": """# Compression Prompt

Compress the following conversation context to fit within token limits while preserving essential information.

## Current Context Size

- Tokens used: {{ tokens_used or 'Unknown' }}
- Token limit: {{ token_limit or '16000' }}

## Guidelines

1. Preserve key facts, decisions, and citations
2. Remove redundant or verbose explanations
3. Keep tool call results summarized
4. Maintain conversation coherence
""",
            "oracle/no_context.md": """# No Context Response

I searched the available sources but couldn't find relevant information to answer your question.

## What I Tried

{{ search_summary or 'Searched code, documentation, and memory threads.' }}

## Suggestions

1. Try rephrasing your question with different keywords
2. Check if the information might be in a specific file path
3. Consider whether this is external knowledge that would require web search

Would you like me to try a different search approach?
""",
            "librarian/organize.md": """# Organization Task

## Task Description

{{ task_description or 'Organize the documentation vault.' }}

## Current Structure

{{ current_structure or 'Unknown - use vault_list to explore.' }}

## Instructions

1. Analyze the current vault structure
2. Identify notes that should be reorganized
3. Create index files where appropriate
4. Move files using vault_move to preserve wikilinks
5. Report all changes made
""",
            "librarian/summarize.md": """# Librarian Summarization Task

You are the **Librarian**, a specialized summarization agent. Create a concise, well-cited summary.

## Task

{{ task }}

## Target Length

Approximately {{ max_tokens or 1000 }} tokens maximum.

## Source Information

Summarizing **{{ source_count or 'unknown' }}** source(s) of type: **{{ primary_type or 'mixed' }}**

## Guidelines

1. **Cite Everything**: Use [[wikilinks]] for vault notes, [thread:id] for threads, [code:path] for code
2. **Preserve Key Info**: Include important facts, decisions, and references
3. **Structure Well**: Use headers, bullets, and the format below
4. **Be Concise**: Avoid repetition, merge related info

## Output Format

```markdown
# Summary: [Topic]

## Key Documents
- [[source1]] - Brief description
- [[source2]] - Brief description

## Main Points
1. First major point (from [[source]])
2. Second major point (from [[source]])

## Sources Cited
All information from: [[source1]], [[source2]], ...
```

## Important

- DO NOT hallucinate or add information not in sources
- DO NOT fabricate citations
- DO cite every piece of information
- Provide ONLY the markdown summary, no preamble
""",
        }

        template_str = inline_prompts.get(path)

        if template_str is None:
            logger.warning(
                "No inline fallback for prompt path",
                extra={"path": path, "available": list(inline_prompts.keys())},
            )
            raise PromptLoaderError(
                f"Prompt not found: {path}. "
                f"Available inline prompts: {list(inline_prompts.keys())}"
            )

        try:
            template = jinja2.Template(template_str)
            rendered = template.render(**context)
            logger.debug(
                "Loaded inline fallback prompt",
                extra={"path": path, "context_keys": list(context.keys())},
            )
            return rendered
        except jinja2.TemplateError as e:
            logger.error(
                "Failed to render inline template",
                extra={"path": path, "error": str(e)},
            )
            raise PromptLoaderError(
                f"Failed to render inline template {path}: {e}"
            ) from e

    def list_available(self) -> Dict[str, list[str]]:
        """List available prompt templates.

        Returns:
            Dictionary with 'filesystem' and 'inline' keys containing lists
            of available template paths.
        """
        result: Dict[str, list[str]] = {
            "filesystem": [],
            "inline": [
                "oracle/system.md",
                "oracle/synthesis.md",
                "oracle/compression.md",
                "oracle/no_context.md",
                "librarian/system.md",
                "librarian/organize.md",
                "librarian/summarize.md",
            ],
        }

        if self.prompts_dir.is_dir():
            for md_file in self.prompts_dir.rglob("*.md"):
                relative_path = md_file.relative_to(self.prompts_dir).as_posix()
                result["filesystem"].append(relative_path)

        return result


__all__ = ["PromptLoader", "PromptLoaderError", "DEFAULT_PROMPTS_DIR"]
