"""Vault persistence service for Deep Research projects."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...models.research import ResearchState, ResearchReport, ResearchSource

logger = logging.getLogger(__name__)


class ResearchVaultPersister:
    """Persists research projects to the user's vault.

    This service takes a completed ResearchState and saves all artifacts
    to a structured folder in the user's vault:

        research/{research_id}/
            index.md          - Research project hub with wikilinks
            brief.md          - Original question and scope
            report.md         - The actual research report
            sources.md        - Bibliography with scores
            methodology.md    - Search queries, decisions, metrics
            notes/
                source-001-{slug}.md  - Per-source detailed notes
                source-002-{slug}.md
                ...

    Templates are loaded from backend/prompts/research/vault/.
    """

    RESEARCH_FOLDER = "research"

    def __init__(self, vault_path: str | Path):
        """Initialize the persister.

        Args:
            vault_path: Path to the user's vault directory.
        """
        self.vault_path = Path(vault_path)
        # Navigate from backend/src/services/research/ to backend/prompts/research/vault/
        # __file__ = backend/src/services/research/vault_persister.py
        # .parent = backend/src/services/research/
        # .parent.parent = backend/src/services/
        # .parent.parent.parent = backend/src/
        # .parent.parent.parent.parent = backend/
        self.templates_path = (
            Path(__file__).parent.parent.parent.parent / "prompts" / "research" / "vault"
        )
        self._env: Optional[Environment] = None

    @property
    def env(self) -> Environment:
        """Lazy-load Jinja2 environment."""
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader(str(self.templates_path)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        return self._env

    def _get_research_folder(self, research_id: str) -> Path:
        """Get or create the research project folder.

        Args:
            research_id: Unique identifier for the research project.

        Returns:
            Path to the research folder.
        """
        folder = self.vault_path / self.RESEARCH_FOLDER / research_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "notes").mkdir(exist_ok=True)
        return folder

    def _render_template(self, template_name: str, **context) -> str:
        """Render a template with context.

        Args:
            template_name: Name of the template file (e.g., "index_template.md").
            **context: Variables to pass to the template.

        Returns:
            Rendered template string.
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug.

        Args:
            text: Text to slugify.

        Returns:
            URL-safe slug (lowercase, alphanumeric + hyphens).
        """
        slug = text.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug[:50].strip("-")

    def persist(self, state: ResearchState) -> str:
        """Persist a completed research project to the vault.

        Creates the full folder structure with all artifacts:
        - index.md: Hub file with wikilinks
        - brief.md: Research brief
        - report.md: Full report content
        - sources.md: Bibliography
        - methodology.md: Research methodology
        - notes/*.md: Individual source notes

        Args:
            state: The completed ResearchState to persist.

        Returns:
            Path to the index.md file.

        Raises:
            ValueError: If state has no report (research incomplete).
        """
        if not state.report:
            raise ValueError("Cannot persist research without a report")

        folder = self._get_research_folder(state.research_id)
        report = state.report

        # Common context for all templates
        common_ctx = {
            "research_id": state.research_id,
            "title": report.title,
            "created_date": state.started_at.isoformat()[:10],
            "completed_date": (
                state.completed_at.isoformat()[:10] if state.completed_at else ""
            ),
        }

        # 1. Write index.md (hub file)
        self._write_index(folder, state, report, common_ctx)

        # 2. Write brief.md
        self._write_brief(folder, state, common_ctx)

        # 3. Write report.md (the actual report content)
        self._write_report(folder, report)

        # 4. Write sources.md (bibliography)
        self._write_sources(folder, report, common_ctx)

        # 5. Write methodology.md
        self._write_methodology(folder, state, common_ctx)

        # 6. Write individual source notes
        for source in report.sources:
            self._write_source_note(folder, source, common_ctx)

        logger.info(
            "Persisted research project to vault",
            extra={
                "research_id": state.research_id,
                "folder": str(folder),
                "sources_count": len(report.sources),
            },
        )
        return str(folder / "index.md")

    def _write_index(
        self,
        folder: Path,
        state: ResearchState,
        report: ResearchReport,
        ctx: dict,
    ) -> None:
        """Write the index.md hub file.

        Args:
            folder: Research folder path.
            state: Complete research state.
            report: Research report.
            ctx: Common template context.
        """
        # Build source notes links as wikilinks
        source_links = []
        for source in report.sources:
            slug = self._slugify(source.title)
            source_links.append(
                f"- [[notes/source-{source.id:03d}-{slug}|{source.title}]]"
            )

        # Build topic tags from subtopics
        topic_tags = ""
        if state.brief and state.brief.subtopics:
            tags = [self._slugify(t)[:20] for t in state.brief.subtopics[:3]]
            topic_tags = ", " + ", ".join(tags)

        # Count cited sources (relevance > 0.5)
        cited_count = len([s for s in report.sources if s.relevance_score > 0.5])

        # Status handling
        status = "completed" if state.completed_at else "in-progress"
        status_emoji = "v" if state.completed_at else "~"

        content = self._render_template(
            "index_template.md",
            **ctx,
            status=status,
            status_emoji=status_emoji,
            original_query=state.request.query,
            sources_count=len(report.sources),
            cited_count=cited_count,
            condensed_summary=report.executive_summary[:300] + "...",
            executive_summary=report.executive_summary,
            source_notes_links="\n".join(source_links),
            topic_tags=topic_tags,
        )

        (folder / "index.md").write_text(content, encoding="utf-8")

    def _write_brief(self, folder: Path, state: ResearchState, ctx: dict) -> None:
        """Write the brief.md file.

        Args:
            folder: Research folder path.
            state: Complete research state.
            ctx: Common template context.
        """
        if not state.brief:
            return

        subtopics_list = "\n".join(f"- {t}" for t in state.brief.subtopics)

        content = self._render_template(
            "brief_template.md",
            **ctx,
            original_query=state.request.query,
            refined_question=state.brief.refined_question,
            scope=state.brief.scope,
            subtopics_list=subtopics_list,
            constraints=state.brief.constraints,
            language=state.brief.language,
        )

        (folder / "brief.md").write_text(content, encoding="utf-8")

    def _write_report(self, folder: Path, report: ResearchReport) -> None:
        """Write the report.md file with the actual report content.

        Args:
            folder: Research folder path.
            report: Research report with sections.
        """
        lines = [f"# {report.title}", "", report.executive_summary, ""]

        # Add report sections
        for section in report.sections:
            lines.append(f"## {section.get('heading', 'Section')}")
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")

        # Add recommendations if present
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Add limitations if present
        if report.limitations:
            lines.append("## Limitations")
            lines.append("")
            for lim in report.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        # Add references section
        lines.append("## References")
        lines.append("")
        for source in report.sources:
            lines.append(f"[{source.id}] {source.title}. {source.url}")

        (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_sources(
        self, folder: Path, report: ResearchReport, ctx: dict
    ) -> None:
        """Write the sources.md bibliography file.

        Args:
            folder: Research folder path.
            report: Research report with sources.
            ctx: Common template context.
        """
        # Build YAML sources block for frontmatter
        sources_yaml_lines = []
        for s in report.sources:
            sources_yaml_lines.append(f'  - id: {s.id}')
            sources_yaml_lines.append(f'    url: "{s.url}"')
            sources_yaml_lines.append(f'    title: "{s.title}"')
            sources_yaml_lines.append(f"    type: {s.source_type.value}")
            sources_yaml_lines.append(f"    relevance: {s.relevance_score:.2f}")
        sources_yaml = "\n".join(sources_yaml_lines)

        # Build bibliography section
        bibliography_lines = []
        for s in report.sources:
            bibliography_lines.append(f"### [{s.id}] {s.title}")
            bibliography_lines.append("")
            bibliography_lines.append(f"**URL**: [{s.url}]({s.url})")
            bibliography_lines.append("")
            bibliography_lines.append(f"**Type**: {s.source_type.value}")
            bibliography_lines.append("")
            bibliography_lines.append(f"**Relevance**: {s.relevance_score:.1%}")
            bibliography_lines.append("")
            bibliography_lines.append(s.content_summary)
            bibliography_lines.append("")
        bibliography = "\n".join(bibliography_lines)

        # Build sources table
        sources_table_lines = []
        for s in report.sources:
            used = "v" if s.relevance_score > 0.5 else "x"
            title_truncated = (
                s.title[:40] + "..." if len(s.title) > 40 else s.title
            )
            sources_table_lines.append(
                f"| {s.id} | {title_truncated} | {s.source_type.value} | "
                f"{s.relevance_score:.0%} | {used} |"
            )
        sources_table = "\n".join(sources_table_lines)

        # Calculate average relevance
        avg_relevance = (
            sum(s.relevance_score for s in report.sources) / len(report.sources)
            if report.sources
            else 0
        )

        cited_count = len([s for s in report.sources if s.relevance_score > 0.5])

        content = self._render_template(
            "sources_template.md",
            **ctx,
            sources_yaml=sources_yaml,
            total_sources=len(report.sources),
            cited_count=cited_count,
            avg_relevance=f"{avg_relevance:.1%}",
            bibliography=bibliography,
            sources_table=sources_table,
        )

        (folder / "sources.md").write_text(content, encoding="utf-8")

    def _write_methodology(
        self, folder: Path, state: ResearchState, ctx: dict
    ) -> None:
        """Write the methodology.md file.

        Args:
            folder: Research folder path.
            state: Complete research state.
            ctx: Common template context.
        """
        # Calculate duration
        duration = ""
        if state.completed_at and state.started_at:
            delta = state.completed_at - state.started_at
            minutes = int(delta.total_seconds() / 60)
            if minutes < 60:
                duration = f"{minutes} minutes"
            else:
                hours = minutes // 60
                mins = minutes % 60
                duration = f"{hours}h {mins}m"

        # Build subtopics details from researchers
        subtopics_lines = []
        for i, r in enumerate(state.researchers, 1):
            status = "v" if r.completed else "x"
            source_count = len(r.sources) if hasattr(r, "sources") else 0
            subtopics_lines.append(
                f"{i}. {status} {r.subtopic} ({source_count} sources)"
            )
        subtopics_details = "\n".join(subtopics_lines)

        # Get quality metrics from report
        report = state.report
        comprehensiveness = (
            f"{report.comprehensiveness:.0%}" if report else "N/A"
        )
        analytical_depth = (
            f"{report.analytical_depth:.0%}" if report else "N/A"
        )
        source_diversity = (
            f"{report.source_diversity:.0%}" if report else "N/A"
        )
        citation_density = (
            f"{report.citation_density:.0%}" if report else "N/A"
        )

        # Estimate cost (rough approximation)
        estimated_cost = f"${state.total_tokens * 0.00001:.4f}"

        content = self._render_template(
            "methodology_template.md",
            **ctx,
            depth=state.request.depth.value,
            max_sources=state.request.max_sources,
            researcher_count=len(state.researchers),
            started_at=state.started_at.isoformat(),
            completed_at=(
                state.completed_at.isoformat()
                if state.completed_at
                else "In progress"
            ),
            duration=duration,
            search_queries="(Logged during research)",
            subtopics_details=subtopics_details,
            comprehensiveness=comprehensiveness,
            analytical_depth=analytical_depth,
            source_diversity=source_diversity,
            citation_density=citation_density,
            total_tokens=state.total_tokens,
            estimated_cost=estimated_cost,
            decisions_log="(Logged during research)",
        )

        (folder / "methodology.md").write_text(content, encoding="utf-8")

    def _write_source_note(
        self, folder: Path, source: ResearchSource, ctx: dict
    ) -> None:
        """Write an individual source note.

        Args:
            folder: Research folder path.
            source: Source to document.
            ctx: Common template context.
        """
        slug = self._slugify(source.title)
        filename = f"source-{source.id:03d}-{slug}.md"

        # Format key quotes
        if source.key_quotes:
            key_quotes_lines = []
            for quote in source.key_quotes:
                key_quotes_lines.append(f"> {quote}")
                key_quotes_lines.append("")
            key_quotes = "\n".join(key_quotes_lines)
        else:
            key_quotes = "*No specific quotes extracted*"

        content = self._render_template(
            "source_note_template.md",
            **ctx,
            source_id=source.id,
            source_title=source.title,
            url=source.url,
            source_type=source.source_type.value,
            relevance_score=f"{source.relevance_score:.2f}",
            accessed_at=source.accessed_at.isoformat()[:10],
            content_summary=source.content_summary,
            key_quotes=key_quotes,
            used_in_sections="*See report for usage*",
        )

        (folder / "notes" / filename).write_text(content, encoding="utf-8")


__all__ = ["ResearchVaultPersister"]
