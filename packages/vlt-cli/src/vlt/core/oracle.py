"""Oracle orchestrator - main entry point for multi-source intelligent retrieval.

T075: Coordinates the entire oracle query pipeline:
1. Analyze query type
2. Retrieve from multiple sources (code, vault, threads)
3. Rerank results
4. Assemble context with token budget
5. Synthesize answer with LLM
6. Track timing and costs
"""

import logging
import time
import httpx
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from dataclasses import asdict

from vlt.core.query_analyzer import analyze_query, QueryType, get_primary_symbol
from vlt.core.retrievers.hybrid import hybrid_retrieve, _create_default_retrievers
from vlt.core.retrievers.vault import VaultRetriever
from vlt.core.retrievers.threads import ThreadRetriever
from vlt.core.retrievers.base import RetrievalResult as DataclassRetrievalResult
from vlt.core.context_assembler import assemble_context, estimate_tokens
from vlt.core.prompts import (
    build_synthesis_prompt,
    build_no_context_response,
    build_explain_trace_section,
    extract_citations_from_response
)
from vlt.core.schemas import OracleQuery, OracleResponse, RetrievalResult as PydanticRetrievalResult
from vlt.core.conversation import ConversationManager
from vlt.config import Settings
from vlt.db import SessionLocal

# Optional: repo map generation (requires tree-sitter)
try:
    from vlt.core.coderag.repomap import (
        extract_symbols_from_ast,
        build_reference_graph,
        calculate_centrality,
        generate_repo_map
    )
    REPO_MAP_AVAILABLE = True
except ImportError:
    REPO_MAP_AVAILABLE = False


logger = logging.getLogger(__name__)


class OracleOrchestrator:
    """Main orchestrator for Oracle queries.

    Coordinates multi-source retrieval, context assembly, and synthesis
    to answer natural language questions about codebases.

    Attributes:
        settings: Settings instance with API keys and configuration
        project_id: Project identifier for scoping searches
        project_path: Path to project root directory
    """

    def __init__(
        self,
        project_id: str,
        project_path: str,
        settings: Optional[Settings] = None
    ):
        """Initialize Oracle orchestrator.

        Args:
            project_id: Project identifier
            project_path: Path to project root directory
            settings: Optional settings instance (uses default if None)
        """
        self.project_id = project_id
        self.project_path = project_path
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized OracleOrchestrator for project: {project_id}")

    async def query(
        self,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        max_context_tokens: int = 16000,
        include_repo_map: bool = True,
        user_id: Optional[str] = None,
        use_conversation: bool = True,
        db: Optional[Session] = None
    ) -> OracleResponse:
        """Execute oracle query pipeline.

        Args:
            question: Natural language question
            sources: Optional filter for sources (['vault', 'code', 'threads'])
            explain: Include retrieval traces in response
            max_context_tokens: Token budget for context assembly
            include_repo_map: Include repository map in context
            user_id: User identifier for conversation tracking
            use_conversation: Enable shared conversation context
            db: Optional database session

        Returns:
            OracleResponse with answer, sources, and metadata
        """
        start_time = time.time()

        self.logger.info(f"Processing oracle query: '{question[:80]}...'")

        # Track timing for different phases
        timings = {}

        # Initialize conversation if enabled
        conversation = None
        conversation_manager = None
        conversation_context = ""

        owns_db = db is None
        if owns_db:
            db = SessionLocal()

        try:
            if use_conversation and user_id:
                conversation_manager = ConversationManager(db=db, settings=self.settings)
                conversation = conversation_manager.get_or_create_conversation(
                    project_id=self.project_id,
                    user_id=user_id
                )

                # Get conversation context to prepend to synthesis
                conversation_context = conversation_manager.get_conversation_context(
                    conversation,
                    max_tokens=max_context_tokens // 4  # Reserve 25% of budget for conversation
                )

                if conversation_context:
                    self.logger.info(
                        f"Including conversation context from {conversation.compression_count} "
                        f"compressions, {conversation.tokens_used} tokens used"
                    )

            # Phase 1: Analyze query type
            phase_start = time.time()
            query_analysis = analyze_query(question)
            timings["query_analysis"] = int((time.time() - phase_start) * 1000)

            self.logger.info(
                f"Query type: {query_analysis.query_type.value} "
                f"(confidence: {query_analysis.confidence:.2f})"
            )

            # Extract primary symbol for focused queries
            primary_symbol = get_primary_symbol(query_analysis)

            # Phase 2: Retrieve from multiple sources
            phase_start = time.time()

            # Determine which retrievers to use
            retrievers = []

            # Code retrievers (vector, BM25, graph) - always included unless filtered
            if not sources or 'code' in sources:
                retrievers.extend(_create_default_retrievers(
                    project_id=self.project_id,
                    project_path=self.project_path,
                    settings=self.settings,
                    db=db
                ))

            # Vault retriever
            if not sources or 'vault' in sources:
                retrievers.append(VaultRetriever(settings=self.settings))

            # Thread retriever
            if not sources or 'threads' in sources:
                retrievers.append(ThreadRetriever(
                    project_id=self.project_id,
                    db=db
                ))

            # Run hybrid retrieval
            results = await hybrid_retrieve(
                query=question,
                project_id=self.project_id,
                project_path=self.project_path,
                retrievers=retrievers,
                top_k=20,
                use_reranking=True,
                settings=self.settings,
                db=db
            )

            timings["retrieval"] = int((time.time() - phase_start) * 1000)

            # If no results, return early with honest response
            if not results:
                self.logger.warning("No relevant context found for query")

                total_time = int((time.time() - start_time) * 1000)

                return OracleResponse(
                    answer=build_no_context_response(question),
                    sources=[],
                    repo_map_slice=None,
                    traces=None,
                    query_type=query_analysis.query_type.value,
                    model="none",
                    tokens_used=0,
                    cost_cents=0.0,
                    duration_ms=total_time
                )

            # Separate results by source type for assembly
            code_results = [r for r in results if r.source_type.value == "code"]
            vault_results = [r for r in results if r.source_type.value == "vault"]
            thread_results = [r for r in results if r.source_type.value == "thread"]

            self.logger.info(
                f"Retrieved {len(code_results)} code, {len(vault_results)} vault, "
                f"{len(thread_results)} thread results"
            )

            # Phase 3: Generate repo map slice if requested
            repo_map_text = None
            if include_repo_map:
                phase_start = time.time()
                # For now, we'll skip repo map generation if it requires heavy parsing
                # This would need the full symbol extraction pipeline
                # TODO: Implement lightweight repo map generation or caching
                timings["repo_map"] = int((time.time() - phase_start) * 1000)

            # Phase 4: Assemble context with token budget
            phase_start = time.time()

            # Adjust token budget to account for conversation context
            retrieval_context_budget = max_context_tokens
            if conversation_context:
                retrieval_context_budget = max_context_tokens - (max_context_tokens // 4)

            context_data = assemble_context(
                code_results=code_results,
                vault_results=vault_results,
                thread_results=thread_results,
                repo_map=repo_map_text,
                max_tokens=retrieval_context_budget,
                query_type=query_analysis.query_type.value
            )

            assembled_context = context_data["context"]
            token_count = context_data["token_count"]

            timings["context_assembly"] = int((time.time() - phase_start) * 1000)

            self.logger.info(f"Assembled context: {token_count} tokens")

            # Phase 5: Synthesize answer with LLM
            phase_start = time.time()

            # Prepend conversation context to assembled context
            full_context = assembled_context
            if conversation_context:
                full_context = f"# Previous Conversation\n{conversation_context}\n\n# Current Context\n{assembled_context}"

            synthesis_prompt = build_synthesis_prompt(
                question=question,
                context=full_context,
                query_type=query_analysis.query_type.value,
                include_citations=True
            )

            # Call LLM for synthesis
            answer, synthesis_tokens, synthesis_cost = await self._synthesize_answer(
                prompt=synthesis_prompt
            )

            timings["synthesis"] = int((time.time() - phase_start) * 1000)

            # Extract citations from answer
            citations = extract_citations_from_response(answer)
            self.logger.info(f"Generated answer with {len(citations)} citations")

            # Log exchange to conversation if enabled
            if conversation and conversation_manager:
                conversation_manager.log_exchange(
                    conversation=conversation,
                    tool_name="ask_oracle",
                    input_data={"question": question, "sources": sources},
                    output_data={"answer": answer, "sources": results[:10]}
                )

            # Build traces if explain mode
            traces = None
            if explain:
                retrieval_stats = {
                    "code": {
                        "count": len(code_results),
                        "avg_score": sum(r.score for r in code_results) / len(code_results) if code_results else 0
                    },
                    "vault": {
                        "count": len(vault_results),
                        "avg_score": sum(r.score for r in vault_results) / len(vault_results) if vault_results else 0
                    },
                    "threads": {
                        "count": len(thread_results),
                        "avg_score": sum(r.score for r in thread_results) / len(thread_results) if thread_results else 0
                    }
                }

                traces = {
                    "query_analysis": {
                        "query_type": query_analysis.query_type.value,
                        "confidence": query_analysis.confidence,
                        "extracted_symbols": query_analysis.extracted_symbols,
                        "reasoning": query_analysis.reasoning
                    },
                    "retrieval_stats": retrieval_stats,
                    "context_stats": {
                        "token_count": token_count,
                        "max_tokens": max_context_tokens,
                        "sources_included": context_data["sources_included"],
                        "sources_excluded": context_data["sources_excluded"]
                    },
                    "timings_ms": timings
                }

                if conversation:
                    traces["conversation"] = {
                        "tokens_used": conversation.tokens_used,
                        "token_budget": conversation.token_budget,
                        "compression_count": conversation.compression_count,
                        "status": conversation.status.value
                    }

            # Calculate total cost and time
            total_time = int((time.time() - start_time) * 1000)
            total_cost = synthesis_cost  # In cents

            self.logger.info(
                f"Oracle query complete: {total_time}ms, ${total_cost:.4f}, "
                f"{synthesis_tokens} tokens"
            )

            # Convert dataclass RetrievalResult to Pydantic RetrievalResult
            pydantic_sources = [
                PydanticRetrievalResult(**asdict(r))
                for r in results[:10]  # Top 10 sources for response
            ]

            return OracleResponse(
                answer=answer,
                sources=pydantic_sources,
                repo_map_slice=repo_map_text,
                traces=traces,
                query_type=query_analysis.query_type.value,
                model=self.settings.openrouter_model,
                tokens_used=synthesis_tokens,
                cost_cents=total_cost,
                duration_ms=total_time
            )

        except Exception as e:
            self.logger.error(f"Error in oracle query: {e}", exc_info=True)
            raise
        finally:
            if owns_db and db:
                db.close()

    async def _synthesize_answer(self, prompt: str) -> tuple[str, int, float]:
        """Synthesize answer using LLM.

        Args:
            prompt: Full synthesis prompt

        Returns:
            Tuple of (answer_text, tokens_used, cost_in_cents)
        """
        if not self.settings.openrouter_api_key:
            self.logger.error("No OpenRouter API key configured")
            return "Error: No OpenRouter API key configured. Run `vlt config set-key <key>`", 0, 0.0

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings.openrouter_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.3,  # Lower for more factual answers
                        "max_tokens": 4000,  # Reserve for answer
                    }
                )

                if response.status_code != 200:
                    self.logger.error(f"LLM API error {response.status_code}: {response.text}")
                    return f"Error: LLM API returned {response.status_code}", 0, 0.0

                data = response.json()

                answer = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)

                # Estimate cost (rough: $0.001 per 1K tokens for typical models)
                cost_per_1k = 0.001
                cost_cents = (total_tokens / 1000) * cost_per_1k * 100

                return answer, total_tokens, cost_cents

        except httpx.TimeoutException:
            self.logger.error("LLM synthesis timed out")
            return "Error: LLM synthesis timed out (60s)", 0, 0.0
        except Exception as e:
            self.logger.error(f"Error during synthesis: {e}", exc_info=True)
            return f"Error during synthesis: {str(e)}", 0, 0.0


# Convenience function for simple queries
async def ask_oracle(
    question: str,
    project_id: str,
    project_path: str,
    user_id: Optional[str] = None,
    use_conversation: bool = True,
    settings: Optional[Settings] = None,
    **kwargs
) -> OracleResponse:
    """Ask the oracle a question about a project.

    Convenience function for simple oracle queries.

    Args:
        question: Natural language question
        project_id: Project identifier
        project_path: Path to project root
        user_id: User identifier for conversation tracking
        use_conversation: Enable shared conversation context
        settings: Optional settings instance
        **kwargs: Additional arguments for orchestrator.query()

    Returns:
        OracleResponse with answer and metadata
    """
    orchestrator = OracleOrchestrator(
        project_id=project_id,
        project_path=project_path,
        settings=settings
    )

    return await orchestrator.query(
        question,
        user_id=user_id,
        use_conversation=use_conversation,
        **kwargs
    )
