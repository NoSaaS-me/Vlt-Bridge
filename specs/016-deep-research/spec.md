# Deep Research Mode for Vlt Oracle

**Feature ID**: 016-deep-research
**Status**: Draft
**Created**: 2026-01-04
**Author**: TBD

## Overview

This feature adds a "deep research mode" to the Vlt Oracle agent, enabling automated multi-step research with parallel execution, source synthesis, and comprehensive report generation.

## Background

Based on analysis of [LangChain's Open Deep Research](https://github.com/langchain-ai/open_deep_research), which achieved a 0.4344 RACE score on the Deep Research Bench leaderboard (ranked #6).

See `research.md` for detailed implementation analysis.

## Problem Statement

Current Oracle agent handles single-turn queries well but lacks:
1. Structured multi-step research workflows
2. Parallel research delegation
3. Automatic source synthesis and citation
4. Comprehensive report generation

## Goals

1. Enable deep, automated research on complex topics
2. Leverage existing Oracle tools (web_search, vault, coderag, github_read)
3. Generate well-structured, cited research reports
4. Support research persistence and continuation

## Non-Goals

1. Replace existing Oracle functionality
2. Implement full LangGraph dependency
3. Support human-in-the-loop approval flows (initially)

## User Stories

TODO: Define user stories based on research.md findings

## Technical Design

TODO: Detail architecture based on research.md integration considerations

## Success Criteria

TODO: Define metrics based on Open Deep Research evaluation patterns

## Dependencies

- Existing Oracle agent infrastructure
- web_search tool
- Vault tools for persistence
- Optional: CodeRAG for code research

## Timeline

TODO: Estimate based on implementation phases from research.md

## References

- [Open Deep Research Repository](https://github.com/langchain-ai/open_deep_research)
- [Deep Research Bench Leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)
- [LangChain Blog Post](https://blog.langchain.com/open-deep-research/)
- Local analysis: `specs/016-deep-research/research.md`
