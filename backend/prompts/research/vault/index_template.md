---
title: "Research: {{ title }}"
type: research-project
status: {{ status }}
created: {{ created_date }}
completed: {{ completed_date | default("") }}
tags: [research, deep-research{{ topic_tags }}]
research_id: {{ research_id }}
query: "{{ original_query }}"
sources_count: {{ sources_count }}
condensed: |
  {{ condensed_summary }}
---

# {{ title }}

**Research Brief**: [[brief]]
**Status**: {{ status_emoji }} {{ status }}

## Summary

{{ executive_summary }}

## Report

-> [[report]]

## Sources

-> [[sources]] ({{ sources_count }} sources, {{ cited_count }} cited)

## Research Notes

{{ source_notes_links }}

## Methodology

-> [[methodology]]

---
*Research completed by Vlt Oracle Deep Research on {{ completed_date }}*
