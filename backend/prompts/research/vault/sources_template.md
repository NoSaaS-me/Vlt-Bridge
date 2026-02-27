---
title: "Sources: {{ title }}"
type: research-sources
parent: "[[index]]"
created: {{ created_date }}
sources:
{{ sources_yaml }}
---

# Sources

## Summary

- **Total Sources Evaluated**: {{ total_sources }}
- **Sources Cited**: {{ cited_count }}
- **Average Relevance Score**: {{ avg_relevance }}

## Bibliography

{{ bibliography }}

---

## Source Quality Notes

| ID | Title | Type | Relevance | Used |
|----|-------|------|-----------|------|
{{ sources_table }}
