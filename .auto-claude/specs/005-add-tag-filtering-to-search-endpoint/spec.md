# Add Tag Filtering to Search Endpoint

## Overview
Extend the /api/search endpoint to support tag-based filtering in addition to full-text search. This leverages the existing note_tags table and FTS5 search infrastructure.

The matching tool descriptions and tool prompts need to reflect this.


## Rationale

The IndexerService already maintains a note_tags table with tag indexing (see indexer.py lines 130-137). The search_notes method uses FTS5 for full-text search. Tags are displayed in NoteViewer and retrieved via /api/tags but cannot be used as search filters. The data model supports this - just needs query extension.

---
*This spec was created from ideation and is pending detailed specification.*
