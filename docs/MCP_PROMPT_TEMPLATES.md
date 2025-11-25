# MCP Prompt Templates for Document-MCP

Inspired by [jupyter-mcp-server prompt templates](https://github.com/datalayer/jupyter-mcp-server), these templates help you effectively use Document-MCP with AI agents.

## 📝 Note Management Prompts

### Creating Documentation

```
I need to create a new documentation note about [TOPIC]. 
Please:
1. Search for any existing notes about this topic
2. Create a new note at [PATH] with:
   - A clear title
   - Proper frontmatter with relevant tags
   - Well-structured markdown content
   - Links to related notes using [[wikilinks]]
```

### Updating Existing Notes

```
Please update the note at [PATH]:
1. Read the current content
2. Add a new section about [NEW_TOPIC]
3. Update the tags in frontmatter to include [NEW_TAG]
4. Add wikilinks to related notes: [[Note1]], [[Note2]]
```

### Organizing Notes

```
Help me organize my vault:
1. List all notes in the [FOLDER] folder
2. Find notes with tag [TAG]
3. Show me backlinks for [NOTE_PATH]
4. Suggest which notes might need better organization
```

## 🔍 Search and Discovery Prompts

### Finding Information

```
Search my vault for information about [TOPIC]. 
Show me:
- The most relevant notes (top 5)
- Snippets showing where the topic appears
- Related notes via backlinks
```

### Exploring Connections

```
I'm working on [NOTE_PATH]. Show me:
1. All notes that link to this note (backlinks)
2. All notes this note links to
3. Notes with similar tags
4. Recent notes that might be related
```

## 🏷️ Tag Management Prompts

### Tag Analysis

```
Analyze my vault's tag usage:
1. List all tags and their usage counts
2. Identify tags that are used only once (orphaned tags)
3. Suggest tags that might need to be merged
4. Show me notes without any tags
```

### Tag Cleanup

```
Help me clean up tags:
1. Find all notes with tag [OLD_TAG]
2. Update them to use [NEW_TAG] instead
3. Remove the old tag from the tag list
```

## 📚 Documentation Workflow Prompts

### Creating a Guide

```
Create a comprehensive guide about [TOPIC]:
1. Search for existing related notes
2. Create a main guide note at guides/[TOPIC].md
3. Break it into sections with proper headings
4. Link to related notes using [[wikilinks]]
5. Add tags: [guide, TOPIC]
6. Create supporting notes if needed
```

### Maintaining Documentation

```
Review my documentation:
1. Find notes that haven't been updated in 30+ days
2. Check for broken wikilinks (unresolved [[links]])
3. Identify notes with missing frontmatter
4. Suggest notes that might need updates
```

## 🔗 Wikilink Management Prompts

### Resolving Links

```
Check the wikilinks in [NOTE_PATH]:
1. Read the note
2. Extract all [[wikilinks]]
3. Verify each link resolves to an existing note
4. List any broken links
5. Suggest the correct note names for broken links
```

### Creating Link Networks

```
Help me create a knowledge graph:
1. Find all notes related to [TOPIC]
2. Show me the connection graph (which notes link to which)
3. Identify central notes (notes with many backlinks)
4. Suggest notes that should be linked but aren't
```

## 🎯 Best Practices Prompts

### Code Documentation

```
Document this code concept: [CONCEPT_NAME]
1. Check if a note already exists
2. Create/update a note at docs/concepts/[CONCEPT_NAME].md
3. Include:
   - Overview
   - Key points
   - Examples
   - Related concepts (wikilinks)
   - Tags: [concept, code]
```

### Meeting Notes

```
Create meeting notes for [MEETING_NAME]:
1. Create note at meetings/[DATE]-[MEETING_NAME].md
2. Structure with:
   - Attendees
   - Agenda
   - Discussion points
   - Action items
   - Tags: [meeting, DATE]
```

### Project Documentation

```
Set up documentation for project [PROJECT_NAME]:
1. Create project root note: projects/[PROJECT_NAME].md
2. Create sub-notes:
   - projects/[PROJECT_NAME]/goals.md
   - projects/[PROJECT_NAME]/progress.md
   - projects/[PROJECT_NAME]/resources.md
3. Link them all together with wikilinks
4. Tag with: [project, PROJECT_NAME]
```

## 💡 Advanced Workflows

### Daily Note Workflow

```
Create today's daily note:
1. Check if [DATE]-daily.md exists
2. If not, create it with:
   - Date header
   - Sections for: tasks, notes, ideas
   - Link to yesterday's daily note
   - Tag: [daily, DATE]
3. If it exists, show me what's already there
```

### Research Workflow

```
Help me research [TOPIC]:
1. Search for existing notes about [TOPIC]
2. Create a research note at research/[TOPIC].md
3. Structure with:
   - Summary
   - Key findings
   - Sources (as wikilinks to source notes)
   - Questions to explore
   - Tags: [research, TOPIC]
4. Link to related research notes
```

### Learning Path Creation

```
Create a learning path for [SUBJECT]:
1. Search for existing notes about [SUBJECT]
2. Create a learning path note: learning/[SUBJECT]-path.md
3. Organize notes in a logical sequence
4. Create wikilinks between notes showing the path
5. Add tags: [learning, SUBJECT, path]
```

## 🚀 Tips for Effective Prompts

1. **Be Specific**: Instead of "create a note", specify the path, structure, and content
2. **Use Context**: Reference existing notes by path or search results
3. **Request Verification**: Ask the AI to check for existing content before creating
4. **Structure Matters**: Request specific markdown structure (headings, lists, etc.)
5. **Link Everything**: Always ask for wikilinks to related notes
6. **Tag Consistently**: Request specific tags or ask for tag suggestions

## Example: Complete Documentation Task

```
I need to document our API design. Please:

1. Search for any existing API documentation
2. Create a main note: docs/api/design.md
3. Structure it with:
   - Overview section
   - Endpoints section (with subsections for each endpoint)
   - Authentication section
   - Examples section
4. Create supporting notes:
   - docs/api/authentication.md
   - docs/api/endpoints/overview.md
5. Link all notes together with [[wikilinks]]
6. Tag with: [api, documentation, design]
7. Link to related notes about our architecture
8. Check for any broken links and fix them
```

This approach ensures comprehensive, well-linked documentation that's easy to navigate and maintain.

