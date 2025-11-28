# Data Model: UI Polish Pack

**Feature**: 006-ui-polish
**Date**: 2025-11-28
**Type**: Client-side only (no backend/database models)

## Overview

All data for UI polish features is stored client-side in browser localStorage or ephemeral React component state. No backend API changes or database modifications required.

## Entities

### 1. FontSizePreference

**Storage**: localStorage
**Key**: `note-font-size`
**Type**: `"small" | "medium" | "large"`
**Scope**: Global (applies to all notes across sessions)

**Attributes**:
- `size`: The user's selected font size
  - `"small"` = 0.875rem (14px)
  - `"medium"` = 1rem (16px) - default
  - `"large"` = 1.125rem (18px)

**Lifecycle**:
- Created: First time user clicks font size button
- Updated: Each font size change
- Persisted: Indefinitely (until user clears browser data)
- Default: `"medium"` if not set

**Usage**:
```typescript
// Read
const savedSize = localStorage.getItem('note-font-size') ?? 'medium';

// Write
localStorage.setItem('note-font-size', 'large');
```

---

### 2. TOCPanelState

**Storage**: localStorage
**Key**: `toc-panel-open`
**Type**: `boolean`
**Scope**: Global (persists across sessions)

**Attributes**:
- `isOpen`: Whether TOC panel is currently visible
  - `true` = panel expanded
  - `false` = panel collapsed

**Lifecycle**:
- Created: First time user clicks TOC button
- Updated: Each TOC toggle
- Persisted: Indefinitely
- Default: `false` (collapsed) if not set

**Usage**:
```typescript
// Read
const isOpen = localStorage.getItem('toc-panel-open') === 'true';

// Write
localStorage.setItem('toc-panel-open', String(isOpen));
```

---

### 3. WikilinkPreviewCache

**Storage**: React component state (ephemeral)
**Type**: `Map<string, string>`
**Scope**: Component lifecycle (NoteViewer or markdown renderer)

**Attributes**:
- `key`: Note path (string) - e.g., "guides/Quick Reference.md"
- `value`: Preview text (string) - first 150-200 characters of note body

**Lifecycle**:
- Created: Component mount
- Updated: On first preview fetch for each unique wikilink
- Cleared: Component unmount
- Not persisted to localStorage (ephemeral per session)

**Usage**:
```typescript
const [previewCache, setPreviewCache] = useState<Map<string, string>>(new Map());

// Check cache
const cached = previewCache.get(notePath);
if (cached) return cached;

// Fetch and cache
const preview = await fetchPreview(notePath);
setPreviewCache(prev => new Map(prev).set(notePath, preview));
```

**Rationale for ephemeral storage**:
- Preview content can change between sessions
- Avoids stale data in localStorage
- Cache hit rate still high within single reading session

---

### 4. TOCHeading

**Storage**: Computed (not persisted)
**Type**: Array of heading objects
**Scope**: Per-note (recomputed when note.body changes)

**Attributes**:
- `id`: Slugified heading text for anchor links (string)
  - Generated via: `text.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '')`
  - Example: "API Documentation" → "api-documentation"
- `text`: Raw heading text (string) - e.g., "API Documentation"
- `level`: Heading depth (1 | 2 | 3)
  - 1 = H1 (`#`)
  - 2 = H2 (`##`)
  - 3 = H3 (`###`)
  - H4-H6 not included in TOC

**Lifecycle**:
- Computed: When markdown is rendered via react-markdown
- Updated: When note.body changes
- Cached: Via useMemo to avoid re-computation on re-renders
- Not persisted (derived data)

**Example**:
```typescript
const headings: TOCHeading[] = [
  { id: 'getting-started', text: 'Getting Started', level: 1 },
  { id: 'installation', text: 'Installation', level: 2 },
  { id: 'prerequisites', text: 'Prerequisites', level: 3 },
];
```

**Usage**:
```typescript
const headings = useMemo(() => {
  const extracted: TOCHeading[] = [];
  // Extract from markdown during render
  // (via custom react-markdown heading renderer)
  return extracted;
}, [note.body]);
```

---

## State Management

### Component Hierarchy

```text
MainApp (root)
├── fontSize state (localStorage-backed)
├── DirectoryTree
│   └── expandAll/collapseAll triggers (ephemeral)
└── NoteViewer
    ├── TOC state (localStorage-backed)
    ├── TOC headings (computed from markdown)
    └── Wikilink preview cache (ephemeral Map)
```

### Data Flow

**Font Size**:
```
User clicks A+/A-/A
  ↓
useFontSize hook updates state
  ↓
localStorage.setItem('note-font-size', newSize)
  ↓
CSS variable --content-font-size updated
  ↓
.prose container text resizes
```

**TOC**:
```
Note body changes
  ↓
useTableOfContents extracts headings (useMemo)
  ↓
User clicks TOC button
  ↓
toggleTOC() updates isOpen state + localStorage
  ↓
TableOfContents component renders/hides
```

**Wikilink Preview**:
```
User hovers on [[wikilink]]
  ↓
500ms delay
  ↓
Check previewCache Map
  ↓ (cache miss)
Fetch note via GET /api/notes/{path}
  ↓
Extract first 150 chars
  ↓
Update cache + display HoverCard
```

---

## Validation Rules

### FontSizePreference
- Must be one of: `"small" | "medium" | "large"`
- If invalid value in localStorage, default to `"medium"`

### TOCPanelState
- Must be parseable as boolean
- If invalid value, default to `false`

### TOCHeading.id
- Must be unique within a note (handle duplicate headings)
- Must be valid HTML ID (a-z, 0-9, -, _)
- If duplicate heading text, append `-2`, `-3`, etc.

---

## Performance Considerations

### LocalStorage Access
- Read once on component mount
- Write only on user action (not on every render)
- Max size: ~5MB per domain (plenty for small preference values)

### Wikilink Cache
- Map lookups are O(1) - efficient
- Cleared on unmount to avoid memory leaks
- Max realistic size: ~50 previews × 200 chars = ~10KB

### TOC Extraction
- useMemo prevents re-computation on unrelated re-renders
- Only recomputes when note.body changes
- Max realistic cost: 50 headings × minimal parsing = <10ms

---

## Migration Notes

**No migrations required** - All data is client-side and newly created.

If user has existing localStorage data:
- New keys (`note-font-size`, `toc-panel-open`) won't conflict
- Existing keys (e.g., `tts-volume`) unaffected

---

## Relationships

No relationships between entities - each is independent:
- Font size doesn't affect TOC
- TOC state doesn't affect preview cache
- All features operate independently

---

## Summary

| Entity | Storage | Persisted | Scope | Size |
|--------|---------|-----------|-------|------|
| FontSizePreference | localStorage | Yes | Global | ~10 bytes |
| TOCPanelState | localStorage | Yes | Global | ~5 bytes |
| WikilinkPreviewCache | React state | No | Component | ~10 KB |
| TOCHeading | Computed | No | Per-note | Negligible |

**Total localStorage footprint**: ~15 bytes (negligible)
**Total memory footprint**: ~10 KB (preview cache)
