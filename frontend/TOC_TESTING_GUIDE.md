# Table of Contents Feature - Testing Guide

This guide provides instructions for testing the Table of Contents (TOC) feature implementation.

## Overview

The TOC feature (User Story 5, Tasks T037-T052) adds a collapsible sidebar panel that displays document headings and enables quick navigation.

## Features Implemented

### Core Functionality
- **Heading Extraction**: Automatically extracts H1, H2, and H3 headings from rendered markdown
- **Unique ID Generation**: Uses slugify algorithm with duplicate handling (-2, -3, etc.)
- **Smooth Scrolling**: Respects `prefers-reduced-motion` media query
- **State Persistence**: Panel open/closed state saved to localStorage (`toc-panel-open`)
- **Hierarchical Display**: Headings indented by level (H1=0px, H2=12px, H3=24px)
- **Empty State**: Shows helpful message when no headings found

### UI Components
- **TOC Button**: Added to NoteViewer toolbar (List icon + "TOC" text)
- **Resizable Panel**: Right sidebar with adjustable width (15-40% of viewer)
- **Click Navigation**: Click any heading to scroll to that section

## Manual Testing Checklist

### T051: Panel State Persistence
**Goal**: Verify TOC panel state persists after reload

1. Open the application and view any note
2. Click the "TOC" button to open the panel
3. Verify the panel opens on the right side
4. Refresh the browser (F5 or Cmd+R)
5. **Expected**: TOC panel should still be open after reload
6. Click "TOC" to close the panel
7. Refresh the browser again
8. **Expected**: TOC panel should remain closed after reload

**localStorage Check**:
- Open DevTools > Application > Local Storage
- Look for key `toc-panel-open`
- Value should be `true` when open, `false` when closed

### T052: Performance Test (<500ms for 50 headings)
**Goal**: Verify TOC generation completes in <500ms for 50 headings

**Test Document Creation**:
Create a test note with 50 headings (mix of H1, H2, H3):

```markdown
# Heading 1
## Subheading 1.1
### Detail 1.1.1
### Detail 1.1.2
## Subheading 1.2
# Heading 2
## Subheading 2.1
...
(repeat pattern to reach 50 headings)
```

**Performance Measurement**:
1. Open DevTools > Performance tab
2. Start recording
3. Navigate to the test note with 50 headings
4. Wait for note to fully render
5. Open the TOC panel
6. Stop recording
7. **Expected**: Total time from note load to TOC display < 500ms

**Alternative - Console Timing**:
Add temporary timing code to `useTableOfContents.ts`:
```typescript
const extractHeadings = useCallback(() => {
  const start = performance.now();
  // ... existing code ...
  const duration = performance.now() - start;
  console.log(`TOC extraction took ${duration.toFixed(2)}ms for ${extracted.length} headings`);
}, []);
```

### Additional Tests

#### Heading Navigation
1. Create a note with multiple headings
2. Open TOC panel
3. Click various headings in the TOC
4. **Expected**: Page smoothly scrolls to clicked heading

#### Duplicate Heading Handling
1. Create a note with duplicate heading text:
   ```markdown
   # Introduction
   ## Introduction
   ### Introduction
   ```
2. Open TOC panel
3. Inspect heading IDs (DevTools > Elements)
4. **Expected**: IDs should be `introduction`, `introduction-2`, `introduction-3`

#### Reduced Motion Support
1. Enable reduced motion in OS settings:
   - **macOS**: System Preferences > Accessibility > Display > Reduce motion
   - **Windows**: Settings > Ease of Access > Display > Show animations
   - **Linux**: Varies by desktop environment
2. Open TOC and click a heading
3. **Expected**: Scroll should be instant (no smooth animation)

#### Empty State
1. Create a note with no headings (only body text)
2. Open TOC panel
3. **Expected**: Should show "No headings found" message

#### Panel Resize
1. Open TOC panel
2. Drag the resize handle between content and TOC
3. **Expected**: Panel width adjusts smoothly between 15-40% of viewer

## File Locations

### Created Files
- `/home/wolfe/Projects/Document-MCP/frontend/src/hooks/useTableOfContents.ts` - TOC state management hook
- `/home/wolfe/Projects/Document-MCP/frontend/src/components/TableOfContents.tsx` - TOC UI component

### Modified Files
- `/home/wolfe/Projects/Document-MCP/frontend/src/lib/markdown.tsx` - Added heading ID generation
- `/home/wolfe/Projects/Document-MCP/frontend/src/components/NoteViewer.tsx` - Integrated TOC panel

## Technical Implementation Details

### Heading Extraction Algorithm
The TOC uses a MutationObserver to detect when markdown is rendered:

1. Observer watches `.prose` container for DOM changes
2. On mutation, queries for `h1, h2, h3` elements
3. Extracts text content and existing IDs
4. Builds heading array with `{ id, text, level }`

### Slugify Algorithm
Converts heading text to valid HTML IDs:
```typescript
text.toLowerCase()
  .replace(/\s+/g, '-')        // spaces to hyphens
  .replace(/[^\w-]/g, '')      // remove special chars
```

Duplicate handling via global cache that increments on collision.

### Scroll Behavior
```typescript
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
element.scrollIntoView({
  behavior: prefersReducedMotion ? 'auto' : 'smooth',
  block: 'start'
});
```

## Known Limitations

1. **Only H1-H3 supported**: H4-H6 headings are not extracted (as per spec)
2. **Cache reset per note**: Slug cache resets when switching notes to avoid ID conflicts
3. **Simple text extraction**: Complex heading content (links, code) may not render perfectly in TOC

## Troubleshooting

### TOC Panel Not Showing
- Check browser console for errors
- Verify ResizablePanel components are imported correctly
- Ensure `toc-panel-open` localStorage value is set

### Headings Not Appearing
- Verify markdown is rendering (check for `.prose` container)
- Check if headings have IDs in DevTools
- Look for MutationObserver errors in console

### Scroll Not Working
- Verify heading IDs match TOC `id` values
- Check for JavaScript errors when clicking
- Ensure `scrollToHeading` function is connected

## Success Criteria

All tasks (T037-T052) are complete when:
- ✅ Hook and component files created and functional
- ✅ Headings render with unique IDs
- ✅ TOC panel toggles via toolbar button
- ✅ Panel state persists across reloads
- ✅ Clicking headings scrolls smoothly
- ✅ Performance < 500ms for 50 headings
- ✅ Empty state displays when no headings
- ✅ Hierarchical indentation works correctly
