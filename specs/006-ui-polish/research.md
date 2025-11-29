# Research: UI Polish Pack Implementation

**Feature**: 006-ui-polish
**Date**: 2025-11-28
**Researcher**: Explore Agent (Sonnet 4.5)

## 1. Font Size Adjuster

**Decision**: CSS Custom Properties (CSS Variables) scoped to content area

**Rationale**:
- WCAG 2.2 compliance requires text scalable to 200% without loss of functionality
- CSS custom properties provide centralized control and easy dynamic updates
- Existing codebase already uses CSS variables for theming
- Avoids inline styles and className switching complexity

**Implementation Details**:
- Add CSS variable `--content-font-size` to `:root`
- Apply to `.prose` container (already used in NoteViewer.tsx:150)
- Range: 0.875rem (14px small), 1rem (16px medium), 1.125rem (18px large)
- Persist to localStorage key `note-font-size`
- Use rem units (relative to root) for scalability

**Alternatives Considered**:
- Inline styles: Too scattered, hard to maintain
- className switching: Limited to predefined sizes
- Browser zoom: Affects entire page including UI chrome

**Code References**:
- CSS variables: `frontend/src/index.css:6-50`
- Prose container: `frontend/src/components/NoteViewer.tsx:150`
- Slider pattern: `frontend/src/components/ui/slider.tsx`
- TTS volume pattern: `frontend/src/components/NoteViewer.tsx:115-130`

---

## 2. Directory Tree Expand/Collapse

**Decision**: Global expand/collapse state management in DirectoryTree component

**Rationale**:
- Current implementation uses local `isOpen` state per folder
- Auto-expands first 2 levels (depth < 2)
- Pattern already established, needs propagation mechanism

**Implementation Details**:
- Add buttons above directory tree: "Expand All" / "Collapse All"
- Add `forceExpandState` prop (undefined | boolean) to TreeNodeItem
- Override local `isOpen` when `forceExpandState` is set
- Reset to undefined after 300ms transition
- Leverage existing `buildTree()` structure

**Pattern**:
```typescript
const [expandAllState, setExpandAllState] = useState<boolean | undefined>(undefined);
const effectiveIsOpen = forceExpandState ?? isOpen;
```

**Alternatives Considered**:
- Fully controlled component: More complex, loses local state
- Context API: Overkill for simple toggle
- Recursive traversal: Already supported by tree structure

**Code References**:
- Folder state: `frontend/src/components/DirectoryTree.tsx:97`
- Tree building: `frontend/src/components/DirectoryTree.tsx:29-86`
- Folder rendering: `frontend/src/components/DirectoryTree.tsx:134-173`

---

## 3. Wikilink Preview Tooltips

**Decision**: Use shadcn/ui HoverCard with 500ms delay

**Rationale**:
- HoverCard supports rich content vs Tooltip (simple text)
- 500ms delay prevents flickering during cursor movement
- Designed specifically for link previews
- Existing wikilink handling in markdown.tsx

**Implementation Details**:
- Install: `npx shadcn@latest add hover-card`
- Wrap wikilink spans with HoverCard component
- Set `openDelay={500}` and `closeDelay={100}`
- Fetch first 200 characters of target note body
- Cache previews in React.useMemo
- Show loading skeleton during fetch

**Cache Strategy**:
```typescript
const [previewCache, setPreviewCache] = useState<Map<string, string>>(new Map());
```

**Accessibility Note**:
- HoverCard is hover-only (not keyboard accessible)
- Ensure wikilinks remain clickable and keyboard navigable

**Alternatives Considered**:
- Tooltip: Too limited for content
- Popover: Requires click, disrupts flow
- Custom implementation: More work, less accessible

**Code References**:
- Wikilink component: `frontend/src/lib/markdown.tsx:16-44`
- Note fetching: `frontend/src/services/api.ts` (getNote)

---

## 4. Table of Contents

**Decision**: Extract headings via custom react-markdown renderer, render in collapsible sidebar

**Rationale**:
- react-markdown already parses headings
- Can intercept rendering to extract TOC data
- ResizablePanel pattern matches existing architecture
- Smooth scrolling with prefers-reduced-motion support

**Implementation Details**:
- Add ID generation to h1-h6 renderers
- Slugify: `text.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '')`
- Extract: `{ id, text, level }[]` using useRef
- Render in ResizablePanel (right sidebar)
- Smooth scroll:
```typescript
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
element.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
```

**Panel Pattern**:
- Button in NoteViewer header (near TTS)
- Default collapsed, opens on click
- Hierarchical indentation by heading level
- Persist state to localStorage

**Alternatives Considered**:
- Sheet component: Not in codebase, needs installation
- Dialog: Blocks content, poor UX
- Inline floating: Overlaps on small screens

**Code References**:
- Heading renderers: `frontend/src/lib/markdown.tsx:61-75`
- ResizablePanel: `frontend/src/components/NoteEditor.tsx:8`
- Three-panel layout: `frontend/src/pages/MainApp.tsx:583-778`

---

## 5. Smooth Transitions

**Decision**: CSS transitions with tailwindcss-animate

**Rationale**:
- Already using `tailwindcss-animate` plugin
- 60fps GPU-accelerated performance
- No JavaScript overhead
- Existing animations: fade-in, slide-in-up, etc.
- Framer Motion adds 90KB - unnecessary

**Implementation Details**:
- Extend tailwind.config.js keyframes
- Add transition utilities:
```css
transition: {
  'smooth': 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
  'fade': 'opacity 0.2s ease-in-out',
}
```
- Apply via className
- Animate only transform/opacity (GPU properties)

**Performance Rules**:
- AVOID: margin, padding, border, width, height (causes reflow)
- USE: transform, opacity only
- Use `will-change` sparingly

**Alternatives Considered**:
- Framer Motion: 90KB, complex API
- React Spring: Physics-based, unnecessary
- GSAP: 50KB, license issues

**Code References**:
- Existing animations: `frontend/tailwind.config.js:57-97`
- Usage: `frontend/src/components/NoteViewer.tsx:79-163`
- Package: `package.json:61` (tailwindcss-animate)

---

## 6. Reading Time Estimation

**Decision**: Calculate from markdown using markdownToPlainText, 200 WPM standard

**Rationale**:
- Industry standard: 200 WPM (conservative)
- Existing utility strips formatting
- Simple: `Math.ceil(wordCount / 200)`
- Display as Badge near title

**Implementation Details**:
- Use `markdownToPlainText(note.body)`
- Word count: `plainText.trim().split(/\s+/).length`
- Format: "X min read"
- Location: After note title (NoteViewer.tsx:82-83)
- Cache in useMemo

**Pattern**:
```typescript
const readingTime = useMemo(() => {
  const plainText = markdownToPlainText(note.body);
  const wordCount = plainText.trim().split(/\s+/).length;
  const minutes = Math.ceil(wordCount / 200);
  return minutes >= 1 ? `${minutes} min read` : null;
}, [note.body]);
```

**Alternatives Considered**:
- reading-time npm: Overkill
- 238/250 WPM: Less conservative
- Character-based: Less accurate

**Code References**:
- markdownToPlainText: `frontend/src/lib/markdownToText.ts`
- Badge: `frontend/src/components/ui/badge.tsx`
- Header: `frontend/src/components/NoteViewer.tsx:79-146`

---

## 7. Particle Effects (Stretch)

**Decision**: CSS-only particles OR shadcn particles component

**Rationale**:
- CSS-only: Zero bundle, 60fps
- shadcn has Particles component (installable)
- Decorative only - no core impact
- Must respect prefers-reduced-motion

**Implementation Options**:

**Option A: CSS-Only (Recommended)**
- @keyframes with transform/opacity
- Multiple elements, staggered delays
- Box-shadow for many particles
- Low z-index, pointer-events: none

**Option B: shadcn Particles**
- Install: `npx shadcn@latest add particles`
- Limit 50-100 particles max
- Disable on mobile

**Performance Mitigation**:
- Max 50 particles
- Transform-only animations
- Disable on mobile (media query)
- Respect prefers-reduced-motion
- Optional/opt-in feature

**When to Use**:
- Success celebrations
- Easter eggs
- Optional backgrounds

**When NOT to Use**:
- During reading
- On lower-end devices
- By default in production

**Alternatives Considered**:
- Lottie: Large files
- Canvas particles: Complex
- WebGL: Massive overkill

**Code References**:
- shadcn particles: Available via CLI
- Animations: `frontend/tailwind.config.js:57-97`
- Settings: `frontend/src/pages/Settings.tsx` (toggle location)

---

## Technology Decisions Summary

| Feature | Technology | Rationale |
|---------|-----------|-----------|
| Font Size | CSS Custom Properties | WCAG compliance, existing pattern |
| Expand/Collapse | State propagation | Matches existing architecture |
| Wikilink Preview | shadcn HoverCard | Rich content, accessibility |
| Table of Contents | Custom renderer + ResizablePanel | Existing markdown/layout patterns |
| Transitions | tailwindcss-animate | Already installed, performant |
| Reading Time | markdownToPlainText + 200 WPM | Industry standard, simple |
| Particles | CSS-only or shadcn | Zero/minimal bundle impact |

## Dependencies to Add

- `hover-card` (shadcn/ui component) - for wikilink previews
- `particles` (optional, stretch goal) - for particle effects

## Performance Targets

- Font size change: < 100ms
- Expand All (100 folders): < 2s
- Wikilink preview: < 600ms (500ms delay + 100ms fetch)
- TOC generation: < 500ms (50 headings)
- All animations: 60 FPS
- No layout thrashing or jank

## Accessibility Compliance

- WCAG 2.2 Level AA compliance
- Text scalable to 200%
- Keyboard navigation support
- prefers-reduced-motion respect
- ARIA labels on controls
- Minimum touch target: 24x24px
