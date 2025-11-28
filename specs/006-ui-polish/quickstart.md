# Quick Start Guide: UI Polish Pack

**Feature**: 006-ui-polish
**Branch**: `006-ui-polish`
**Estimated Time**: 4-6 hours implementation

## Prerequisites

- Node.js 18+ and npm
- Git
- Code editor (VS Code recommended)
- Modern browser (Chrome/Firefox/Safari/Edge)

## Development Setup

### 1. Check Out Feature Branch

```bash
# Ensure you're in project root
cd /home/wolfe/Projects/Document-MCP

# Check out feature branch
git checkout 006-ui-polish

# Verify branch
git branch --show-current
# Should show: 006-ui-polish
```

### 2. Install Dependencies

```bash
# Navigate to frontend
cd frontend

# Install new shadcn/ui component for wikilink previews
npx shadcn@latest add hover-card

# Verify installation
ls src/components/ui/hover-card.tsx
# Should exist

# Return to project root
cd ..
```

### 3. Start Development Server

```bash
# Start frontend dev server
cd frontend
npm run dev

# Server starts at http://localhost:5173
# Open in browser to begin testing
```

### 4. Verify Existing Features

Before implementing polish features, verify baseline functionality:

- [ ] Notes load and display correctly
- [ ] Directory tree shows folders and files
- [ ] Clicking notes navigates successfully
- [ ] Wikilinks render and are clickable
- [ ] TTS controls work (if available)

## Implementation Order

Follow priority order from spec:

### Phase 1: P1 Features (Core)

1. **Font Size Adjuster** (~15 min)
   - Files: `frontend/src/index.css`, `frontend/src/hooks/useFontSize.ts`, `frontend/src/components/NoteViewer.tsx`
   - Test: Click A-/A/A+ buttons, verify text resizes

2. **Expand/Collapse All** (~20 min)
   - Files: `frontend/src/components/DirectoryTree.tsx`, `frontend/src/pages/MainApp.tsx`
   - Test: Click "Expand All" / "Collapse All" buttons

### Phase 2: P2 Features (Important)

3. **Reading Time Estimator** (~10 min)
   - Files: `frontend/src/components/NoteViewer.tsx`
   - Test: Open notes of varying lengths, verify badge appears

4. **Wikilink Preview Tooltips** (~30 min)
   - Files: `frontend/src/lib/markdown.tsx`, `frontend/src/components/ui/hover-card.tsx`
   - Test: Hover over `[[wikilink]]` for 500ms, see preview

5. **Table of Contents** (~40 min)
   - Files: `frontend/src/hooks/useTableOfContents.ts`, `frontend/src/components/TableOfContents.tsx`, `frontend/src/components/NoteViewer.tsx`
   - Test: Open long note, click TOC button, navigate headings

### Phase 3: P3 Features (Polish)

6. **Smooth Transitions** (~10 min)
   - Files: `frontend/tailwind.config.js`, `frontend/src/pages/MainApp.tsx`
   - Test: Switch notes, toggle views, verify animations

### Phase 4: Stretch (Optional)

7. **Particle Effects** (~45 min + tuning)
   - Files: CSS or shadcn particles component
   - Test: Click UI elements, verify no lag

## Testing Checklist

Use this checklist during development and before deployment:

### Font Size Adjuster

- [ ] **A+ button**: Text increases to 18px
- [ ] **A- button**: Text decreases to 14px
- [ ] **A button**: Text resets to 16px (default)
- [ ] **Persistence**: Reload page, verify size retained
- [ ] **Scope**: Only note body text resizes (not UI chrome)
- [ ] **Accessibility**: Can tab to buttons with keyboard
- [ ] **Mobile**: Buttons visible and tappable on small screen

### Expand/Collapse All

- [ ] **Expand All**: All folders open, showing contents
- [ ] **Collapse All**: All folders close, hiding contents
- [ ] **Mixed state**: Works correctly when some folders already open
- [ ] **Performance**: Completes in < 2s for 100+ folders
- [ ] **Visual feedback**: Buttons show active state
- [ ] **Accessibility**: Keyboard navigable

### Wikilink Preview Tooltips

- [ ] **500ms delay**: Tooltip appears after hovering 500ms
- [ ] **Preview content**: Shows first 150-200 characters
- [ ] **Mouse away**: Tooltip disappears when mouse moves away
- [ ] **Broken link**: Shows "Note not found" for invalid links
- [ ] **No flickering**: Quick mouse movement doesn't trigger tooltips
- [ ] **Cache**: Second hover on same link is instant
- [ ] **Accessibility**: Wikilinks still keyboard navigable

### Reading Time Estimator

- [ ] **Badge appears**: Shows "X min read" for notes > 200 words
- [ ] **No badge**: Hidden for very short notes (< 200 words)
- [ ] **Accuracy**: Estimates reasonable (within ±20% of actual)
- [ ] **Position**: Badge near note title, visually balanced
- [ ] **Mobile**: Badge visible on small screens

### Table of Contents

- [ ] **TOC button**: Appears in note toolbar
- [ ] **Panel opens**: Clicking button shows TOC sidebar
- [ ] **Headings listed**: H1, H2, H3 shown with indentation
- [ ] **Click to scroll**: Clicking heading scrolls to section
- [ ] **Smooth scroll**: Scrolling is smooth (not instant)
- [ ] **Close panel**: Clicking button again or outside closes panel
- [ ] **Persistence**: Panel state retained after reload
- [ ] **Empty state**: Shows "No headings found" for notes without headings
- [ ] **Performance**: Generates TOC in < 500ms for 50 headings
- [ ] **Mobile**: Panel overlays content, doesn't break layout

### Smooth Transitions

- [ ] **Note switch**: New note fades in over 300ms
- [ ] **Graph toggle**: Graph slides in over 250ms
- [ ] **Tree selection**: Selected item highlights smoothly
- [ ] **60 FPS**: No dropped frames or jank
- [ ] **Reduced motion**: Respects `prefers-reduced-motion` setting

### Particle Effects (Stretch)

- [ ] **Click feedback**: Particles appear on click
- [ ] **Performance**: No lag or frame drops
- [ ] **Multiple clicks**: Handles rapid clicking gracefully
- [ ] **Reduced motion**: Disabled when user prefers reduced motion
- [ ] **Mobile**: Disabled on mobile or optimized

## Browser Testing

Test in multiple browsers:

- [ ] Chrome/Chromium (latest)
- [ ] Firefox (latest)
- [ ] Safari (if on macOS)
- [ ] Edge (latest)
- [ ] Mobile Safari (iOS)
- [ ] Mobile Chrome (Android)

## Performance Verification

Use browser DevTools to verify performance:

### Chrome DevTools Performance

1. Open DevTools → Performance tab
2. Click Record
3. Test feature (e.g., click font size button)
4. Stop recording
5. Verify:
   - [ ] No long tasks (> 50ms)
   - [ ] 60 FPS maintained
   - [ ] No layout thrashing

### Memory Profiling

1. Open DevTools → Memory tab
2. Take heap snapshot before testing
3. Use features extensively
4. Take heap snapshot after
5. Verify:
   - [ ] No memory leaks
   - [ ] Preview cache clears on unmount
   - [ ] localStorage usage minimal (< 1KB)

## Build & Deploy

### Local Build

```bash
# Build production bundle
cd frontend
npm run build

# Expected output:
# - dist/ directory created
# - Assets minified and optimized
# - No TypeScript errors
# - No linting errors

# Verify build output
ls -lh dist/
# Should show:
# - index.html
# - assets/ (JS/CSS chunks)
# - Total size ~400-500KB gzipped
```

### Preview Production Build

```bash
# Serve production build locally
npm run preview

# Open http://localhost:4173
# Test all features in production mode
```

### Deploy to HuggingFace Space

```bash
# Ensure all changes committed
git status
# Should show clean working tree

# Push to remote
git push origin 006-ui-polish

# HuggingFace Space auto-deploys from git
# Monitor build logs in Space settings
# Deployment takes ~3-5 minutes
```

## Troubleshooting

### HoverCard Not Found

**Error**: `Cannot find module '@/components/ui/hover-card'`

**Solution**:
```bash
cd frontend
npx shadcn@latest add hover-card
```

### Font Size Not Applying

**Error**: Font size changes don't affect text

**Solution**:
- Check `--content-font-size` CSS variable in index.css
- Verify `.prose` class applied to note body
- Check browser console for CSS errors

### TOC Not Extracting Headings

**Error**: TOC panel shows "No headings found" but note has headings

**Solution**:
- Verify headings use markdown syntax (#, ##, ###)
- Check custom renderer in markdown.tsx
- Check browser console for extraction errors

### Animations Janky

**Error**: Transitions stutter or drop frames

**Solution**:
- Use transform/opacity only (not width/height)
- Check for layout thrashing in DevTools
- Reduce animation duration or complexity

### LocalStorage Not Persisting

**Error**: Preferences lost on reload

**Solution**:
- Check browser doesn't have localStorage disabled
- Verify localStorage.setItem() calls succeed
- Check browser console for quota errors

## Rollback Plan

If issues arise in production:

```bash
# Revert to previous working state
git checkout master

# Or revert specific commit
git revert <commit-hash>

# Push to trigger redeployment
git push origin master
```

## Next Steps

After implementation and testing:

1. Run `/speckit.tasks` to generate detailed task breakdown
2. Implement features in priority order
3. Manual test after each feature
4. Commit incrementally (one feature per commit)
5. Merge to master when P1+P2 complete
6. Deploy to production
7. Monitor for issues

## Support Resources

- [React Docs](https://react.dev)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [shadcn/ui Components](https://ui.shadcn.com)
- [react-markdown Docs](https://github.com/remarkjs/react-markdown)
- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)

---

**Ready to Start**: All setup complete, begin with P1 features! 🚀
