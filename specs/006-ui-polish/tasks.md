---

description: "Implementation tasks for UI Polish Pack feature"
---

# Tasks: UI Polish Pack

**Input**: Design documents from `/specs/006-ui-polish/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: No automated tests - manual verification per quickstart.md checklist

**Organization**: Tasks are grouped by user story (P1 → P2 → P3 → STRETCH) to enable independent implementation and testing of each polish feature.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `frontend/src/` for all React/TypeScript code
- No backend changes required for this feature

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Install dependencies and prepare development environment

- [x] T001 Install shadcn/ui HoverCard component via `npx shadcn@latest add hover-card` in frontend/
- [x] T002 [P] Verify tailwindcss-animate is installed in frontend/package.json (existing dependency)
- [x] T003 [P] Verify development server runs successfully via `npm run dev` in frontend/

**Checkpoint**: Dependencies ready, dev server running

---

## Phase 2: Foundational (No blocking prerequisites)

**Purpose**: This feature has NO foundational blocking tasks - all user stories are independent frontend polish features

**⚠️ NOTE**: User story implementation can begin immediately after Phase 1

**Checkpoint**: Foundation ready - proceed directly to user stories

---

## Phase 3: User Story 1 - Adjust Reading Comfort (Priority: P1) 🎯 MVP

**Goal**: Allow users to adjust font size (small/medium/large) with localStorage persistence for comfortable reading

**Independent Test**: Open any note, click A-/A/A+ buttons, verify text resizes instantly and persists after page reload

### Implementation for User Story 1

- [x] T004 [P] [US1] Add CSS custom property `--content-font-size: 1rem;` to `:root` in frontend/src/index.css
- [x] T005 [P] [US1] Apply `font-size: var(--content-font-size)` to `.prose` class in frontend/src/index.css
- [x] T006 [US1] Create useFontSize hook in frontend/src/hooks/useFontSize.ts with localStorage persistence
- [x] T007 [US1] Add font size state management to MainApp component in frontend/src/pages/MainApp.tsx
- [x] T008 [US1] Pass fontSize props to NoteViewer in frontend/src/pages/MainApp.tsx
- [x] T009 [US1] Add font size buttons (A-, A, A+) to NoteViewer toolbar in frontend/src/components/NoteViewer.tsx
- [x] T010 [US1] Update CSS variable dynamically when font size changes in frontend/src/hooks/useFontSize.ts
- [x] T011 [US1] Verify font size only affects .prose content, not UI chrome (manual test)

**Checkpoint**: Font size adjuster complete - text resizes with buttons, preference persists, UI chrome unaffected

---

## Phase 4: User Story 2 - Navigate Large Documentation Structures (Priority: P1)

**Goal**: Provide "Expand All" and "Collapse All" buttons for directory tree navigation in large vaults

**Independent Test**: Create vault with 3+ levels of nested folders, click Expand All to see entire tree, click Collapse All to hide all folders

### Implementation for User Story 2

- [x] T012 [P] [US2] Add expandAll state to DirectoryTree component in frontend/src/components/DirectoryTree.tsx
- [x] T013 [P] [US2] Add collapseAll state to DirectoryTree component in frontend/src/components/DirectoryTree.tsx
- [x] T014 [US2] Add forceExpandState prop to TreeNodeItem recursive component in frontend/src/components/DirectoryTree.tsx
- [x] T015 [US2] Implement expand/collapse state propagation logic in frontend/src/components/DirectoryTree.tsx
- [x] T016 [US2] Add "Expand All" button above directory tree in frontend/src/components/DirectoryTree.tsx
- [x] T017 [US2] Add "Collapse All" button above directory tree in frontend/src/components/DirectoryTree.tsx
- [x] T018 [US2] Verify expand all completes in <2s for 100+ folders (performance test)

**Checkpoint**: Expand/collapse all buttons work, all folders respond to state changes, performance acceptable

---

## Phase 5: User Story 3 - Preview Note Content Without Navigation (Priority: P2)

**Goal**: Show HoverCard preview with first 150 characters of linked note when hovering over wikilink for 500ms

**Independent Test**: Create notes with wikilinks, hover over links for 500ms, verify preview appears with first 150 characters

### Implementation for User Story 3

- [x] T019 [P] [US3] Create wikilink preview cache state (Map<string, string>) in markdown.tsx or NoteViewer component
- [x] T020 [P] [US3] Import HoverCard component in frontend/src/lib/markdown.tsx
- [x] T021 [US3] Wrap wikilink spans with HoverCard in wikilink renderer in frontend/src/lib/markdown.tsx
- [x] T022 [US3] Set HoverCard openDelay={500} and closeDelay={100} in frontend/src/lib/markdown.tsx
- [x] T023 [US3] Implement preview fetch logic using existing GET /api/notes/{path} endpoint
- [x] T024 [US3] Extract first 150 characters from note body for preview display
- [x] T025 [US3] Add loading skeleton during preview fetch in HoverCard content
- [x] T026 [US3] Handle broken wikilinks (show "Note not found" message)
- [x] T027 [US3] Verify no tooltips appear when mouse moves away before 500ms (flicker prevention test)
- [x] T028 [US3] Verify preview cache works (second hover on same link is instant)

**Checkpoint**: Wikilink previews appear after 500ms hover, show correct content, handle errors gracefully

---

## Phase 6: User Story 4 - Estimate Reading Time (Priority: P2)

**Goal**: Display reading time badge ("X min read") for notes estimated to take 1+ minutes at 200 WPM

**Independent Test**: Open notes of varying lengths, verify reading time badge appears with accurate estimates

### Implementation for User Story 4

- [x] T029 [P] [US4] Import markdownToPlainText utility in frontend/src/components/NoteViewer.tsx
- [x] T030 [P] [US4] Import Badge component from shadcn/ui in frontend/src/components/NoteViewer.tsx
- [x] T031 [US4] Create useMemo hook to calculate reading time from note.body in frontend/src/components/NoteViewer.tsx
- [x] T032 [US4] Extract word count: plainText.trim().split(/\\s+/).length in reading time calculation
- [x] T033 [US4] Calculate minutes: Math.ceil(wordCount / 200) in reading time calculation
- [x] T034 [US4] Return null if <1 minute (200 words threshold) in reading time calculation
- [x] T035 [US4] Render Badge with "X min read" near note title in frontend/src/components/NoteViewer.tsx
- [x] T036 [US4] Verify badge appears only for notes >200 words (manual test)

**Checkpoint**: Reading time badge displays for long notes, hidden for short notes, estimates accurate within ±20%

---

## Phase 7: User Story 5 - Navigate Long Technical Documentation (Priority: P2)

**Goal**: Provide table of contents flyout panel with clickable headings (H1-H3) for long notes

**Independent Test**: Open long note with H1-H3 headings, toggle TOC panel, click headings to scroll, verify persistence

### Implementation for User Story 5

- [x] T037 [P] [US5] Create useTableOfContents hook in frontend/src/hooks/useTableOfContents.ts
- [x] T038 [P] [US5] Create TableOfContents component in frontend/src/components/TableOfContents.tsx
- [x] T039 [US5] Add heading ID generation to h1/h2/h3 renderers in frontend/src/lib/markdown.tsx
- [x] T040 [US5] Implement slugify function: text.toLowerCase().replace(/\\s+/g, '-').replace(/[^\\w-]/g, '') in markdown.tsx
- [x] T041 [US5] Extract headings { id, text, level }[] using useRef during render in useTableOfContents hook
- [x] T042 [US5] Add TOC panel state (isOpen) to NoteViewer in frontend/src/components/NoteViewer.tsx
- [x] T043 [US5] Persist TOC panel state to localStorage key 'toc-panel-open' in useTableOfContents hook
- [x] T044 [US5] Add "TOC" button to NoteViewer toolbar in frontend/src/components/NoteViewer.tsx
- [x] T045 [US5] Render TableOfContents component as ResizablePanel (right sidebar) in frontend/src/components/NoteViewer.tsx
- [x] T046 [US5] Implement scrollToHeading function with smooth scroll behavior in useTableOfContents hook
- [x] T047 [US5] Respect prefers-reduced-motion media query in scroll behavior in useTableOfContents hook
- [x] T048 [US5] Add hierarchical indentation by heading level in TableOfContents component
- [x] T049 [US5] Show "No headings found" message when headings array is empty in TableOfContents component
- [x] T050 [US5] Handle duplicate heading text by appending -2, -3 to IDs in slugify function
- [x] T051 [US5] Verify TOC panel state persists after reload (manual test)
- [x] T052 [US5] Verify TOC generation completes in <500ms for 50 headings (performance test)

**Checkpoint**: TOC panel toggles, headings are clickable, smooth scrolling works, state persists, performance acceptable

---

## Phase 8: User Story 6 - Experience Polished Interface (Priority: P3)

**Goal**: Add smooth CSS transitions/animations to note switching, graph toggle, and tree selection

**Independent Test**: Switch between notes, toggle graph view, select tree items to verify animations play smoothly at 60 FPS

### Implementation for User Story 6

- [x] T053 [P] [US6] Extend Tailwind config with custom transition utilities in frontend/tailwind.config.js
- [x] T054 [P] [US6] Add fade-in transition keyframe (300ms) in frontend/tailwind.config.js
- [x] T055 [P] [US6] Add slide-in transition keyframe (250ms) in frontend/tailwind.config.js
- [x] T056 [US6] Apply fade-in transition to note content container in frontend/src/components/NoteViewer.tsx
- [x] T057 [US6] Apply slide-in transition to graph view toggle in frontend/src/pages/MainApp.tsx
- [x] T058 [US6] Apply smooth transition to directory tree item selection in frontend/src/components/DirectoryTree.tsx
- [x] T059 [US6] Add prefers-reduced-motion media query support to all transitions
- [x] T060 [US6] Verify animations maintain 60 FPS (Chrome DevTools Performance tab test)
- [x] T061 [US6] Verify animations only use transform/opacity (no layout thrashing)

**Checkpoint**: All transitions smooth, 60 FPS maintained, reduced motion respected

---

## Phase 9: User Story 7 - Visual Click Feedback (Priority: STRETCH)

**Goal**: Add playful particle effects on click events for wikilinks, buttons, and tree items

**Independent Test**: Click various UI elements, verify particle animations appear without lag

**⚠️ STRETCH**: Only implement if time permits and P1/P2/P3 features are stable

### Implementation for User Story 7

- [ ] T062 [P] [US7] Decide implementation approach: CSS-only vs shadcn particles component
- [ ] T063 [P] [US7] If CSS-only: Create particle animation keyframes in frontend/src/index.css
- [ ] T064 [P] [US7] If shadcn: Install particles component via `npx shadcn@latest add particles`
- [ ] T065 [US7] Add particle effect trigger to wikilink click handler in frontend/src/lib/markdown.tsx
- [ ] T066 [US7] Add particle effect trigger to tree item click handler in frontend/src/components/DirectoryTree.tsx
- [ ] T067 [US7] Add particle effect trigger to button click handlers in NoteViewer/MainApp
- [ ] T068 [US7] Limit particle count to 50 max for performance
- [ ] T069 [US7] Disable particles on mobile devices (media query)
- [ ] T070 [US7] Respect prefers-reduced-motion (disable particles if set)
- [ ] T071 [US7] Verify no lag or frame drops during rapid clicking (stress test)
- [ ] T072 [US7] Verify particles use transform-only animations (no layout thrashing)

**Checkpoint**: Particle effects enhance engagement without performance degradation

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and cross-story polish

- [ ] T073 [P] Run complete quickstart.md testing checklist for all implemented stories
- [ ] T074 [P] Test font size adjuster checklist (7 items) from quickstart.md
- [ ] T075 [P] Test expand/collapse all checklist (6 items) from quickstart.md
- [ ] T076 [P] Test wikilink preview tooltips checklist (7 items) from quickstart.md
- [ ] T077 [P] Test reading time estimator checklist (5 items) from quickstart.md
- [ ] T078 [P] Test table of contents checklist (10 items) from quickstart.md
- [ ] T079 [P] Test smooth transitions checklist (5 items) from quickstart.md
- [ ] T080 [P] Test particle effects checklist (5 items) from quickstart.md (if implemented)
- [ ] T081 [P] Browser compatibility testing: Chrome, Firefox, Safari, Edge
- [ ] T082 [P] Mobile responsiveness testing: iOS Safari, Android Chrome
- [ ] T083 [P] Accessibility verification: keyboard navigation, ARIA labels, WCAG 2.2 Level AA
- [ ] T084 [P] Performance profiling: Chrome DevTools Performance tab
- [ ] T085 [P] Memory profiling: Check for leaks in preview cache and event listeners
- [x] T086 Build production bundle via `npm run build` in frontend/
- [x] T087 Preview production build via `npm run preview` and re-test all features
- [x] T088 Verify bundle size is reasonable (~400-500KB gzipped)
- [ ] T089 Update CLAUDE.md if new patterns established (already done in planning phase)
- [ ] T090 Commit final changes with message per git commit guidelines

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: N/A - no blocking foundational tasks for this feature
- **User Stories (Phase 3-9)**: Each user story is independent and can start after Phase 1
  - US1 (Font Size): Can start immediately after Phase 1
  - US2 (Expand/Collapse): Can start immediately after Phase 1
  - US3 (Wikilink Preview): Can start immediately after Phase 1
  - US4 (Reading Time): Can start immediately after Phase 1
  - US5 (Table of Contents): Can start immediately after Phase 1
  - US6 (Transitions): Can start immediately after Phase 1
  - US7 (Particles): STRETCH - only if time permits after P1/P2/P3
- **Polish (Phase 10)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: INDEPENDENT - No dependencies on other stories
- **User Story 2 (P1)**: INDEPENDENT - No dependencies on other stories
- **User Story 3 (P2)**: INDEPENDENT - No dependencies on other stories
- **User Story 4 (P2)**: INDEPENDENT - No dependencies on other stories
- **User Story 5 (P2)**: INDEPENDENT - No dependencies on other stories
- **User Story 6 (P3)**: INDEPENDENT - No dependencies on other stories (applies transitions to existing UI)
- **User Story 7 (STRETCH)**: INDEPENDENT - No dependencies on other stories

### Within Each User Story

- Tasks within a story may have sequential dependencies (e.g., create hook before using it)
- Tasks marked [P] can run in parallel (different files)
- Complete story checkpoint before moving to next priority

### Parallel Opportunities

- **Phase 1**: T002 and T003 can run in parallel with T001
- **User Stories**: All 7 user stories can be implemented in parallel by different developers (after Phase 1)
- **Within Stories**: Tasks marked [P] can run in parallel
  - US1: T004 and T005 (CSS changes) can run in parallel
  - US2: T012 and T013 (state additions) can run in parallel
  - US3: T019 and T020 (imports) can run in parallel
  - US4: T029 and T030 (imports) can run in parallel
  - US5: T037 and T038 (hook + component scaffolding) can run in parallel
  - US6: T053, T054, T055 (Tailwind config) can run in parallel
  - US7: T062, T063, T064 (approach decision) can run in parallel
- **Polish (Phase 10)**: T073-T085 (all testing tasks) can run in parallel

---

## Parallel Example: User Story 1 (Font Size)

```bash
# Launch CSS changes in parallel:
Task T004: "Add CSS custom property --content-font-size to :root in frontend/src/index.css"
Task T005: "Apply font-size: var(--content-font-size) to .prose class in frontend/src/index.css"

# Then create hook (sequential - needed before component integration):
Task T006: "Create useFontSize hook in frontend/src/hooks/useFontSize.ts"

# Then integrate into components (sequential - depends on hook):
Task T007: "Add font size state management to MainApp"
Task T008: "Pass fontSize props to NoteViewer"
Task T009: "Add font size buttons to NoteViewer toolbar"
```

---

## Parallel Example: User Story 5 (Table of Contents)

```bash
# Launch hook and component scaffolding in parallel:
Task T037: "Create useTableOfContents hook in frontend/src/hooks/useTableOfContents.ts"
Task T038: "Create TableOfContents component in frontend/src/components/TableOfContents.tsx"

# Then implement markdown heading extraction (sequential):
Task T039: "Add heading ID generation to h1/h2/h3 renderers"
Task T040: "Implement slugify function"
Task T041: "Extract headings using useRef during render"

# Then integrate (sequential - depends on hook + component):
Task T042-T052: "Integrate TOC panel into NoteViewer with state/persistence/scrolling"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only - Both P1)

1. Complete Phase 1: Setup (~5 min)
2. Complete Phase 3: User Story 1 - Font Size (~15 min)
3. **STOP and VALIDATE**: Test font size independently per quickstart.md
4. Complete Phase 4: User Story 2 - Expand/Collapse All (~20 min)
5. **STOP and VALIDATE**: Test expand/collapse independently per quickstart.md
6. Deploy/demo if ready (MVP with 2 core features)

**Total MVP Time**: ~40 minutes for P1 features only

### Incremental Delivery (Priority Order)

1. **Setup (Phase 1)** → Dependencies ready (~5 min)
2. **P1 Features (Phases 3-4)** → Font Size + Expand/Collapse → Test → Deploy (~40 min total)
3. **P2 Features (Phases 5-7)** → Wikilink Preview + Reading Time + TOC → Test → Deploy (~80 min total)
4. **P3 Features (Phase 8)** → Smooth Transitions → Test → Deploy (~10 min)
5. **STRETCH (Phase 9)** → Particle Effects → Test → Deploy (~45 min) - OPTIONAL
6. **Polish (Phase 10)** → Comprehensive testing + build → Deploy (~30 min)

**Total Time Estimate**:
- P1 only: ~40 min (MVP)
- P1 + P2: ~120 min (recommended hackathon scope)
- P1 + P2 + P3: ~130 min (full polish)
- All features (including STRETCH): ~175 min (if time permits)

### Parallel Team Strategy

With multiple developers (after Phase 1 setup):

**Option A: Priority-based (Recommended)**
1. Developer A: User Story 1 (Font Size) - P1
2. Developer B: User Story 2 (Expand/Collapse) - P1
3. After P1 complete and tested:
   - Developer A: User Story 3 (Wikilink Preview) - P2
   - Developer B: User Story 4 (Reading Time) - P2
   - Developer C: User Story 5 (Table of Contents) - P2

**Option B: Full parallel**
1. Developer A: User Stories 1 & 4 (Font Size + Reading Time)
2. Developer B: User Stories 2 & 6 (Expand/Collapse + Transitions)
3. Developer C: User Stories 3 & 5 (Wikilink Preview + TOC)
4. Developer D: User Story 7 (Particles - STRETCH)

**Option C: Serial (Single Developer)**
1. Follow priority order: US1 → US2 → US4 → US3 → US5 → US6 → US7 (STRETCH)
2. Stop after any checkpoint to demo/deploy incremental value

---

## Notes

- **[P] tasks**: Different files, no dependencies - safe to parallelize
- **[Story] label**: Maps task to specific user story for traceability
- **No tests**: Manual verification only per quickstart.md checklist
- **Independent stories**: Each story can be completed and tested without others
- **Frontend-only**: No backend changes required
- **Existing patterns**: Leverages CSS variables, React hooks, shadcn/ui, tailwindcss-animate
- **New dependencies**: Only @radix-ui/react-hover-card (installed in Phase 1)
- **Performance targets**: All documented in success criteria (SC-001 through SC-010)
- **Accessibility**: WCAG 2.2 Level AA compliance required
- **Commit strategy**: Commit after each completed user story checkpoint
- **Deployment**: Can deploy after any user story completion (incremental value)
- **STRETCH goal**: User Story 7 (Particles) only if P1/P2/P3 stable and time permits

---

**Recommended Hackathon Scope**: P1 + P2 features (User Stories 1-5) = ~120 min implementation + 30 min testing/polish = **2.5 hours total**

This delivers maximum polish impact with minimal risk. P3 (Transitions) and STRETCH (Particles) can be added if time remains.
