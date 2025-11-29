# Feature Specification: UI Polish Pack

**Feature Branch**: `006-ui-polish`
**Created**: 2025-11-28
**Status**: Draft
**Input**: User description: "Font size adjuster, small, medium (default), and large. Smooth transitions. I am also thinking particle effects when clicking on something, lets put this as a stretch goal as it might be buggy to do right. Expand and collapse all folders button. Wikilink preview tooltip. Reading time estimator. Table of contents as a fly out pane."

## User Scenarios & Testing

### User Story 1 - Adjust Reading Comfort (Priority: P1)

As a user with vision preferences or accessibility needs, I want to adjust the font size of note content so that I can read documentation comfortably without straining my eyes or using browser zoom.

**Why this priority**: Accessibility is critical and affects all users reading notes. This is the simplest feature to implement and delivers immediate value independently.

**Independent Test**: Can be fully tested by opening any note, clicking the font size buttons (A-, A, A+), and verifying text resizes. Delivers immediate reading comfort value without requiring other features.

**Acceptance Scenarios**:

1. **Given** I am viewing a note, **When** I click the "A+" button, **Then** the note body text increases to large size (18px) and my preference is saved
2. **Given** I am viewing a note with large text, **When** I click the "A-" button, **Then** the note body text decreases to small size (14px)
3. **Given** I have set a font size preference, **When** I navigate to a different note or refresh the page, **Then** my font size preference is retained
4. **Given** I am viewing a note, **When** I click the "A" button, **Then** the note body text resets to default size (16px)

---

### User Story 2 - Navigate Large Documentation Structures (Priority: P1)

As a user working with a vault containing many nested folders, I want to quickly expand all folders or collapse them all so that I can find notes faster without manually clicking each folder.

**Why this priority**: Critical for usability in large vaults. Users with 50+ notes in nested folders waste significant time navigating. Can be tested independently by creating a test vault with nested folders.

**Independent Test**: Can be fully tested by creating a vault with 3+ levels of nested folders, clicking "Expand All" to see entire tree, then "Collapse All" to hide all folders. Delivers navigation efficiency value independently.

**Acceptance Scenarios**:

1. **Given** I have a vault with multiple nested folders, **When** I click "Expand All", **Then** all folder nodes in the directory tree expand to show their contents
2. **Given** all folders are expanded, **When** I click "Collapse All", **Then** all folder nodes close, hiding their contents
3. **Given** some folders are expanded and some collapsed, **When** I click "Expand All", **Then** all folders expand regardless of their previous state
4. **Given** I have folders expanded, **When** I click "Collapse All", **Then** the tree returns to showing only top-level items

---

### User Story 3 - Preview Note Content Without Navigation (Priority: P2)

As a user exploring documentation with many cross-references, I want to hover over a wikilink and see a preview of the linked note's content so that I can decide whether to navigate without losing my current reading context.

**Why this priority**: Significantly improves information browsing efficiency, but not essential for basic reading. Requires wikilink detection to be working, making it dependent on core functionality.

**Independent Test**: Can be fully tested by creating notes with wikilinks, hovering over links for 500ms, and verifying preview appears with first 150 characters. Delivers context-aware navigation value independently.

**Acceptance Scenarios**:

1. **Given** I am viewing a note containing a wikilink, **When** I hover my mouse over the wikilink for 500 milliseconds, **Then** a tooltip appears showing the first 150 characters of the linked note
2. **Given** a wikilink preview tooltip is displayed, **When** I move my mouse away, **Then** the tooltip disappears
3. **Given** a wikilink points to a non-existent note, **When** I hover over it, **Then** the tooltip shows "Note not found" or similar message
4. **Given** I quickly move my mouse across multiple wikilinks, **When** I don't pause for 500ms on any link, **Then** no tooltips appear (prevents flickering)

---

### User Story 4 - Estimate Reading Time (Priority: P2)

As a user planning my documentation reading sessions, I want to see an estimated reading time for each note so that I can decide whether to read now or bookmark for later.

**Why this priority**: Helpful for time management but not critical for basic functionality. Very quick to implement and adds professional polish.

**Independent Test**: Can be fully tested by opening notes of varying lengths and verifying reading time badge appears with accurate estimates. Delivers time-planning value independently.

**Acceptance Scenarios**:

1. **Given** I am viewing a note with 600 words, **When** the note loads, **Then** I see a badge showing "~3 min read" near the note title
2. **Given** I am viewing a note with fewer than 200 words, **When** the note loads, **Then** no reading time badge is displayed (< 1 minute threshold)
3. **Given** I am viewing a note with 1500 words, **When** the note loads, **Then** I see "~8 min read" (1500 / 200 = 7.5, rounded up)

---

### User Story 5 - Navigate Long Technical Documentation (Priority: P2)

As a user reading lengthy technical documentation, I want to see a table of contents with clickable headings so that I can quickly jump to specific sections without scrolling through the entire document.

**Why this priority**: Very valuable for long documents but only applies to a subset of notes. More complex to implement than other features.

**Independent Test**: Can be fully tested by opening a long note with H1-H3 headings, toggling the TOC panel, clicking headings to scroll, and verifying persistence. Delivers section navigation value independently.

**Acceptance Scenarios**:

1. **Given** I am viewing a note with multiple headings, **When** I click the "TOC" button in the toolbar, **Then** a flyout panel appears on the right side showing all H1, H2, and H3 headings
2. **Given** the TOC panel is open, **When** I click on a heading in the TOC, **Then** the note scrolls smoothly to that section
3. **Given** the TOC panel is open, **When** I click the "TOC" button again or click outside the panel, **Then** the panel closes
4. **Given** I have opened the TOC panel, **When** I close it and reload the page, **Then** the panel state (open/closed) is remembered
5. **Given** I am viewing a note with no headings, **When** I click "TOC", **Then** the panel shows "No headings found" message

---

### User Story 6 - Experience Polished Interface (Priority: P3)

As a user navigating between notes and views, I want to see smooth transitions and animations so that the interface feels polished and professional rather than abrupt.

**Why this priority**: Nice-to-have polish that improves perceived quality but doesn't add functional value. Should be implemented last.

**Independent Test**: Can be fully tested by switching between notes, toggling graph view, and selecting tree items to verify animations play. Delivers visual polish value independently.

**Acceptance Scenarios**:

1. **Given** I am viewing one note, **When** I select a different note, **Then** the new note content fades in over 300 milliseconds
2. **Given** I am in note view, **When** I toggle to graph view, **Then** the graph slides in over 250 milliseconds
3. **Given** I click on a directory tree item, **When** the item becomes selected, **Then** it highlights with a smooth transition

---

### User Story 7 - Visual Click Feedback (Priority: STRETCH)

As a user clicking on interactive elements, I want to see playful particle effects so that the interface feels more engaging and responsive to my actions.

**Why this priority**: Stretch goal - adds delight but risks feeling gimmicky or causing performance issues. Only implement if time permits and other features are stable.

**Independent Test**: Can be fully tested by clicking various UI elements (links, buttons, tree items) and verifying particle animations appear without lag. Delivers engagement value independently but has higher risk of bugs.

**Acceptance Scenarios**:

1. **Given** I am using the interface, **When** I click on a wikilink, **Then** a small particle burst animation appears at the click point
2. **Given** I am using the interface, **When** I click on a tree item, **Then** particle effects appear without causing UI lag or freezing
3. **Given** particle effects are enabled, **When** I perform rapid clicks, **Then** the system gracefully handles multiple simultaneous animations

---

### Edge Cases

- What happens when a user sets font size to large and then views a note with very long lines? (Text should wrap properly, not overflow)
- How does the TOC handle notes with duplicate heading text? (Should still scroll to correct position based on document order)
- What happens when a wikilink preview is requested for a very large note? (Show only first 150 characters, no performance impact)
- How does "Expand All" perform with 100+ folders? (Should complete in under 2 seconds, show loading state if needed)
- What happens when a user hovers on a wikilink but the note hasn't been indexed yet? (Show "Loading..." then content or error)

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide three font size options: Small (14px), Medium (16px), and Large (18px) for note body text
- **FR-002**: System MUST persist user's font size preference across browser sessions using localStorage
- **FR-003**: System MUST display "Expand All" and "Collapse All" buttons above the directory tree
- **FR-004**: "Expand All" button MUST open all folder nodes in the directory tree regardless of current state
- **FR-005**: "Collapse All" button MUST close all folder nodes in the directory tree
- **FR-006**: System MUST display a preview tooltip when user hovers over a wikilink for 500 milliseconds
- **FR-007**: Wikilink preview tooltip MUST show the first 150 characters of the target note's body content
- **FR-008**: System MUST calculate reading time as word count divided by 200 words per minute
- **FR-009**: System MUST display reading time badge only for notes estimated to take 1 minute or longer
- **FR-010**: System MUST generate a table of contents from H1, H2, and H3 headings in the current note
- **FR-011**: TOC panel MUST be toggleable via a "TOC" button in the note viewer toolbar
- **FR-012**: Clicking a TOC heading MUST smoothly scroll the note to that section
- **FR-013**: TOC panel state (open/closed) MUST persist across page reloads using localStorage
- **FR-014**: Note content transitions MUST use fade-in animation (300ms duration)
- **FR-015**: Graph view transitions MUST use slide-in animation (250ms duration)
- **FR-016**: Font size changes MUST only affect note body text, not UI chrome (sidebar, toolbar, headers)
- **FR-017**: Wikilink preview MUST not appear if mouse moves away before 500ms delay
- **FR-018** (STRETCH): System MAY display particle burst effects on click events for wikilinks, buttons, and tree items

### Key Entities

- **Font Size Preference**: User's selected text size (small/medium/large), persisted in localStorage
- **TOC Panel State**: Boolean indicating whether TOC panel is open or closed, persisted in localStorage
- **TOC Heading**: Extracted heading from markdown with text, level (H1/H2/H3), and scroll position
- **Reading Time Estimate**: Calculated value in minutes based on note word count

## Success Criteria

### Measurable Outcomes

- **SC-001**: Users can adjust font size and see the change applied instantly (< 100ms)
- **SC-002**: Font size preference persists correctly across 100% of browser sessions
- **SC-003**: "Expand All" completes in under 2 seconds for vaults with up to 100 folders
- **SC-004**: Wikilink preview tooltips appear within 100ms after the 500ms hover delay
- **SC-005**: Reading time estimates are accurate within ±20% of actual reading time for 90% of notes
- **SC-006**: TOC generation completes in under 500ms for notes with up to 50 headings
- **SC-007**: All transitions and animations complete without visual jank or frame drops (60 FPS)
- **SC-008**: TOC panel state persists correctly across 100% of page reloads
- **SC-009**: Users can navigate to any section in a 20-heading document in under 3 seconds using TOC
- **SC-010**: Rapid hovering over multiple wikilinks does not cause tooltip flickering or performance degradation

## Assumptions

- Users primarily read documentation rather than edit it, so reading comfort features are valuable
- Most vaults contain between 10-100 notes with 2-4 levels of folder nesting
- Average reading speed is 200 words per minute (industry standard)
- Users are accessing the application via modern browsers with localStorage support
- Notes use standard markdown heading syntax (#, ##, ###)
- Font size preference applies globally to all notes, not per-note
- Wikilink previews fetching note content from existing API endpoints is acceptable (no new backend changes required)
- Particle effects (stretch goal) would use CSS-based animations or lightweight canvas library
- TOC panel appears on right side of note viewer, overlaying content on smaller screens
- Smooth transitions use CSS transitions or React spring animations (existing capabilities)
