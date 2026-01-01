/**
 * SafeSnippet component - Safely renders search result snippets with highlighting
 *
 * This component parses snippet strings containing <mark> tags (from SQLite FTS5)
 * and renders them as React elements without using dangerouslySetInnerHTML.
 *
 * Security: All HTML content except <mark> tags is already escaped by the backend
 * sanitizer (backend/src/services/sanitizer.py), providing defense-in-depth.
 * This component provides an additional layer by rendering only the allowed
 * <mark> elements as React components.
 */

import { Fragment } from 'react';

interface SafeSnippetProps {
  /** The snippet text with <mark>...</mark> tags for highlighting */
  snippet: string;
  /** Optional className for the container element */
  className?: string;
}

/**
 * Parses a snippet string and extracts segments with their highlighting status.
 *
 * Algorithm:
 * 1. Split by <mark> opening tags
 * 2. For each segment after a <mark>, split by </mark> to separate highlighted from normal text
 * 3. Track nesting depth to handle malformed HTML gracefully
 *
 * @param snippet - The snippet string with <mark> tags
 * @returns Array of [text, isHighlighted] tuples
 */
function parseSnippet(snippet: string): Array<[string, boolean]> {
  const segments: Array<[string, boolean]> = [];

  // Split by <mark> tags
  const parts = snippet.split('<mark>');

  // First part is always non-highlighted (before any <mark>)
  if (parts[0]) {
    segments.push([parts[0], false]);
  }

  // Process remaining parts (each starts after a <mark>)
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i];
    const closeIndex = part.indexOf('</mark>');

    if (closeIndex !== -1) {
      // Found closing tag - split into highlighted and non-highlighted
      const highlighted = part.substring(0, closeIndex);
      const afterClose = part.substring(closeIndex + 7); // 7 = length of '</mark>'

      if (highlighted) {
        segments.push([highlighted, true]);
      }
      if (afterClose) {
        segments.push([afterClose, false]);
      }
    } else {
      // No closing tag - treat entire segment as highlighted (graceful degradation)
      if (part) {
        segments.push([part, true]);
      }
    }
  }

  return segments;
}

/**
 * Safely renders a search snippet with <mark> highlighting.
 *
 * Unlike dangerouslySetInnerHTML, this component:
 * - Parses the snippet string programmatically
 * - Renders only text nodes and <mark> elements
 * - Prevents execution of any injected scripts or HTML
 *
 * The backend sanitizer already escapes all HTML except <mark> tags,
 * so this provides defense-in-depth by explicitly only rendering the
 * allowed elements.
 */
export function SafeSnippet({ snippet, className }: SafeSnippetProps) {
  if (!snippet) {
    return null;
  }

  const segments = parseSnippet(snippet);

  return (
    <span className={className}>
      {segments.map(([text, isHighlighted], index) => (
        <Fragment key={index}>
          {isHighlighted ? <mark>{text}</mark> : text}
        </Fragment>
      ))}
    </span>
  );
}
