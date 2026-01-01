/**
 * T072: Extract wikilinks from markdown text
 * Matches [[link text]] pattern
 */
export function extractWikilinks(text: string): string[] {
  const pattern = /\[\[([^\]]+)\]\]/g;
  const matches: string[] = [];
  let match;
  
  while ((match = pattern.exec(text)) !== null) {
    matches.push(match[1]);
  }
  
  return matches;
}

/**
 * T073: Normalize text to a URL-safe slug
 * - Lowercase
 * - Replace spaces and underscores with dashes
 * - Strip non-alphanumeric except dashes and forward slashes (to preserve paths)
 */
export function normalizeSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\s_]+/g, '-')
    .replace(/[^a-z0-9-\/]/g, '')  // Preserve forward slashes for path matching
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Parse wikilink syntax in markdown text
 * Returns array of { text, linkText } objects
 */
export function parseWikilinks(text: string): Array<{ text: string; linkText: string | null }> {
  const parts: Array<{ text: string; linkText: string | null }> = [];
  const pattern = /\[\[([^\]]+)\]\]/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    // Add text before the wikilink
    if (match.index > lastIndex) {
      parts.push({
        text: text.slice(lastIndex, match.index),
        linkText: null,
      });
    }

    // Add the wikilink
    parts.push({
      text: match[0],
      linkText: match[1],
    });

    lastIndex = pattern.lastIndex;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({
      text: text.slice(lastIndex),
      linkText: null,
    });
  }

  return parts;
}

/**
 * Convert wikilink text to a probable note path
 * Adds .md extension and normalizes
 */
export function wikilinkToPath(linkText: string): string {
  const slug = normalizeSlug(linkText);
  return `${slug}.md`;
}

