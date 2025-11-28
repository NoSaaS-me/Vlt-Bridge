/**
 * Convert markdown content into a plaintext string suitable for TTS.
 * Strips code blocks, images, and excess whitespace.
 */
export function markdownToPlainText(markdown: string): string {
  if (!markdown) return '';

  let text = markdown;

  // Remove fenced code blocks
  text = text.replace(/```[\\s\\S]*?```/g, '');
  // Remove inline code
  text = text.replace(/`([^`]*)`/g, '$1');
  // Remove images: ![alt](url)
  text = text.replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  // Replace markdown links with link text
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
  // Replace headings with emphasized text
  text = text.replace(/^(#{1,6})\s*(.*)$/gm, '$2');
  // Replace list markers with dash
  text = text.replace(/^\s*[-*+]\s+/gm, '- ');
  // Normalize whitespace
  text = text.replace(/\\s+\\n/g, '\\n').replace(/\\n{3,}/g, '\\n\\n');

  return text.trim();
}
