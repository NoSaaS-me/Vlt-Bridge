/**
 * T074: Markdown rendering configuration and wikilink handling
 */
import React from 'react';
import type { Components } from 'react-markdown';

export interface WikilinkComponentProps {
  linkText: string;
  resolved: boolean;
  onClick?: (linkText: string) => void;
}

/**
 * Custom renderer for wikilinks in markdown
 */
export function createWikilinkComponent(
  onWikilinkClick?: (linkText: string) => void
): Components {
  return {
    // Override the text renderer to handle wikilinks
    text: ({ value }) => {
      const parts: React.ReactNode[] = [];
      const pattern = /\[\[([^\]]+)\]\]/g;
      let lastIndex = 0;
      let match;
      let key = 0;

      while ((match = pattern.exec(value)) !== null) {
        // Add text before the wikilink
        if (match.index > lastIndex) {
          parts.push(value.slice(lastIndex, match.index));
        }

        // Add the wikilink as a clickable element
        const linkText = match[1];
        parts.push(
          React.createElement(
            'span',
            {
              key: key++,
              className: 'wikilink cursor-pointer text-primary hover:underline',
              onClick: (e: React.MouseEvent) => {
                e.preventDefault();
                onWikilinkClick?.(linkText);
              },
              role: 'link',
              tabIndex: 0,
              onKeyDown: (e: React.KeyboardEvent) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onWikilinkClick?.(linkText);
                }
              },
            },
            `[[${linkText}]]`
          )
        );

        lastIndex = pattern.lastIndex;
      }

      // Add remaining text
      if (lastIndex < value.length) {
        parts.push(value.slice(lastIndex));
      }

      return parts.length > 0 ? React.createElement(React.Fragment, {}, ...parts) : value;
    },

    // Style code blocks
    code: ({ className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      const isInline = !match;

      if (isInline) {
        return React.createElement(
          'code',
          {
            className: 'bg-muted px-1.5 py-0.5 rounded text-sm font-mono',
            ...props,
          },
          children
        );
      }

      return React.createElement(
        'code',
        {
          className: `${className} block bg-muted p-4 rounded-lg overflow-x-auto text-sm font-mono`,
          ...props,
        },
        children
      );
    },

    // Style links
    a: ({ href, children, ...props }: any) => {
      const isExternal = href?.startsWith('http');
      return React.createElement(
        'a',
        {
          href,
          className: 'text-primary hover:underline',
          target: isExternal ? '_blank' : undefined,
          rel: isExternal ? 'noopener noreferrer' : undefined,
          ...props,
        },
        children
      );
    },

    // Style headings
    h1: ({ children, ...props }: any) =>
      React.createElement('h1', { className: 'text-3xl font-bold mt-6 mb-4', ...props }, children),
    h2: ({ children, ...props }: any) =>
      React.createElement('h2', { className: 'text-2xl font-semibold mt-5 mb-3', ...props }, children),
    h3: ({ children, ...props }: any) =>
      React.createElement('h3', { className: 'text-xl font-semibold mt-4 mb-2', ...props }, children),

    // Style lists
    ul: ({ children, ...props }: any) =>
      React.createElement('ul', { className: 'list-disc list-inside my-2 space-y-1', ...props }, children),
    ol: ({ children, ...props }: any) =>
      React.createElement('ol', { className: 'list-decimal list-inside my-2 space-y-1', ...props }, children),

    // Style blockquotes
    blockquote: ({ children, ...props }: any) =>
      React.createElement('blockquote', { className: 'border-l-4 border-muted-foreground pl-4 italic my-4', ...props }, children),

    // Style tables
    table: ({ children, ...props }: any) =>
      React.createElement(
        'div',
        { className: 'overflow-x-auto my-4' },
        React.createElement('table', { className: 'min-w-full border-collapse border border-border', ...props }, children)
      ),
    th: ({ children, ...props }: any) =>
      React.createElement('th', { className: 'border border-border px-4 py-2 bg-muted font-semibold text-left', ...props }, children),
    td: ({ children, ...props }: any) =>
      React.createElement('td', { className: 'border border-border px-4 py-2', ...props }, children),
  };
}

/**
 * Render broken wikilinks with distinct styling
 */
export function renderBrokenWikilink(
  linkText: string,
  onCreate?: () => void
): React.ReactElement {
  return React.createElement(
    'span',
    {
      className: 'wikilink-broken text-destructive border-b border-dashed border-destructive cursor-pointer hover:bg-destructive/10',
      onClick: onCreate,
      role: 'link',
      tabIndex: 0,
      title: `Note "${linkText}" not found. Click to create.`,
    },
    `[[${linkText}]]`
  );
}
