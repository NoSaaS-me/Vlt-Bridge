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
    // Style links
    a: ({ href, children, ...props }) => {
      if (href?.startsWith('wikilink:')) {
        const linkText = decodeURIComponent(href.replace('wikilink:', ''));
        return (
          <span
            className="wikilink cursor-pointer text-primary hover:underline font-medium text-blue-500 dark:text-blue-400"
            onClick={(e) => {
              e.preventDefault();
              onWikilinkClick?.(linkText);
            }}
            role="link"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onWikilinkClick?.(linkText);
              }
            }}
            title={`Go to ${linkText}`}
          >
            {children}
          </span>
        );
      }

      const isExternal = href?.startsWith('http');
      return (
        <a
          href={href}
          className="text-primary hover:underline"
          target={isExternal ? '_blank' : undefined}
          rel={isExternal ? 'noopener noreferrer' : undefined}
          {...props}
        >
          {children}
        </a>
      );
    },

    // Style headings
    h1: ({ children, ...props }) => (
      <h1 className="text-3xl font-bold mt-6 mb-4" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, ...props }) => (
      <h2 className="text-2xl font-semibold mt-5 mb-3" {...props}>
        {children}
      </h2>
    ),
    h3: ({ children, ...props }) => (
      <h3 className="text-xl font-semibold mt-4 mb-2" {...props}>
        {children}
      </h3>
    ),

    // Style lists
    ul: ({ children, ...props }) => (
      <ul className="list-disc list-inside my-2 space-y-1" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }) => (
      <ol className="list-decimal list-inside my-2 space-y-1" {...props}>
        {children}
      </ol>
    ),

    // Style blockquotes
    blockquote: ({ children, ...props }) => (
      <blockquote className="border-l-4 border-muted-foreground pl-4 italic my-4" {...props}>
        {children}
      </blockquote>
    ),

    // Style tables
    table: ({ children, ...props }) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full border-collapse border border-border" {...props}>
          {children}
        </table>
      </div>
    ),
    th: ({ children, ...props }) => (
      <th className="border border-border px-4 py-2 bg-muted font-semibold text-left" {...props}>
        {children}
      </th>
    ),
    td: ({ children, ...props }) => (
      <td className="border border-border px-4 py-2" {...props}>
        {children}
      </td>
    ),
  };
}

/**
 * Render broken wikilinks with distinct styling
 */
export function renderBrokenWikilink(
  linkText: string,
  onCreate?: () => void
): React.ReactElement {
  return (
    <span
      className="wikilink-broken text-destructive border-b border-dashed border-destructive cursor-pointer hover:bg-destructive/10"
      onClick={onCreate}
      role="link"
      tabIndex={0}
      title={`Note "${linkText}" not found. Click to create.`}
    >
      [[{linkText}]]
    </span>
  );
}

