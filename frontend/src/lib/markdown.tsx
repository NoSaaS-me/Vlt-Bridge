/**
 * T074: Markdown rendering configuration and wikilink handling
 * T019-T028: Wikilink preview tooltips with HoverCard
 * T039-T040: Heading ID generation for Table of Contents
 */
import React, { useState } from 'react';
import type { Components } from 'react-markdown';
import { HoverCard, HoverCardTrigger, HoverCardContent } from '@/components/ui/hover-card';
import { Badge } from '@/components/ui/badge';
import { resolveWikilink, getNotePreview } from '@/services/api';
import type { NotePreview } from '@/types/note';
import { formatDistanceToNow } from '@/lib/utils';

export interface WikilinkComponentProps {
  linkText: string;
  resolved: boolean;
  onClick?: (linkText: string) => void;
}

/**
 * T019: Resolution cache - maps wikilink text to resolved note path
 * Longer-lived since note paths rarely change
 */
const resolutionCache = new Map<string, string | null>();

/**
 * T019: Preview cache - maps note path to preview data
 * Can be invalidated more frequently as note content changes
 */
const previewCache = new Map<string, NotePreview>();

/**
 * T039-T040: Track slugs to handle duplicates
 */
const slugCache = new Map<string, number>();

/**
 * T040: Slugify heading text to create valid HTML IDs
 * Handles duplicates by appending -2, -3, etc.
 */
function slugify(text: string): string {
  // Basic slugification
  const baseSlug = text
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w-]/g, '');

  // T050: Handle duplicates
  const count = slugCache.get(baseSlug) || 0;
  slugCache.set(baseSlug, count + 1);

  if (count === 0) {
    return baseSlug;
  }

  return `${baseSlug}-${count + 1}`;
}

/**
 * Reset slug cache (call when rendering a new document)
 */
export function resetSlugCache(): void {
  slugCache.clear();
}

/**
 * T021-T026: Wikilink preview component with HoverCard
 */
function WikilinkPreview({
  linkText,
  children,
  onClick
}: {
  linkText: string;
  children: React.ReactNode;
  onClick?: () => void;
}) {
  const [preview, setPreview] = useState<NotePreview | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBroken, setIsBroken] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  // T023: Fetch preview when hover card opens
  React.useEffect(() => {
    if (!isOpen) return;

    // Start loading
    setIsLoading(true);

    const fetchPreview = async () => {
      try {
        // Step 1: Resolve wikilink text to note path (with caching)
        let targetPath: string | null = null;

        if (resolutionCache.has(linkText)) {
          // Use cached resolution
          targetPath = resolutionCache.get(linkText)!;
        } else {
          // Resolve and cache the result
          const resolution = await resolveWikilink(linkText);
          targetPath = resolution.is_resolved ? resolution.target_path : null;
          resolutionCache.set(linkText, targetPath);
        }

        // Check if resolution failed (broken link)
        if (!targetPath) {
          setIsBroken(true);
          setPreview(null);
          setIsLoading(false);
          return;
        }

        // Step 2: Fetch preview data for resolved path (with caching)
        let previewData: NotePreview;

        if (previewCache.has(targetPath)) {
          // Use cached preview
          previewData = previewCache.get(targetPath)!;
        } else {
          // Fetch and cache preview data
          previewData = await getNotePreview(targetPath);
          previewCache.set(targetPath, previewData);
        }

        // Display the preview
        setPreview(previewData);
        setIsBroken(false);
      } catch (error) {
        // T026: Handle broken wikilinks
        setIsBroken(true);
        setPreview(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPreview();
  }, [isOpen, linkText]);

  return (
    <HoverCard openDelay={500} closeDelay={100} onOpenChange={setIsOpen}>
      <HoverCardTrigger asChild>
        <span onClick={onClick}>
          {children}
        </span>
      </HoverCardTrigger>
      <HoverCardContent className={`w-80 ${isBroken ? 'border-destructive/50 bg-destructive/5' : ''}`}>
        {isLoading ? (
          // T025: Loading skeleton matching preview card layout
          <div className="space-y-3">
            {/* Title skeleton (h4 height, slightly larger) */}
            <div className="h-5 bg-muted animate-pulse rounded w-3/4" />

            {/* Snippet skeleton (3 text lines) */}
            <div className="space-y-2">
              <div className="h-3.5 bg-muted animate-pulse rounded" />
              <div className="h-3.5 bg-muted animate-pulse rounded w-5/6" />
              <div className="h-3.5 bg-muted animate-pulse rounded w-4/6" />
            </div>

            {/* Badge placeholders (2 badge-sized skeletons) */}
            <div className="flex gap-1.5">
              <div className="h-5 w-16 bg-muted animate-pulse rounded-full" />
              <div className="h-5 w-20 bg-muted animate-pulse rounded-full" />
            </div>

            {/* Footer skeleton (with border-top separator) */}
            <div className="pt-2 border-t">
              <div className="h-3 bg-muted animate-pulse rounded w-32" />
            </div>
          </div>
        ) : isBroken ? (
          // T033: Broken link card with red-tinted styling
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5 text-destructive"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <h4 className="font-semibold text-base text-destructive">
                Note not found
              </h4>
            </div>

            <p className="text-sm text-muted-foreground">
              The note "{linkText}" does not exist in your vault.
            </p>

            <div className="text-xs text-muted-foreground pt-2 border-t border-destructive/20">
              Click the wikilink to create this note
            </div>
          </div>
        ) : preview ? (
          // T031: Rich preview card
          <div className="space-y-3">
            {/* Title */}
            <h4 className="font-semibold text-base leading-tight">
              {preview.title}
            </h4>

            {/* Snippet text (max 3 lines) */}
            <p className="text-sm text-muted-foreground line-clamp-3 leading-relaxed">
              {preview.snippet}
            </p>

            {/* Tags (max 3) */}
            {preview.tags && preview.tags.length > 0 && (
              <div className="flex gap-1.5 flex-wrap">
                {preview.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}

            {/* Updated timestamp footer */}
            <div className="text-xs text-muted-foreground pt-2 border-t">
              Updated {formatDistanceToNow(preview.updated)}
            </div>
          </div>
        ) : null}
      </HoverCardContent>
    </HoverCard>
  );
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
          <WikilinkPreview
            linkText={linkText}
            onClick={(e?: React.MouseEvent) => {
              e?.preventDefault();
              onWikilinkClick?.(linkText);
            }}
          >
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
          </WikilinkPreview>
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

    // T039: Style headings with ID generation for TOC
    h1: ({ children, ...props }) => {
      const text = typeof children === 'string' ? children : '';
      const id = text ? slugify(text) : undefined;
      return (
        <h1 id={id} className="text-3xl font-bold mt-6 mb-4" {...props}>
          {children}
        </h1>
      );
    },
    h2: ({ children, ...props }) => {
      const text = typeof children === 'string' ? children : '';
      const id = text ? slugify(text) : undefined;
      return (
        <h2 id={id} className="text-2xl font-semibold mt-5 mb-3" {...props}>
          {children}
        </h2>
      );
    },
    h3: ({ children, ...props }) => {
      const text = typeof children === 'string' ? children : '';
      const id = text ? slugify(text) : undefined;
      return (
        <h3 id={id} className="text-xl font-semibold mt-4 mb-2" {...props}>
          {children}
        </h3>
      );
    },

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

