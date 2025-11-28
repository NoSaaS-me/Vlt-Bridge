/**
 * T037-T047: Table of Contents hook
 * Extracts headings from rendered markdown and provides scroll navigation
 */
import { useEffect, useState, useCallback } from 'react';

export interface Heading {
  id: string;
  text: string;
  level: number;
}

interface UseTableOfContentsReturn {
  headings: Heading[];
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  scrollToHeading: (id: string) => void;
}

/**
 * T040: Slugify text to create valid HTML IDs
 * Handles duplicates by appending -2, -3, etc.
 */
export function slugify(text: string, existingSlugs: Set<string> = new Set()): string {
  // T040: Basic slugification
  const baseSlug = text
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w-]/g, '');

  // T050: Handle duplicates
  if (!existingSlugs.has(baseSlug)) {
    return baseSlug;
  }

  let counter = 2;
  let uniqueSlug = `${baseSlug}-${counter}`;
  while (existingSlugs.has(uniqueSlug)) {
    counter++;
    uniqueSlug = `${baseSlug}-${counter}`;
  }

  return uniqueSlug;
}

/**
 * T037-T047: Hook for managing TOC state and heading extraction
 */
export function useTableOfContents(): UseTableOfContentsReturn {
  // T042: TOC panel open state
  const [isOpen, setIsOpenState] = useState<boolean>(() => {
    // T043: Restore from localStorage
    const saved = localStorage.getItem('toc-panel-open');
    return saved ? JSON.parse(saved) : false;
  });

  // T041: Store extracted headings
  const [headings, setHeadings] = useState<Heading[]>([]);

  // T043: Persist panel state to localStorage
  const setIsOpen = useCallback((open: boolean) => {
    setIsOpenState(open);
    localStorage.setItem('toc-panel-open', JSON.stringify(open));
  }, []);

  // T041: Extract headings from DOM (called after render)
  const extractHeadings = useCallback(() => {
    const headingElements = document.querySelectorAll('h1, h2, h3');
    const extracted: Heading[] = [];

    headingElements.forEach((element) => {
      const id = element.id;
      const text = element.textContent || '';
      const level = parseInt(element.tagName.charAt(1));

      if (id && text) {
        extracted.push({ id, text, level });
      }
    });

    setHeadings(extracted);
  }, []);

  // T046-T047: Scroll to heading with smooth behavior and accessibility
  const scrollToHeading = useCallback((id: string) => {
    const element = document.getElementById(id);
    if (element) {
      // T047: Respect prefers-reduced-motion
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      element.scrollIntoView({
        behavior: prefersReducedMotion ? 'auto' : 'smooth',
        block: 'start'
      });
    }
  }, []);

  // Re-extract headings when content changes
  useEffect(() => {
    // Use MutationObserver to detect when markdown is rendered
    const observer = new MutationObserver(() => {
      extractHeadings();
    });

    // Observe the markdown content container
    const contentContainer = document.querySelector('.prose');
    if (contentContainer) {
      observer.observe(contentContainer, {
        childList: true,
        subtree: true,
      });

      // Initial extraction
      extractHeadings();
    }

    return () => {
      observer.disconnect();
    };
  }, [extractHeadings]);

  return {
    headings,
    isOpen,
    setIsOpen,
    scrollToHeading,
  };
}
