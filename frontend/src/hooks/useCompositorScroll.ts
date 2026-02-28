/**
 * useCompositorScroll — Manages horizontal scroll state for the terminal compositor.
 *
 * Features:
 * - IntersectionObserver to track which panes are visible
 * - Smooth lerp animation for programmatic scrolling (button clicks, sidebar picks)
 * - CSS scroll-snap handles user swipe/scroll natively
 * - Integrates with useReducedMotion for accessibility
 */
import { useRef, useState, useEffect, useCallback } from 'react';
import { useReducedMotion } from '@/hooks/useReducedMotion';

interface UseCompositorScrollOptions {
  paneCount: number;
}

interface UseCompositorScrollReturn {
  viewportRef: React.RefObject<HTMLDivElement | null>;
  scrollToIndex: (index: number) => void;
  scrollPrev: () => void;
  scrollNext: () => void;
  focusedIndex: number;
  setFocusedIndex: (i: number) => void;
  visibleIndices: Set<number>;
}

/** easeOutCubic — fast start, smooth deceleration */
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function useCompositorScroll({
  paneCount,
}: UseCompositorScrollOptions): UseCompositorScrollReturn {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [visibleIndices, setVisibleIndices] = useState<Set<number>>(new Set([0]));
  const animationRef = useRef<number | null>(null);
  const prefersReducedMotion = useReducedMotion();

  // ── Lerp scroll animation ────────────────────────────────────────────
  const lerpScrollTo = useCallback((target: number, duration = 400) => {
    const el = viewportRef.current;
    if (!el) return;

    // Cancel any in-progress animation
    if (animationRef.current != null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (prefersReducedMotion) {
      el.scrollLeft = target;
      return;
    }

    const start = el.scrollLeft;
    const delta = target - start;
    if (Math.abs(delta) < 1) return;

    const t0 = performance.now();

    function tick(now: number) {
      const p = Math.min((now - t0) / duration, 1);
      el!.scrollLeft = start + delta * easeOutCubic(p);
      if (p < 1) {
        animationRef.current = requestAnimationFrame(tick);
      } else {
        animationRef.current = null;
      }
    }

    animationRef.current = requestAnimationFrame(tick);
  }, [prefersReducedMotion]);

  // ── Scroll to index ──────────────────────────────────────────────────
  const scrollToIndex = useCallback((index: number) => {
    const el = viewportRef.current;
    if (!el) return;
    const clamped = Math.max(0, Math.min(index, paneCount - 1));
    setFocusedIndex(clamped);

    // Find the child element at this index
    const child = el.children[clamped] as HTMLElement | undefined;
    if (!child) return;

    // Calculate scroll target relative to the scroll container, not offsetParent.
    // offsetLeft is unreliable here because the scroll container may not be the offsetParent.
    const target = child.getBoundingClientRect().left - el.getBoundingClientRect().left + el.scrollLeft;
    lerpScrollTo(target);
  }, [paneCount, lerpScrollTo]);

  const scrollPrev = useCallback(() => {
    scrollToIndex(Math.max(0, focusedIndex - 1));
  }, [focusedIndex, scrollToIndex]);

  const scrollNext = useCallback(() => {
    scrollToIndex(Math.min(paneCount - 1, focusedIndex + 1));
  }, [focusedIndex, paneCount, scrollToIndex]);

  // ── IntersectionObserver for visibility tracking ──────────────────────
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        setVisibleIndices((prev) => {
          const next = new Set(prev);
          for (const entry of entries) {
            const idx = Array.from(el.children).indexOf(entry.target as HTMLElement);
            if (idx === -1) continue;
            if (entry.isIntersecting) {
              next.add(idx);
            } else {
              next.delete(idx);
            }
          }
          // Avoid unnecessary re-renders if nothing changed
          if (next.size === prev.size && [...next].every(i => prev.has(i))) return prev;
          return next;
        });
      },
      { root: el, threshold: 0.3 },
    );

    // Observe all pane wrappers
    for (const child of Array.from(el.children)) {
      observer.observe(child);
    }

    return () => observer.disconnect();
  }, [paneCount]);

  // ── Keyboard navigation ──────────────────────────────────────────────
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Only handle if no input element is focused
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      if (e.key === 'ArrowLeft' && e.altKey) {
        e.preventDefault();
        scrollToIndex(Math.max(0, focusedIndex - 1));
      } else if (e.key === 'ArrowRight' && e.altKey) {
        e.preventDefault();
        scrollToIndex(Math.min(paneCount - 1, focusedIndex + 1));
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [focusedIndex, paneCount, scrollToIndex]);

  // ── Cleanup animation on unmount ─────────────────────────────────────
  useEffect(() => {
    return () => {
      if (animationRef.current != null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return {
    viewportRef,
    scrollToIndex,
    scrollPrev,
    scrollNext,
    focusedIndex,
    setFocusedIndex,
    visibleIndices,
  };
}
