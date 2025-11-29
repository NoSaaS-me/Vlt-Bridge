/**
 * useReducedMotion Hook
 *
 * Accessibility hook that detects user's motion preference.
 * When enabled, particle effects should be disabled or minimized.
 *
 * VISUAL ACCESSIBILITY CONTEXT:
 * Some users experience motion sickness, vestibular disorders, or simply
 * prefer reduced animation. The "prefers-reduced-motion" media query
 * indicates this preference.
 *
 * When this hook returns true:
 * - Particle effects should NOT be rendered
 * - OR particles should appear statically (no animation)
 * - This respects user autonomy and WCAG guidelines
 */
import { useState, useEffect } from 'react';

/**
 * Detects if the user prefers reduced motion
 *
 * @returns true if user prefers reduced motion, false otherwise
 *
 * USAGE:
 * ```tsx
 * const prefersReducedMotion = useReducedMotion();
 *
 * if (prefersReducedMotion) {
 *   // Skip particle animation entirely
 *   return null;
 * }
 * ```
 */
export function useReducedMotion(): boolean {
  // Server-side rendering safety: default to false
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    // Check if matchMedia is available (browser environment)
    if (typeof window === 'undefined' || !window.matchMedia) {
      return;
    }

    // Create media query matcher
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    // Set initial value
    setPrefersReducedMotion(mediaQuery.matches);

    // Listen for changes (user might toggle system preference)
    const handleChange = (event: MediaQueryListEvent) => {
      setPrefersReducedMotion(event.matches);
    };

    // Modern browsers use addEventListener
    mediaQuery.addEventListener('change', handleChange);

    // Cleanup listener on unmount
    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  return prefersReducedMotion;
}

export default useReducedMotion;
