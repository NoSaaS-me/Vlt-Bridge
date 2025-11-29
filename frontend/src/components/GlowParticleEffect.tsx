/**
 * GlowParticleEffect Component
 *
 * A wrapper component that adds soft glowing particle effects to its children.
 * Particles spawn at click locations and float upward while fading.
 *
 * VISUAL ARCHITECTURE:
 * ┌──────────────────────────────────────────────────────────┐
 * │  Wrapper Container (position: relative)                  │
 * │  ┌────────────────────────────────────────────────────┐  │
 * │  │ Children (note content, wikilinks, etc.)           │  │
 * │  │                                                    │  │
 * │  │  Some text with [[wikilink]] that you can click    │  │
 * │  │                      ↑                             │  │
 * │  │                  Click here                        │  │
 * │  └────────────────────────────────────────────────────┘  │
 * │  ┌────────────────────────────────────────────────────┐  │
 * │  │ Canvas Overlay (position: absolute, pointer-events:│  │
 * │  │ none)                                              │  │
 * │  │                 ░░▒▓██▓▒░░                         │  │
 * │  │              ░░▒▓██▓▒░░                            │  │
 * │  │           ░░▒▓██▓▒░░   <- Particles float up       │  │
 * │  │                                                    │  │
 * │  └────────────────────────────────────────────────────┘  │
 * └──────────────────────────────────────────────────────────┘
 *
 * VISUAL FLOW:
 * 1. User clicks on wikilink (or any child element)
 * 2. Click event bubbles up to wrapper
 * 3. Particles spawn at click coordinates
 * 4. Particles animate independently on canvas
 * 5. Canvas has pointer-events: none, so clicks pass through
 *
 * ACCESSIBILITY:
 * - Respects prefers-reduced-motion
 * - Canvas is purely decorative (no interaction)
 * - Does not interfere with screen readers
 */

import React, { useRef, useCallback, useEffect, useState } from 'react';
import { useGlowParticles } from '@/hooks/useGlowParticles';
import { useReducedMotion } from '@/hooks/useReducedMotion';
import type { ParticleConfig, ParticlePreset } from '@/types/particles';
import { DEFAULT_PARTICLE_CONFIG, PARTICLE_PRESETS } from '@/types/particles';

export interface GlowParticleEffectProps {
  /**
   * Content to wrap with particle effect
   */
  children: React.ReactNode;

  /**
   * Particle configuration (colors, sizes, speeds, etc.)
   * Can be a preset name or custom config object
   */
  config?: Partial<ParticleConfig> | ParticlePreset;

  /**
   * CSS class for the wrapper container
   */
  className?: string;

  /**
   * Whether particles are enabled
   * Automatically disabled when prefers-reduced-motion is set
   */
  enabled?: boolean;

  /**
   * Optional callback when particles are spawned
   * Useful for analytics or sound effects
   */
  onParticleBurst?: (x: number, y: number) => void;

  /**
   * Selector for elements that should trigger particles
   * If not provided, ANY click in the container triggers particles
   * Example: '.wikilink' to only trigger on wikilink clicks
   */
  triggerSelector?: string;
}

/**
 * Resolve config from preset name or partial config
 */
function resolveConfig(
  configProp?: Partial<ParticleConfig> | ParticlePreset
): ParticleConfig {
  if (!configProp) {
    return DEFAULT_PARTICLE_CONFIG;
  }

  // If it's a string, look up the preset
  if (typeof configProp === 'string') {
    return PARTICLE_PRESETS[configProp] || DEFAULT_PARTICLE_CONFIG;
  }

  // Merge with defaults
  return { ...DEFAULT_PARTICLE_CONFIG, ...configProp };
}

/**
 * GlowParticleEffect Component
 *
 * Wraps children with an invisible canvas overlay that renders particles.
 *
 * USAGE:
 * ```tsx
 * <GlowParticleEffect config="elegant" triggerSelector=".wikilink">
 *   <div>Content with [[wikilinks]]</div>
 * </GlowParticleEffect>
 * ```
 */
export function GlowParticleEffect({
  children,
  config: configProp,
  className = '',
  enabled = true,
  onParticleBurst,
  triggerSelector,
}: GlowParticleEffectProps) {
  // Check accessibility preference
  const prefersReducedMotion = useReducedMotion();

  // Resolve final configuration
  const config = resolveConfig(configProp);

  // Determine if particles should be active
  const isDisabled = prefersReducedMotion || !enabled;

  // Reference to wrapper for coordinate calculation
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Canvas dimensions state
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Initialize particle system
  const { canvasRef, createParticleBurst } = useGlowParticles(config, isDisabled);

  /**
   * UPDATE CANVAS DIMENSIONS
   *
   * VISUAL PURPOSE: Canvas must match container size for correct positioning.
   * If canvas is smaller than container, particles would be clipped.
   * If larger, particles could render in invisible area.
   *
   * We use ResizeObserver to handle:
   * - Initial mount
   * - Window resize
   * - Layout changes
   */
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const updateDimensions = () => {
      const rect = wrapper.getBoundingClientRect();
      setDimensions({
        width: rect.width,
        height: rect.height,
      });
    };

    // Initial measurement
    updateDimensions();

    // Watch for size changes
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(wrapper);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  /**
   * HANDLE CLICK FOR PARTICLE SPAWN
   *
   * VISUAL FLOW:
   * 1. User clicks somewhere in the wrapper
   * 2. We check if click target matches triggerSelector (if provided)
   * 3. We calculate click position relative to canvas
   * 4. We spawn particles at that position
   *
   * The particles appear exactly where the user clicked,
   * creating a direct visual response to the interaction.
   */
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      // Skip if disabled
      if (isDisabled) return;

      // Check if trigger selector is specified
      if (triggerSelector) {
        // Find if click target matches selector (or is inside matching element)
        const target = event.target as HTMLElement;
        const matchingElement = target.closest(triggerSelector);
        if (!matchingElement) {
          // Click was not on a triggering element
          return;
        }
      }

      const wrapper = wrapperRef.current;
      if (!wrapper) return;

      /**
       * CALCULATE RELATIVE COORDINATES
       *
       * VISUAL: event.clientX/Y are viewport-relative
       * We need coordinates relative to the canvas origin (top-left of wrapper)
       *
       * rect.left/top give us the wrapper's position in the viewport
       * Subtracting gives us the click position within the wrapper
       */
      const rect = wrapper.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Spawn particles at click location
      createParticleBurst(x, y);

      // Notify callback if provided
      onParticleBurst?.(x, y);
    },
    [isDisabled, triggerSelector, createParticleBurst, onParticleBurst]
  );

  return (
    <div
      ref={wrapperRef}
      className={`relative ${className}`}
      onClick={handleClick}
    >
      {/* Children (the actual content) */}
      {children}

      {/*
        CANVAS OVERLAY

        VISUAL ARCHITECTURE:
        - Positioned absolutely to cover the entire wrapper
        - pointer-events: none allows clicks to pass through to children
        - z-index ensures particles render above content

        VISUAL: The canvas is invisible until particles are drawn on it.
        It acts as a transparent layer floating above the content.
      */}
      {!isDisabled && (
        <canvas
          ref={canvasRef}
          width={dimensions.width}
          height={dimensions.height}
          className="absolute inset-0 pointer-events-none z-10"
          style={{
            // Ensure canvas covers entire container
            width: '100%',
            height: '100%',
          }}
          // Accessibility: Mark as decorative/presentation
          role="presentation"
          aria-hidden="true"
        />
      )}
    </div>
  );
}

/**
 * Re-export presets for convenience
 */
export { PARTICLE_PRESETS } from '@/types/particles';
export type { ParticleConfig, ParticlePreset } from '@/types/particles';

export default GlowParticleEffect;
