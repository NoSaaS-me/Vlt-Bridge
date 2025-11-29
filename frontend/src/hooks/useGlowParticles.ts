/**
 * useGlowParticles Hook
 *
 * Core particle system hook that manages creation, animation, and rendering
 * of soft glowing particles on a canvas element.
 *
 * VISUAL ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────┐
 * │  Canvas (positioned absolutely over content)        │
 * │                                                     │
 * │     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
 * │     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
 * │     ░░░░░░░░░░░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │  <- Particles float
 * │     ░░░░░░░░░░▓███▓░░░░░░░░░░▒▒▒░░░░░░░░░░░░░░░    │     upward and fade
 * │     ░░░░░░░░░░░▓▓▓░░░░░░░░░░▒███▒░░░░░░░░░░░░░░    │
 * │     ░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒░░░░░░░░░░░░░░░    │
 * │     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
 * │                      ↑                             │
 * │                  Click here                        │
 * │               [Wikilink Text]                      │
 * └─────────────────────────────────────────────────────┘
 *
 * Each particle (▓██▓ above) consists of:
 * - Bright center (█) - solid color
 * - Gradient falloff (▓) - fading to transparent
 * - Soft glow halo (▒) - shadowBlur effect
 */

import { useRef, useCallback, useEffect } from 'react';
import type { Particle, ParticleConfig } from '@/types/particles';
import { DEFAULT_PARTICLE_CONFIG } from '@/types/particles';

/**
 * Parse hex color to RGB components
 *
 * VISUAL PURPOSE: We need RGB values to create gradient stops with
 * varying alpha (transparency) values. Hex colors don't support alpha
 * in the same way, so we convert to rgba() format.
 *
 * Example: '#6366f1' -> { r: 99, g: 102, b: 241 }
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : { r: 99, g: 102, b: 241 }; // Fallback to indigo
}

/**
 * Generate a random number within a range
 */
function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Unique ID generator for particles
 */
let particleIdCounter = 0;

export interface UseGlowParticlesReturn {
  /**
   * Ref to attach to the canvas element
   * The canvas should be positioned absolutely over the content area
   */
  canvasRef: React.RefObject<HTMLCanvasElement | null>;

  /**
   * Trigger a particle burst at specific coordinates
   * @param x - X coordinate relative to canvas
   * @param y - Y coordinate relative to canvas
   *
   * VISUAL: Creates particleCount particles at (x,y) that burst outward
   * in random directions
   */
  createParticleBurst: (x: number, y: number) => void;

  /**
   * Check if animation loop is currently running
   */
  isAnimating: boolean;
}

/**
 * Main particle system hook
 *
 * @param config - Particle visual configuration (colors, sizes, speeds, etc.)
 * @param disabled - If true, particles won't be created (accessibility)
 * @returns Canvas ref and burst trigger function
 *
 * VISUAL LIFECYCLE OVERVIEW:
 * 1. User clicks -> createParticleBurst(x, y) called
 * 2. Particles spawn at click point with random velocities
 * 3. Animation loop runs at 60 FPS:
 *    a. Clear previous frame
 *    b. Update each particle (position, life, size)
 *    c. Draw each particle (gradient + glow)
 *    d. Remove dead particles (life <= 0)
 * 4. Loop stops when all particles are gone (performance optimization)
 */
export function useGlowParticles(
  config: ParticleConfig = DEFAULT_PARTICLE_CONFIG,
  disabled: boolean = false
): UseGlowParticlesReturn {
  // Mutable refs that persist across renders without causing re-renders
  const particlesRef = useRef<Particle[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isAnimatingRef = useRef<boolean>(false);

  /**
   * CREATE PARTICLE BURST
   *
   * VISUAL EFFECT: When called, spawns N particles at the click location.
   * Each particle:
   * - Starts at (x, y)
   * - Gets a random velocity in a random direction
   * - Gets a random size and color from config
   * - Has full life (1.0) at spawn
   *
   * The "burst" pattern comes from random angles - particles explode
   * outward in all directions like a tiny firework.
   *
   * Visual analogy: Like tapping a dandelion and seeds flying off
   */
  const createParticleBurst = useCallback(
    (x: number, y: number) => {
      // Skip if disabled (accessibility) or no canvas
      if (disabled) return;

      const newParticles: Particle[] = [];

      for (let i = 0; i < config.particleCount; i++) {
        /**
         * RANDOM ANGLE for burst direction
         *
         * VISUAL: Math.random() * Math.PI * 2 gives angle from 0 to 2π
         * This means particles fly in ALL directions (360 degrees)
         *
         * Alternative patterns:
         * - Math.PI * 1.5 to Math.PI * 2.5 = upward-biased cone
         * - Fixed angle = particles move in same direction
         */
        const angle = Math.random() * Math.PI * 2;

        /**
         * RANDOM SPEED within configured range
         *
         * VISUAL: Variation in speed creates depth - some particles
         * appear to move faster/closer while others drift slowly
         */
        const speed = randomBetween(config.minSpeed, config.maxSpeed);

        /**
         * VELOCITY COMPONENTS from angle and speed
         *
         * VISUAL:
         * - vx = speed * cos(angle): horizontal component
         * - vy = speed * sin(angle): vertical component
         *
         * Together, these create the direction of movement
         */
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;

        /**
         * RANDOM SIZE for visual variety
         *
         * VISUAL: Mix of sizes creates depth and interest.
         * Smaller particles appear more distant/delicate.
         */
        const size = randomBetween(config.minSize, config.maxSize);

        /**
         * RANDOM COLOR from palette
         *
         * VISUAL: Each particle gets one of the configured colors.
         * The gradient will fade from this color (center) to transparent (edge).
         */
        const color = config.colors[Math.floor(Math.random() * config.colors.length)];

        newParticles.push({
          x,
          y,
          vx,
          vy,
          size,
          life: 1.0, // Full life at spawn (completely visible)
          color,
          id: particleIdCounter++,
        });
      }

      // Add new particles to existing ones
      particlesRef.current = [...particlesRef.current, ...newParticles];

      // Start animation if not already running
      if (!isAnimatingRef.current) {
        isAnimatingRef.current = true;
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    },
    [config, disabled]
  );

  /**
   * DRAW SINGLE PARTICLE
   *
   * This is where the visual magic happens. Each particle is rendered as:
   * 1. A radial gradient (bright center -> transparent edge)
   * 2. With a soft glow effect (shadowBlur)
   *
   * VISUAL BREAKDOWN:
   *
   *        ░░░░░░░░░░░░░░░  <- Glow halo (shadowBlur)
   *       ░░░░░░░░░░░░░░░░░
   *      ░░░░░▒▒▒▒▒▒▒░░░░░░  <- Outer gradient (50% opacity, fading)
   *     ░░░░▒▒▒▒▒▒▒▒▒▒░░░░░
   *    ░░░░▒▒▒▓▓▓▓▓▒▒▒░░░░░  <- Mid gradient (70% opacity)
   *    ░░░░▒▒▓▓███▓▓▒▒░░░░░  <- Inner core (100% opacity)
   *    ░░░░▒▒▒▓▓▓▓▓▒▒▒░░░░░
   *     ░░░░▒▒▒▒▒▒▒▒▒░░░░░░
   *      ░░░░░▒▒▒▒▒▒▒░░░░░░
   *       ░░░░░░░░░░░░░░░░░
   *        ░░░░░░░░░░░░░░░
   */
  const drawParticle = useCallback(
    (ctx: CanvasRenderingContext2D, particle: Particle) => {
      // Calculate current visual properties based on life
      // As life decreases, particle becomes smaller and more transparent
      const currentSize = particle.size * particle.life;
      const currentOpacity = particle.life;

      // Skip if too small to see
      if (currentSize < 0.5) return;

      // Parse the base color to RGB for gradient creation
      const rgb = hexToRgb(particle.color);

      /**
       * CREATE RADIAL GRADIENT
       *
       * ctx.createRadialGradient(x0, y0, r0, x1, y1, r1)
       *
       * PARAMETERS EXPLAINED:
       * - (x0, y0, r0): Inner circle - center of particle, radius 0
       * - (x1, y1, r1): Outer circle - center of particle, radius = currentSize/2
       *
       * VISUAL: The gradient will interpolate colors from the inner circle
       * to the outer circle. Since both circles share the same center,
       * this creates a circular gradient radiating outward.
       *
       *   Inner (r0=0)    Outer (r1=radius)
       *        █          ░░░░░░░░░░
       *        ↓          ↓
       *      [solid] -> [transparent]
       */
      const radius = currentSize / 2;
      const gradient = ctx.createRadialGradient(
        particle.x,
        particle.y,
        0, // Inner circle: center point, radius 0 (a dot)
        particle.x,
        particle.y,
        radius // Outer circle: same center, radius = particle size
      );

      /**
       * ADD COLOR STOPS
       *
       * Color stops define how the gradient transitions from center to edge.
       *
       * VISUAL EFFECT OF EACH STOP:
       *
       * Stop 0.0 (center): Full color, full opacity
       *   - This is the "hot spot" - brightest point
       *   - Like the white center of a candle flame
       *
       * Stop 0.4 (40% radius): Full color, 70% opacity
       *   - Still bright but starting to fade
       *   - Creates a solid-looking core
       *
       * Stop 0.7 (70% radius): Full color, 30% opacity
       *   - Clearly fading now
       *   - The "warm glow" zone
       *
       * Stop 1.0 (edge): Full color, 0% opacity
       *   - Completely transparent
       *   - No hard edge - particle "melts" into background
       *
       * The currentOpacity multiplier makes the whole particle
       * more transparent as it ages (life decreases from 1 to 0)
       */
      gradient.addColorStop(
        0,
        `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${1.0 * currentOpacity})`
      );
      gradient.addColorStop(
        0.4,
        `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.7 * currentOpacity})`
      );
      gradient.addColorStop(
        0.7,
        `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.3 * currentOpacity})`
      );
      gradient.addColorStop(
        1,
        `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`
      );

      /**
       * APPLY GLOW EFFECT (shadowBlur)
       *
       * VISUAL: shadowBlur creates a soft halo around whatever we draw.
       * Unlike the radial gradient (which IS the particle shape),
       * the shadow extends BEYOND the particle boundary.
       *
       * Think of it like:
       * - Radial gradient = the physical LED light bulb
       * - shadowBlur = the glow you see around the bulb
       *
       * shadowBlur = 15 means:
       * - A 15-pixel soft blur extends beyond the particle
       * - Color matches shadowColor (the particle's color)
       * - Intensity determined by currentOpacity
       *
       * As particle life decreases:
       * - Glow intensity decreases proportionally
       * - Creates the effect of "cooling down" or "fading away"
       */
      ctx.shadowBlur = config.glowIntensity * currentOpacity;
      ctx.shadowColor = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.6 * currentOpacity})`;

      /**
       * DRAW THE PARTICLE
       *
       * ctx.arc(x, y, radius, startAngle, endAngle)
       * - Draws a circle (full arc = 0 to 2π)
       * - Fill with our radial gradient
       *
       * VISUAL: The combination of radial gradient + shadowBlur creates:
       * - Bright core (gradient center)
       * - Soft falloff (gradient edge)
       * - Extended glow (shadow blur)
       *
       * Together: a soft, glowing orb that looks like a tiny light source
       */
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Reset shadow to avoid affecting other drawings
      ctx.shadowBlur = 0;
      ctx.shadowColor = 'transparent';
    },
    [config.glowIntensity]
  );

  /**
   * ANIMATION LOOP
   *
   * This runs at approximately 60 frames per second (synced with display).
   * Each frame:
   * 1. Clear the previous frame (canvas becomes transparent)
   * 2. Update each particle's physics (position, life, size)
   * 3. Draw each particle in its new state
   * 4. Remove particles that have "died" (life <= 0)
   * 5. Continue loop if particles remain, else stop
   *
   * VISUAL FLOW:
   * Frame 0: Particles at spawn position, full brightness
   * Frame 30: Particles have moved, 50% life remaining
   * Frame 60: Particles nearly gone, very faint
   * Frame 67: All particles dead, loop stops
   *
   * Performance note: Loop only runs when particles exist.
   * When no particles remain, we stop to save CPU.
   */
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      isAnimatingRef.current = false;
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      isAnimatingRef.current = false;
      return;
    }

    /**
     * CLEAR PREVIOUS FRAME
     *
     * VISUAL: Wipes the canvas completely transparent.
     * Without this, particles would leave "trails" as they move.
     *
     * Alternative: Use ctx.fillRect with semi-transparent color
     * for intentional trail/afterimage effects (not desired here).
     */
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    /**
     * UPDATE AND DRAW EACH PARTICLE
     */
    particlesRef.current = particlesRef.current.filter((particle) => {
      /**
       * UPDATE POSITION
       *
       * VISUAL: Move particle by its velocity
       * - x += vx: horizontal movement
       * - y += vy: vertical movement
       *
       * Result: Particle drifts in its assigned direction
       */
      particle.x += particle.vx;
      particle.y += particle.vy;

      /**
       * APPLY UPWARD DRIFT
       *
       * VISUAL: Adds small upward acceleration each frame
       * Negative vy = moving up (canvas y increases downward)
       *
       * Effect: Particles gradually curve upward like:
       * - Sparks rising from a fire
       * - Bubbles floating in liquid
       * - Heat shimmer rising
       */
      particle.vy += config.upwardDrift;

      /**
       * UPDATE LIFE (opacity)
       *
       * VISUAL: Decrease life each frame
       * life goes from 1.0 -> 0.0 over fadeRate * frames
       *
       * At 60 FPS with fadeRate 0.016:
       * 1.0 / 0.016 = 62.5 frames = ~1 second to fade out
       */
      particle.life -= config.fadeRate;

      /**
       * UPDATE SIZE (shrinking)
       *
       * VISUAL: Multiply size by shrinkRate each frame
       * shrinkRate 0.985 means particle is 98.5% of previous size
       *
       * After 60 frames: 0.985^60 = 0.40 = 40% of original size
       *
       * Combined with fade, creates "dissipating" effect
       */
      particle.size *= config.shrinkRate;

      /**
       * DRAW THE PARTICLE
       *
       * Only if still alive (visible)
       */
      if (particle.life > 0) {
        drawParticle(ctx, particle);
      }

      /**
       * KEEP OR REMOVE
       *
       * Return true to keep particle in array, false to remove
       * Remove when life <= 0 (completely faded out)
       */
      return particle.life > 0;
    });

    /**
     * CONTINUE OR STOP LOOP
     *
     * VISUAL: If particles remain, schedule next frame
     * If no particles, stop the animation loop
     *
     * Performance: Prevents unnecessary CPU usage when idle
     */
    if (particlesRef.current.length > 0) {
      animationFrameRef.current = requestAnimationFrame(animate);
    } else {
      isAnimatingRef.current = false;
      animationFrameRef.current = null;
    }
  }, [config.fadeRate, config.shrinkRate, config.upwardDrift, drawParticle]);

  /**
   * CLEANUP ON UNMOUNT
   *
   * Cancel any pending animation frame to prevent memory leaks
   * and errors from drawing on unmounted canvas
   */
  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return {
    canvasRef,
    createParticleBurst,
    isAnimating: isAnimatingRef.current,
  };
}

export default useGlowParticles;
