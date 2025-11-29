/**
 * Particle System Types
 *
 * VISUAL OVERVIEW:
 * These types define the data structures for creating soft, glowing particles
 * that appear when users interact with elements (like wikilinks).
 *
 * Each particle is a small glowing orb (6-12px) with:
 * - A radial gradient from bright center to transparent edge
 * - A soft blur halo that extends beyond the particle boundary
 * - Smooth animation: floating outward, shrinking, and fading
 *
 * Visual Analogy: Think of particles as tiny floating embers or LED lights
 * viewed through frosted glass - soft, warm, and gradually fading.
 */

/**
 * Individual Particle State
 *
 * Each particle is an autonomous visual element with its own position,
 * velocity, appearance, and lifecycle state.
 */
export interface Particle {
  /**
   * Horizontal position in canvas coordinates (pixels from left edge)
   * VISUAL: Determines where the glowing orb appears on screen
   */
  x: number;

  /**
   * Vertical position in canvas coordinates (pixels from top edge)
   * VISUAL: Combined with x, places the particle's center point
   */
  y: number;

  /**
   * Horizontal velocity (pixels per animation frame)
   * VISUAL: Positive = moving right, Negative = moving left
   * Typical range: -3 to +3 for gentle floating movement
   */
  vx: number;

  /**
   * Vertical velocity (pixels per animation frame)
   * VISUAL: Positive = moving down, Negative = moving up
   * Typically starts negative with small positive bias for upward drift
   */
  vy: number;

  /**
   * Particle diameter in pixels
   * VISUAL: Determines the size of the glowing orb
   * The radial gradient spans from center (0) to this radius (size/2)
   * Typical range: 4-14 pixels for subtle effect
   */
  size: number;

  /**
   * Lifecycle progress from 1.0 (birth) to 0.0 (death)
   * VISUAL EFFECTS:
   * - Multiplied with opacity: particle becomes more transparent as life decreases
   * - Multiplied with glow intensity: halo shrinks as particle dies
   * - When life <= 0, particle is removed from the system
   *
   * Visual analogy: Like a candle burning down - starts bright, gradually dims
   */
  life: number;

  /**
   * Base color for the particle gradient (hex format: '#RRGGBB')
   * VISUAL: This is the CENTER color of the radial gradient
   * The edge fades to transparent using the same hue
   *
   * Palette typically includes:
   * - '#6366f1' (indigo/blue) - cool, professional
   * - '#a855f7' (purple) - rich, balanced
   * - '#ec4899' (pink) - warm accent
   */
  color: string;

  /**
   * Unique identifier for React reconciliation
   * Not visually relevant, but necessary for efficient DOM updates
   */
  id: number;
}

/**
 * Particle System Configuration
 *
 * These settings control the overall visual behavior of the particle system.
 * Adjusting these values changes the "feel" of the effect.
 */
export interface ParticleConfig {
  /**
   * Number of particles spawned per trigger (click/interaction)
   * VISUAL IMPACT:
   * - Low (5-10): Subtle, minimal effect - professional/elegant
   * - Medium (12-20): Noticeable but not overwhelming
   * - High (25+): Dense, celebratory - can feel tacky if overdone
   *
   * Recommendation: 12-15 for subtle elegance
   */
  particleCount: number;

  /**
   * Minimum particle diameter in pixels
   * VISUAL: Sets the lower bound for particle size variation
   * Smaller particles appear more distant/delicate
   * Typical value: 4-6 pixels
   */
  minSize: number;

  /**
   * Maximum particle diameter in pixels
   * VISUAL: Sets the upper bound for particle size variation
   * Larger particles appear closer/more prominent
   * Typical value: 8-12 pixels
   */
  maxSize: number;

  /**
   * Minimum initial velocity magnitude (pixels per frame)
   * VISUAL: How slow the slowest particles move
   * Lower = more leisurely floating
   * Typical value: 1.5-2.0
   */
  minSpeed: number;

  /**
   * Maximum initial velocity magnitude (pixels per frame)
   * VISUAL: How fast the fastest particles move
   * Higher = more energetic burst
   * Typical value: 3.0-4.5
   */
  maxSpeed: number;

  /**
   * Glow halo intensity (shadowBlur value in pixels)
   * VISUAL IMPACT:
   * - 5-10: Subtle glow, barely perceptible halo
   * - 12-18: Moderate glow, clearly visible soft halo
   * - 20+: Strong glow, dramatic ethereal effect
   *
   * This creates the "frosted glass" effect around particles.
   * The glow color matches the particle color.
   *
   * Visual analogy: Like the halo around a streetlight on a foggy night
   */
  glowIntensity: number;

  /**
   * Life decrement per animation frame (typically at 60 FPS)
   * VISUAL IMPACT:
   * - 0.010: Slow fade (~1.7 seconds to disappear)
   * - 0.015: Medium fade (~1.1 seconds) - RECOMMENDED
   * - 0.025: Fast fade (~0.7 seconds)
   *
   * Controls how quickly particles become transparent and die.
   * Lower values = longer-lasting particles
   */
  fadeRate: number;

  /**
   * Size multiplier per animation frame (0 < shrinkRate < 1)
   * VISUAL IMPACT:
   * - 0.995: Very slow shrink, particles stay nearly same size
   * - 0.985: Moderate shrink - RECOMMENDED (40% original size after 60 frames)
   * - 0.970: Fast shrink, particles quickly become tiny
   *
   * Combined with fadeRate, creates the "dissipating" visual effect.
   */
  shrinkRate: number;

  /**
   * Upward drift acceleration per frame (added to vy each frame)
   * VISUAL: Negative values make particles float upward over time
   * Creates the "rising heat" or "bubbles ascending" effect
   *
   * Typical value: -0.02 to -0.05
   * 0 = no drift, particles move in straight lines
   */
  upwardDrift: number;

  /**
   * Available colors for random selection
   * VISUAL: Each spawned particle gets a random color from this array
   * Colors should be harmonious for a cohesive effect.
   *
   * Default palette: blue (#6366f1), purple (#a855f7), pink (#ec4899)
   * Matches Tailwind indigo-500, violet-500, pink-500
   */
  colors: string[];
}

/**
 * Default configuration for subtle, elegant particles
 *
 * VISUAL RESULT:
 * - 12 small particles (6-10px) spawn per click
 * - Move outward at moderate speed (2-3.5 px/frame)
 * - Have a visible but not overwhelming glow (15px blur)
 * - Fade and shrink over approximately 1 second
 * - Float gently upward as they age
 * - Colors: cool blue/purple/pink palette
 */
export const DEFAULT_PARTICLE_CONFIG: ParticleConfig = {
  particleCount: 12,
  minSize: 6,
  maxSize: 10,
  minSpeed: 2.0,
  maxSpeed: 3.5,
  glowIntensity: 15,
  fadeRate: 0.016,    // ~60 frames = 1 second to fade out
  shrinkRate: 0.985,  // ~40% original size after 60 frames
  upwardDrift: -0.03, // Gentle upward float
  colors: [
    '#6366f1', // Indigo (blue-ish)
    '#a855f7', // Violet (purple)
    '#ec4899', // Pink
  ],
};

/**
 * Pre-configured visual presets for different use cases
 *
 * Each preset creates a distinct visual "mood"
 */
export const PARTICLE_PRESETS = {
  /**
   * SUBTLE preset
   * VISUAL: Small, quick, minimal particles
   * USE CASE: Professional applications, frequent interactions
   * IMPRESSION: Polished, refined, not distracting
   */
  subtle: {
    particleCount: 8,
    minSize: 4,
    maxSize: 7,
    minSpeed: 2.0,
    maxSpeed: 3.0,
    glowIntensity: 10,
    fadeRate: 0.020,    // Faster fade (~0.8 seconds)
    shrinkRate: 0.980,
    upwardDrift: -0.02,
    colors: ['#6366f1', '#a855f7', '#ec4899'],
  } satisfies ParticleConfig,

  /**
   * ELEGANT preset (default)
   * VISUAL: Balanced size and count, moderate glow
   * USE CASE: Standard interactions, wikilink clicks
   * IMPRESSION: Premium UI polish, noticeable but tasteful
   */
  elegant: DEFAULT_PARTICLE_CONFIG,

  /**
   * VIBRANT preset
   * VISUAL: More particles, larger, longer-lasting
   * USE CASE: Celebrations, achievements, special actions
   * IMPRESSION: Playful, energetic, celebratory
   */
  vibrant: {
    particleCount: 20,
    minSize: 8,
    maxSize: 14,
    minSpeed: 2.5,
    maxSpeed: 4.5,
    glowIntensity: 20,
    fadeRate: 0.012,    // Slower fade (~1.4 seconds)
    shrinkRate: 0.990,
    upwardDrift: -0.04,
    colors: ['#6366f1', '#a855f7', '#ec4899', '#f472b6'],
  } satisfies ParticleConfig,

  /**
   * MINIMAL preset
   * VISUAL: Very few, very small, very quick particles
   * USE CASE: High-frequency interactions, accessibility considerations
   * IMPRESSION: Almost imperceptible polish
   */
  minimal: {
    particleCount: 5,
    minSize: 3,
    maxSize: 5,
    minSpeed: 2.5,
    maxSpeed: 3.5,
    glowIntensity: 8,
    fadeRate: 0.025,    // Fast fade (~0.7 seconds)
    shrinkRate: 0.975,
    upwardDrift: -0.01,
    colors: ['#6366f1', '#a855f7'],
  } satisfies ParticleConfig,
} as const;

export type ParticlePreset = keyof typeof PARTICLE_PRESETS;
