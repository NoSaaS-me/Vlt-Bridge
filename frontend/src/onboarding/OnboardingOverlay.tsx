import React, {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useCallback,
} from 'react';
import { createPortal } from 'react-dom';
import { useOnboarding } from './useOnboarding';
import { OnboardingTooltip } from './OnboardingTooltip';
import type { TooltipPosition, StepControls } from './types';

// ─── Geometry helpers ────────────────────────────────────────────────────────

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const TOOLTIP_WIDTH = 320;
const TOOLTIP_OFFSET = 12;

/**
 * Build a rounded-rect SVG path for the spotlight "hole".
 * Uses arc commands at each corner so the cutout matches spotlightRadius.
 */
function roundedRectPath(
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): string {
  // Clamp radius so it never exceeds half the smaller dimension
  r = Math.min(r, w / 2, h / 2);

  return [
    `M ${x + r} ${y}`,
    `H ${x + w - r}`,
    `A ${r} ${r} 0 0 1 ${x + w} ${y + r}`,
    `V ${y + h - r}`,
    `A ${r} ${r} 0 0 1 ${x + w - r} ${y + h}`,
    `H ${x + r}`,
    `A ${r} ${r} 0 0 1 ${x} ${y + h - r}`,
    `V ${y + r}`,
    `A ${r} ${r} 0 0 1 ${x + r} ${y}`,
    'Z',
  ].join(' ');
}

/**
 * Full SVG fill-rule="evenodd" path:
 *   outer rect (full viewport) minus inner rounded rect (spotlight hole).
 */
function buildMaskPath(
  vw: number,
  vh: number,
  rect: Rect | null,
  pad: number,
  radius: number,
): string {
  const outer = `M 0 0 H ${vw} V ${vh} H 0 Z`;

  if (!rect) {
    // No target element — no hole, full dim
    return outer;
  }

  const hx = rect.x - pad;
  const hy = rect.y - pad;
  const hw = rect.width + pad * 2;
  const hh = rect.height + pad * 2;
  const inner = roundedRectPath(hx, hy, hw, hh, radius);

  return `${outer} ${inner}`;
}

/**
 * Compute tooltip (top, left) given target rect, viewport, and preferred side.
 * Returns position clamped to viewport with a small margin.
 */
function computeTooltipPosition(
  targetRect: Rect | null,
  tooltipHeight: number,
  vw: number,
  vh: number,
  position: TooltipPosition,
): { top: number; left: number } {
  const margin = 8; // minimum distance from viewport edge

  if (!targetRect) {
    // Center in viewport
    return {
      top: Math.max(margin, (vh - tooltipHeight) / 2),
      left: Math.max(margin, (vw - TOOLTIP_WIDTH) / 2),
    };
  }

  const { x, y, width, height } = targetRect;

  // Resolve 'auto' → side with most space
  let side = position as Exclude<TooltipPosition, 'auto'>;
  if (position === 'auto') {
    const spaceAbove = y;
    const spaceBelow = vh - (y + height);
    const spaceLeft = x;
    const spaceRight = vw - (x + width);
    const max = Math.max(spaceAbove, spaceBelow, spaceLeft, spaceRight);
    if (max === spaceBelow) side = 'bottom';
    else if (max === spaceAbove) side = 'top';
    else if (max === spaceRight) side = 'right';
    else side = 'left';
  }

  let top = 0;
  let left = 0;

  switch (side) {
    case 'bottom':
      top = y + height + TOOLTIP_OFFSET;
      left = x + width / 2 - TOOLTIP_WIDTH / 2;
      break;
    case 'top':
      top = y - tooltipHeight - TOOLTIP_OFFSET;
      left = x + width / 2 - TOOLTIP_WIDTH / 2;
      break;
    case 'right':
      top = y + height / 2 - tooltipHeight / 2;
      left = x + width + TOOLTIP_OFFSET;
      break;
    case 'left':
      top = y + height / 2 - tooltipHeight / 2;
      left = x - TOOLTIP_WIDTH - TOOLTIP_OFFSET;
      break;
  }

  // Clamp to viewport
  top = Math.max(margin, Math.min(top, vh - tooltipHeight - margin));
  left = Math.max(margin, Math.min(left, vw - TOOLTIP_WIDTH - margin));

  return { top, left };
}

// ─── Component ───────────────────────────────────────────────────────────────

export function OnboardingOverlay() {
  const { state, steps, next, back, skip } = useOnboarding();

  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  const tooltipRef = useRef<HTMLDivElement>(null);
  // Track measured tooltip height for vertical clamping
  const [tooltipHeight, setTooltipHeight] = useState(140);

  const currentStepIndex = state.currentStep;
  const currentStepDef = steps[currentStepIndex];

  // ── Measure target element ──────────────────────────────────────────────

  const measureTarget = useCallback(() => {
    if (!currentStepDef) {
      setTargetRect(null);
      return;
    }
    const el = document.querySelector<HTMLElement>(currentStepDef.target);
    if (!el) {
      setTargetRect(null);
      return;
    }
    const r = el.getBoundingClientRect();
    setTargetRect({ x: r.left, y: r.top, width: r.width, height: r.height });
  }, [currentStepDef]);

  // Initial measurement (synchronous, before paint)
  useLayoutEffect(() => {
    if (!state.isActive) return;
    measureTarget();
  }, [state.isActive, measureTarget]);

  // Resize listener
  useEffect(() => {
    if (!state.isActive) return;

    const handleResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
      measureTarget();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [state.isActive, measureTarget]);

  // Measure tooltip height after render
  useEffect(() => {
    if (tooltipRef.current) {
      setTooltipHeight(tooltipRef.current.offsetHeight);
    }
  });

  if (!state.isActive || !currentStepDef) return null;

  // ── Derived values ─────────────────────────────────────────────────────

  const pad = currentStepDef.spotlightPadding ?? 8;
  const radius = currentStepDef.spotlightRadius ?? 8;
  const maskPath = buildMaskPath(
    viewport.width,
    viewport.height,
    targetRect,
    pad,
    radius,
  );

  const controls: StepControls = {
    next,
    back,
    skip,
    currentStep: currentStepIndex,
    totalSteps: steps.length,
  };

  const tooltipPos = computeTooltipPosition(
    targetRect,
    tooltipHeight,
    viewport.width,
    viewport.height,
    currentStepDef.position ?? 'auto',
  );

  // ── Render ─────────────────────────────────────────────────────────────

  const overlay = (
    <>
      {/* SVG spotlight mask */}
      <svg
        style={{
          position: 'fixed',
          inset: 0,
          width: '100%',
          height: '100%',
          zIndex: 50,
          pointerEvents: 'auto',
          // Smooth opacity transition when overlay mounts/unmounts
          transition: 'opacity 200ms ease',
        }}
        aria-hidden="true"
      >
        <path
          d={maskPath}
          fill="rgba(0, 0, 0, 0.70)"
          fillRule="evenodd"
          // The filled region captures clicks (intentional no-op)
          onClick={(e) => e.stopPropagation()}
          style={{
            // Smooth path transition between steps
            transition: 'd 250ms ease',
            cursor: 'default',
          }}
        />
      </svg>

      {/* Tooltip positioned in a fixed container above the SVG */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 51,
          pointerEvents: 'none', // pass-through except for tooltip itself
        }}
        aria-live="polite"
      >
        <div
          ref={tooltipRef}
          style={{
            position: 'absolute',
            top: tooltipPos.top,
            left: tooltipPos.left,
            width: TOOLTIP_WIDTH,
            pointerEvents: 'auto',
          }}
        >
          <OnboardingTooltip
            step={currentStepDef}
            controls={controls}
          />
        </div>
      </div>
    </>
  );

  return createPortal(overlay, document.body);
}
