import type { ReactNode } from 'react';

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right' | 'auto';

export interface OnboardingStep {
  /** Unique step identifier */
  id: string;
  /** CSS selector for the target element, e.g. '[data-tour="sidebar"]' */
  target: string;
  /** Step title shown in tooltip header */
  title: string;
  /** Step description shown in tooltip body */
  description: string;
  /** Preferred tooltip position relative to highlighted element */
  position?: TooltipPosition;
  /** Optional custom render function for the tooltip body (overrides description) */
  render?: (controls: StepControls) => ReactNode;
  /** Called before this step is shown — use to open panels, navigate, etc. */
  beforeEnter?: () => void;
  /** Padding (px) around the spotlight cutout. Default: 8 */
  spotlightPadding?: number;
  /** Border radius (px) for the spotlight cutout. Default: 8 */
  spotlightRadius?: number;
}

export interface StepControls {
  next: () => void;
  back: () => void;
  skip: () => void;
  currentStep: number;
  totalSteps: number;
}

export interface OnboardingState {
  isActive: boolean;
  currentStep: number;
  dismissed: boolean;
}

export interface OnboardingContextValue {
  state: OnboardingState;
  steps: OnboardingStep[];
  start: () => void;
  next: () => void;
  back: () => void;
  skip: () => void;
  /** Go to a specific step by index */
  goTo: (index: number) => void;
}
