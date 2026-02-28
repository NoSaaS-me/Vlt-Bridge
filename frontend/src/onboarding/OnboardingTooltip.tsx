import React from 'react';
import { Button } from '@/components/ui/button';
import type { OnboardingStep, StepControls } from './types';

interface OnboardingTooltipProps {
  step: OnboardingStep;
  controls: StepControls;
  /** Positioned absolutely — receives top/left from parent */
  style?: React.CSSProperties;
}

export function OnboardingTooltip({ step, controls, style }: OnboardingTooltipProps) {
  const { next, back, skip, currentStep, totalSteps } = controls;
  const isFirst = currentStep === 0;
  const isLast = currentStep === totalSteps - 1;

  return (
    <div
      style={{ maxWidth: 320, ...style }}
      className="bg-card border border-border rounded-lg shadow-xl p-4 animate-in fade-in-0 duration-200"
    >
      {/* Header */}
      <p className="text-sm font-semibold text-foreground mb-2">{step.title}</p>

      {/* Body */}
      <div className="text-xs text-muted-foreground leading-relaxed mb-4">
        {step.render ? step.render(controls) : step.description}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between gap-2">
        {/* Step counter */}
        <span className="text-xs text-muted-foreground shrink-0">
          Step {currentStep + 1} of {totalSteps}
        </span>

        {/* Button group */}
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" className="text-xs h-7 px-2" onClick={skip}>
            Skip
          </Button>

          {!isFirst && (
            <Button variant="outline" size="sm" className="text-xs h-7 px-2" onClick={back}>
              Back
            </Button>
          )}

          <Button variant="default" size="sm" className="text-xs h-7 px-2" onClick={next}>
            {isLast ? 'Finish' : 'Next'}
          </Button>
        </div>
      </div>
    </div>
  );
}
