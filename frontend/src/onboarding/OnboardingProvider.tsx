import { createContext, useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { OnboardingState, OnboardingContextValue } from './types';
import { onboardingSteps } from './steps';

const STORAGE_KEY = 'vlt-onboarding-v1';

export const OnboardingContext = createContext<OnboardingContextValue | null>(null);

export function OnboardingProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<OnboardingState>(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'dismissed') {
      return { isActive: false, currentStep: 0, dismissed: true };
    }
    return { isActive: true, currentStep: 0, dismissed: false };
  });

  useEffect(() => {
    const step = onboardingSteps[state.currentStep];
    if (state.isActive && step?.beforeEnter) {
      step.beforeEnter();
    }
  }, [state.currentStep, state.isActive]);

  const start = useCallback(() => {
    setState({ isActive: true, currentStep: 0, dismissed: false });
  }, []);

  const skip = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, 'dismissed');
    setState(prev => ({ ...prev, isActive: false, dismissed: true }));
  }, []);

  const next = useCallback(() => {
    setState(prev => {
      const nextIndex = prev.currentStep + 1;
      if (nextIndex >= onboardingSteps.length) {
        localStorage.setItem(STORAGE_KEY, 'dismissed');
        return { ...prev, isActive: false, dismissed: true };
      }
      return { ...prev, currentStep: nextIndex };
    });
  }, []);

  const back = useCallback(() => {
    setState(prev => ({
      ...prev,
      currentStep: Math.max(0, prev.currentStep - 1),
    }));
  }, []);

  const goTo = useCallback((index: number) => {
    setState(prev => ({
      ...prev,
      currentStep: Math.max(0, Math.min(index, onboardingSteps.length - 1)),
    }));
  }, []);

  const value: OnboardingContextValue = {
    state,
    steps: onboardingSteps,
    start,
    next,
    back,
    skip,
    goTo,
  };

  return (
    <OnboardingContext.Provider value={value}>
      {children}
    </OnboardingContext.Provider>
  );
}
