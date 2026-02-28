import { useContext } from 'react';
import { OnboardingContext } from './OnboardingProvider';
import type { OnboardingContextValue } from './types';

export function useOnboarding(): OnboardingContextValue {
  const ctx = useContext(OnboardingContext);
  if (!ctx) {
    throw new Error('useOnboarding must be used within an OnboardingProvider');
  }
  return ctx;
}
