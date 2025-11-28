import { useState, useEffect } from 'react';

type FontSizePreset = 'small' | 'medium' | 'large';

interface FontSizeConfig {
  size: FontSizePreset;
  remValue: number;
}

const FONT_SIZE_PRESETS: Record<FontSizePreset, FontSizeConfig> = {
  small: { size: 'small', remValue: 0.875 },    // 14px
  medium: { size: 'medium', remValue: 1 },       // 16px (default)
  large: { size: 'large', remValue: 1.125 },     // 18px
};

interface UseFontSize {
  fontSize: FontSizePreset;
  setFontSize: (size: FontSizePreset) => void;
  isFontReady: boolean;
}

/**
 * T006, T010: Font size persistence hook with localStorage
 * Manages note content font size preference and applies CSS variable updates
 */
export function useFontSize(): UseFontSize {
  const [fontSize, setFontSizeState] = useState<FontSizePreset>(() => {
    // Load from localStorage, default to 'medium'
    const saved = localStorage.getItem('note-font-size');
    return (saved as FontSizePreset) || 'medium';
  });

  const [isFontReady, setIsFontReady] = useState(false);

  // T010: Update CSS variable whenever fontSize changes
  useEffect(() => {
    const config = FONT_SIZE_PRESETS[fontSize];
    const remValue = config.remValue;
    // Update the CSS custom property on the root element
    document.documentElement.style.setProperty('--content-font-size', `${remValue}rem`);
    // Force synchronous style recalculation to prevent FOUC/flicker
    document.documentElement.getBoundingClientRect();
    // Persist to localStorage
    localStorage.setItem('note-font-size', fontSize);
    // Signal that font styles are applied and ready
    setIsFontReady(true);
  }, [fontSize]);

  const setFontSize = (size: FontSizePreset) => {
    setFontSizeState(size);
  };

  return { fontSize, setFontSize, isFontReady };
}
