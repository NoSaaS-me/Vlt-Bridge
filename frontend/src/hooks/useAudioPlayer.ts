import { useEffect, useRef, useState } from 'react';

type PlayerStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'error';

interface AudioPlayer {
  status: PlayerStatus;
  error: string | null;
  volume: number;
  play: (src: string) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  setVolume: (level: number) => void;
}

/**
 * Lightweight audio controller for blob URLs returned by the TTS service.
 */
export function useAudioPlayer(): AudioPlayer {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [status, setStatus] = useState<PlayerStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [volume, setVolumeState] = useState<number>(() => {
    const saved = localStorage.getItem('tts-volume');
    return saved ? parseFloat(saved) : 0.7;
  });

  const cleanup = () => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.src = '';
      audioRef.current = null;
    }
  };

  const stop = () => {
    cleanup();
    setStatus('idle');
  };

  const play = (src: string) => {
    setError(null);
    cleanup();
    setStatus('loading');
    const audio = new Audio(src);
    audio.volume = volume;
    audioRef.current = audio;

    audio.oncanplay = () => {
      audio.play().catch((err) => {
        setError(err?.message || 'Failed to play audio.');
        setStatus('error');
      });
    };
    audio.onplay = () => setStatus('playing');
    audio.onpause = () => setStatus((prev) => (prev === 'loading' ? 'loading' : 'paused'));
    audio.onended = () => setStatus('idle');
    audio.onerror = () => {
      setError('Audio playback error.');
      setStatus('error');
    };
  };

  const pause = () => {
    const audio = audioRef.current;
    if (audio && !audio.paused) {
      audio.pause();
    }
  };

  const resume = () => {
    const audio = audioRef.current;
    if (audio && audio.paused) {
      audio.play().catch((err) => {
        setError(err?.message || 'Failed to resume audio.');
        setStatus('error');
      });
    }
  };

  const setVolume = (level: number) => {
    const clamped = Math.max(0, Math.min(1, level));
    setVolumeState(clamped);
    localStorage.setItem('tts-volume', clamped.toString());
    if (audioRef.current) {
      audioRef.current.volume = clamped;
    }
  };

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  return { status, error, volume, play, pause, resume, stop, setVolume };
}
