import { useState, useEffect, useRef, useCallback } from 'react';
import { Save, Play, Square, Search, Volume2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  getTtsSettings,
  saveTtsSettings,
  getVoices,
  type Voice,
  type TtsSettingsResponse,
} from '@/services/tts-settings';

const ELEVENLABS_MODELS = [
  { id: 'eleven_multilingual_v2', label: 'Multilingual v2' },
  { id: 'eleven_turbo_v2_5', label: 'Turbo v2.5' },
  { id: 'eleven_flash_v2_5', label: 'Flash v2.5' },
  { id: 'eleven_monolingual_v1', label: 'English v1' },
];

export function TtsSettings() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [settings, setSettings] = useState<TtsSettingsResponse | null>(null);
  const [selectedVoiceId, setSelectedVoiceId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('eleven_multilingual_v2');
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewingId, setPreviewingId] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [voiceList, ttsSettings] = await Promise.all([getVoices(), getTtsSettings()]);
      setVoices(voiceList);
      setSettings(ttsSettings);
      setSelectedVoiceId(ttsSettings.voice_id);
      setSelectedModel(ttsSettings.model || 'eleven_multilingual_v2');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load TTS data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    return () => {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current = null;
      }
    };
  }, [loadData]);

  const handleSave = async () => {
    if (!selectedVoiceId) return;
    setIsSaving(true);
    setSaved(false);
    try {
      await saveTtsSettings({ voice_id: selectedVoiceId, model: selectedModel });
      setSettings({ voice_id: selectedVoiceId, model: selectedModel });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setIsSaving(false);
    }
  };

  const handlePreview = (voice: Voice) => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }

    if (previewingId === voice.voice_id) {
      setPreviewingId(null);
      return;
    }

    if (!voice.preview_url) return;

    const audio = new Audio(voice.preview_url);
    previewAudioRef.current = audio;
    setPreviewingId(voice.voice_id);

    audio.onended = () => {
      setPreviewingId(null);
      previewAudioRef.current = null;
    };
    audio.onerror = () => {
      setPreviewingId(null);
      previewAudioRef.current = null;
    };
    audio.play().catch(() => {
      setPreviewingId(null);
      previewAudioRef.current = null;
    });
  };

  const filteredVoices = voices.filter((v) => {
    const matchesSearch =
      !searchQuery || v.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = categoryFilter === 'all' || v.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  const categories = [...new Set(voices.map((v) => v.category))];
  const hasChanges =
    settings && (selectedVoiceId !== settings.voice_id || selectedModel !== settings.model);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Text-to-Speech</CardTitle>
          <CardDescription>Loading voice options...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Volume2 className="h-5 w-5" />
            Text-to-Speech
          </CardTitle>
          <CardDescription>Choose your preferred voice and model for note reading</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-full max-w-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ELEVENLABS_MODELS.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Voice Selection</CardTitle>
              <CardDescription>
                {voices.length} voices available
                {selectedVoiceId && (
                  <> &mdash; selected: <strong>{voices.find((v) => v.voice_id === selectedVoiceId)?.name}</strong></>
                )}
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={loadData} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3 items-center">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search voices..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All categories</SelectItem>
                {categories.map((cat) => (
                  <SelectItem key={cat} value={cat}>
                    {cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="border rounded-md max-h-[400px] overflow-y-auto">
            {filteredVoices.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground text-sm">
                No voices match your filters
              </div>
            ) : (
              filteredVoices.map((voice) => (
                <div
                  key={voice.voice_id}
                  className={`flex items-center gap-3 px-4 py-3 border-b last:border-b-0 cursor-pointer hover:bg-accent/50 transition-colors ${
                    selectedVoiceId === voice.voice_id ? 'bg-accent' : ''
                  }`}
                  onClick={() => setSelectedVoiceId(voice.voice_id)}
                >
                  <div
                    className={`h-4 w-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                      selectedVoiceId === voice.voice_id
                        ? 'border-primary'
                        : 'border-muted-foreground/30'
                    }`}
                  >
                    {selectedVoiceId === voice.voice_id && (
                      <div className="h-2 w-2 rounded-full bg-primary" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm truncate">{voice.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {voice.category}
                      </Badge>
                      {voice.labels.gender && (
                        <Badge variant="outline" className="text-xs">
                          {voice.labels.gender}
                        </Badge>
                      )}
                      {voice.labels.accent && (
                        <Badge variant="outline" className="text-xs">
                          {voice.labels.accent}
                        </Badge>
                      )}
                    </div>
                  </div>

                  {voice.preview_url && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-shrink-0 h-8 w-8 p-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePreview(voice);
                      }}
                      title={previewingId === voice.voice_id ? 'Stop preview' : 'Preview voice'}
                    >
                      {previewingId === voice.voice_id ? (
                        <Square className="h-3.5 w-3.5" />
                      ) : (
                        <Play className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  )}
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={isSaving || !hasChanges}>
          {saved ? (
            <>
              <svg className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Saved
            </>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              {isSaving ? 'Saving...' : 'Save TTS Settings'}
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
