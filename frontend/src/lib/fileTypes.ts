import {
  FileText,
  FileImage,
  FileVideo,
  FileAudio,
  FileCode,
  FileSpreadsheet,
  File,
  type LucideIcon,
} from 'lucide-react';

export type FileCategory =
  | 'markdown'
  | 'pdf'
  | 'image'
  | 'audio'
  | 'video'
  | 'html'
  | 'spreadsheet'
  | 'text'
  | 'unknown';

const EXT_MAP: Record<string, FileCategory> = {
  '.md': 'markdown',
  '.pdf': 'pdf',
  '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
  '.gif': 'image', '.webp': 'image', '.svg': 'image',
  '.mp3': 'audio', '.wav': 'audio', '.ogg': 'audio', '.flac': 'audio',
  '.mp4': 'video', '.webm': 'video', '.mov': 'video',
  '.html': 'html', '.htm': 'html',
  '.csv': 'spreadsheet',
  '.txt': 'text',
};

export function getFileCategory(path: string): FileCategory {
  const ext = path.slice(path.lastIndexOf('.')).toLowerCase();
  return EXT_MAP[ext] ?? 'unknown';
}

const ICON_MAP: Record<FileCategory, LucideIcon> = {
  markdown: FileText,
  pdf: File,
  image: FileImage,
  audio: FileAudio,
  video: FileVideo,
  html: FileCode,
  spreadsheet: FileSpreadsheet,
  text: FileText,
  unknown: File,
};

export function getFileIcon(category: FileCategory): LucideIcon {
  return ICON_MAP[category] ?? File;
}

export function isEditable(category: FileCategory): boolean {
  return category === 'markdown' || category === 'html' || category === 'spreadsheet' || category === 'text';
}

export function isBinary(category: FileCategory): boolean {
  return category === 'pdf' || category === 'image' || category === 'audio' || category === 'video';
}

export function isAsset(category: FileCategory): boolean {
  return category !== 'markdown';
}
