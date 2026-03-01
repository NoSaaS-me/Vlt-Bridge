import type { OnboardingStep } from './types';

/**
 * Onboarding tour steps.
 *
 * Add/remove/reorder steps here — the tour engine reads this array.
 * Each `target` is a CSS selector matching a `data-tour` attribute
 * in the DOM. Add `data-tour="<id>"` to the element you want highlighted.
 */
export const onboardingSteps: OnboardingStep[] = [
  {
    id: 'welcome',
    target: '[data-tour="app-title"]',
    title: 'Welcome to Vault.MCP',
    description:
      'This is your AI-first documentation workspace. ' +
      'AI agents write and update docs via MCP, while you read and edit through this UI. ' +
      "Let's take a quick tour of the key areas.",
    position: 'bottom',
    spotlightPadding: 16,
  },
  {
    id: 'sidebar',
    target: '[data-tour="sidebar"]',
    title: 'Vault Explorer',
    description:
      'Your notes and files live here. ' +
      'Create new notes, upload files, organize into folders, and search across everything. ' +
      'Click any note to open it in the main pane.',
    position: 'right',
  },
  {
    id: 'main-pane',
    target: '[data-tour="main-pane"]',
    title: 'Main Content Area',
    description:
      'This is where notes are displayed. You can read, edit, and preview Markdown. ' +
      'Wikilinks ([[like this]]) connect notes together. ' +
      'PDFs, images, and other files also render here.',
    position: 'left',
  },
  {
    id: 'chat-button',
    target: '[data-tour="chat-button"]',
    title: 'AI Planning Agent',
    description:
      'Open the AI chat panel to ask questions about your vault, ' +
      'get summaries, or have the Oracle reason about your codebase. ' +
      'It uses RAG to ground answers in your actual documentation.',
    position: 'bottom',
  },
  {
    id: 'threads-button',
    target: '[data-tour="threads-button"]',
    title: 'Vlt Threads',
    description:
      'Threads are persistent reasoning chains. ' +
      'AI agents log decisions, insights, and progress here. ' +
      'Think of them as a development journal that survives context resets.',
    position: 'bottom',
  },
  {
    id: 'settings-button',
    target: '[data-tour="settings-button"]',
    title: 'Settings',
    description:
      "Click to open Settings. Let's take a quick look inside.",
    position: 'bottom',
  },
  {
    id: 'settings-overview',
    target: '[data-tour="settings-tabs"]',
    title: 'Settings Overview',
    description:
      'Settings are organized into tabs: Account, Models, Context, Rules, Notifications, and Oracle. ' +
      'Each tab covers a different aspect of how the app and AI agents behave.',
    position: 'bottom',
    route: '/settings',
    spotlightPadding: 12,
  },
  {
    id: 'settings-api-token',
    target: '[data-tour="settings-account-tab"]',
    title: 'Account & MCP Token',
    description:
      'The Account tab has your API token for connecting MCP clients like Claude Desktop or Claude Code. ' +
      'Copy it into your MCP config to give AI agents access to your vault.',
    position: 'bottom',
    route: '/settings',
  },
  {
    id: 'settings-models',
    target: '[data-tour="settings-models-tab"]',
    title: 'AI Model Configuration',
    description:
      'The Models tab lets you choose which AI provider and model powers the Oracle — ' +
      'OpenAI, Anthropic, Google, or OpenRouter are all supported.',
    position: 'bottom',
    route: '/settings',
  },
  {
    id: 'settings-oracle',
    target: '[data-tour="settings-oracle-tab"]',
    title: 'Oracle MCP Toggle',
    description:
      'The Oracle tab controls whether AI agents can call the oracle via MCP. ' +
      "Disable it if you want agents to use your vault without the Oracle's reasoning layer.",
    position: 'bottom',
    route: '/settings',
  },
];
