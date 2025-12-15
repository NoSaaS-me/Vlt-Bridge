---
title: Document Viewer
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: A bridge for all agents to pass markdown
tags:
  - building-mcp-track-consumer
  - building-mcp-track-enterprise
  - building-mcp-track-creative

---
(the table above is for HF deployments)
# Document Viewer - AI-Powered Documentation System

An Obsidian-style documentation system where AI agents and humans collaborate on creating and maintaining documentation.

Team members: Bigwolfe, AbelFace, Wothmag07

Links:

https://huggingface.co/spaces/MCP-1st-Birthday/Vault.MCP

https://youtu.be/vHCsI1a7MUY

https://github.com/bigwolfeman/Document-MCP

https://www.instagram.com/p/DRqCPOTEcKl/?igsh=MTlhZjJmZ2podzlmcA==

https://x.com/a_i_belgpt/status/1994900016553480593?t=Bqv3nYAluzLv2BMqVyuTSQ&s=09

https://www.linkedin.com/posts/jose-pinales_vault-mcp-share-7400699102797406208-zZwq?utm_source=share&utm_medium=member_android&rcm=ACoAACp_iZMBHLcV-oYR8V0pIIkrTziBPQKXTw4

## ⚠️ Demo Mode

**This is a demonstration instance hosted on HuggingFace with ephemeral storage.**

- All data is temporary and resets on server restart
- Demo content is automatically seeded on each startup
- For production use, deploy your own instance with persistent storage

## 🎯 Features

- **Wikilinks** - Link between notes using `[[Note Name]]` syntax
- **Full-Text Search** - BM25 ranking with recency bonus
- **Backlinks** - Automatically track note references
- **Split-Pane Editor** - Live markdown preview
- **MCP Integration** - AI agents can read/write via Model Context Protocol
- **Multi-Tenant** - Each user gets an isolated vault (HF OAuth)
- **ChatGPT App Integration** - Loads as an iFrame in ChatGPT (while still letting other agents communicate over the same MCP endpoint!)
- **TTS** - Read your notes outloud using Elevenlabs TTS
- **Gemini AI Integration** Using Gemini API there is a full tool calling chatbot that can help organize and talk about the notes inside Vault.MCP
- **Vector DB/LlamaIndex RAG** Full RAG capacity through LlamaIndex

## 🚀 Getting Started

1. Click **"Sign in with Hugging Face"** to authenticate
2. Browse the pre-seeded demo notes
3. Try searching, creating, and editing notes
4. Check out the wikilinks between documents

## 🤖 AI Agent Access (MCP)

After signing in, go to **Settings** to get your API token for MCP access:

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "url": "https://YOUR_USERNAME-Document-MCP.hf.space/mcp",
      "transport": "http",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN"
      }
    }
  }
}
```

For local experiments you can still run the MCP server via STDIO—use the "Local Development" snippet shown in Settings.

AI agents can then use these tools:
- `list_notes` - Browse vault
- `read_note` - Read note content
- `write_note` - Create/update notes
- `search_notes` - Full-text search
- `get_backlinks` - Find references
- `get_tags` - List all tags

## 🏗️ Tech Stack

**Backend:**
- FastAPI - HTTP API server
- FastMCP - MCP server for AI integration
- SQLite FTS5 - Full-text search
- python-frontmatter - YAML metadata

**Frontend:**
- React + Vite - Modern web framework
- shadcn/ui - UI components
- Tailwind CSS - Styling
- react-markdown - Markdown rendering

## 📖 Documentation

Key demo notes to explore:

- **Getting Started** - Introduction and overview
- **API Documentation** - REST API reference
- **MCP Integration** - AI agent configuration
- **Wikilink Examples** - How linking works
- **Architecture Overview** - System design
- **Search Features** - Full-text search details

## ⚙️ Deploy Your Own

Want persistent storage and full control? Deploy your own instance:

1. Clone the repository
2. Set up HF OAuth app
3. Configure environment variables
4. Deploy to HF Spaces or any Docker host

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

## 🔒 Privacy & Data

- **Multi-tenant**: Each HF user gets an isolated vault
- **Demo data**: Resets on restart (ephemeral storage)
- **OAuth**: Secure authentication via Hugging Face
- **No tracking**: We don't collect analytics or personal data

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

---

Built with ❤️ for the AI-human documentation collaboration workflow

