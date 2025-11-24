# UI Fix Required - Restart Dev Server

## Issue
PostCSS is still loading cached Tailwind CSS v4 modules even though we've downgraded to v3.

## Solution
**Restart the Vite dev server** to pick up the correct Tailwind CSS version:

```bash
# Stop the current dev server (Ctrl+C)
# Then restart:
cd /home/wolfe/Projects/Document-MCP/frontend
npm run dev
```

## What Was Done
1. ✅ Uninstalled Tailwind CSS v4.1.17
2. ✅ Installed Tailwind CSS v3.4.17 (compatible with shadcn/ui)
3. ✅ Cleared Vite cache
4. ⏳ Needs dev server restart to complete fix

## Verification
After restarting, navigate to http://localhost:5173/ and you should see:
- Full UI without errors
- Document Viewer header
- Search bar
- Directory tree (empty until backend runs)
- Main content pane
- Footer with note count

The API 500 errors are expected and will resolve once backend routes are implemented.

