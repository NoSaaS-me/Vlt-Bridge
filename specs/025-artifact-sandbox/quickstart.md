# Quickstart: Artifact Sandbox

## Prerequisites

- Daemon running (`vlt daemon start`)
- Backend running (`cd backend && uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`)
- Frontend running (`cd frontend && npm run dev`)
- At least one model endpoint configured (OpenRouter, z.ai, or Gemini)

## Create Your First Artifact

### Via UI

1. Open the app in browser (http://localhost:5173)
2. Navigate to the Agents page
3. Click the **Artifacts** tab (puzzle piece icon) in the left nav
4. Click **New Artifact**
5. Enter a name and description
6. The artifact directory is created and the editor opens

### Via MCP Tool (from Claude Code)

```
Use the vlt_artifact_create tool:
- name: "Hello World"
- description: "A simple test artifact"
- project_id: "default"
```

### Write Artifact Code

Create `frontend/index.html` in the artifact directory:

```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>Hello from Artifact!</h1>
  <button id="save">Save State</button>
  <p id="output"></p>
  <script src="app.js"></script>
</body>
</html>
```

Create `frontend/app.js`:

```javascript
// VltBridge is auto-injected — available globally
document.getElementById('save').addEventListener('click', async () => {
  await VltBridge.storage.set('counter', Date.now());
  const val = await VltBridge.storage.get('counter');
  document.getElementById('output').textContent = `Saved: ${val}`;
});
```

### Add a Backend (Optional)

Create `backend/main.py`:

```python
def handle(action: str, params: dict) -> dict:
    """Entry point called by the daemon for every backend request."""
    if action == "greet":
        name = params.get("name", "World")
        return {"message": f"Hello, {name}!"}
    return {"error": f"Unknown action: {action}"}

# Optional: state preservation for hot reload
def save_state() -> dict:
    return {}

def load_state(state: dict):
    pass
```

Call from frontend:
```javascript
const result = await VltBridge.backend.call("greet", {name: "Wolfe"});
console.log(result.message); // "Hello, Wolfe!"
```

### Add Tests (Optional)

Create `tests/test_backend.py`:

```python
from backend.main import handle

def test_greet():
    result = handle("greet", {"name": "Test"})
    assert result["message"] == "Hello, Test!"

def test_unknown_action():
    result = handle("unknown", {})
    assert "error" in result
```

Update `manifest.json` to include test config:
```json
{
  "tests": {
    "command": "pytest tests/ -v",
    "timeout": 30
  }
}
```

## State Machine

Advance your artifact through states:

1. **draft** → Created, not yet built
2. **building** → Agent or user is actively writing code
3. **testing** → Tests are running (auto-triggered on code change)
4. **reviewing** → Vision model reviewing screenshot (if configured)
5. **approved** → Ready for use
6. **deployed** → Running persistently (for `persistent` type artifacts)

States auto-advance when tests pass and review succeeds. Code changes in `approved` state demote back to `testing`.

## Hot Reload

Hot reload is automatic when the artifact is active:

- **CSS changes**: Stylesheet swapped without page reload
- **JS/HTML changes**: Full iframe reload (with state preservation if `__vlt_save_state` is defined)
- **Python changes**: Backend restarted (with state preservation if `save_state`/`load_state` are defined)
- **Debounce**: 500ms window to batch rapid file writes

## Configure Vision Model (Optional)

1. Go to Settings → Models tab
2. Set the **Vision Model** selector to a vision-capable model
3. Or let the system auto-detect from your configured providers

## Export & Share

- Click **Export** on any artifact to download a zip
- Click **Import** to load a previously exported artifact zip
- Connector credentials are NOT included in exports
