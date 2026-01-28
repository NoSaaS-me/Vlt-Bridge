# Quickstart: Vlt Desktop App & Daemon

## Prerequisites

- **Pixi**: Install Pixi (pixi.sh) for managing the Mojo environment.
- **Rust**: Install Rust (rustup.rs) for Tauri.
- **Node.js**: v18+ for frontend.
- **Tauri CLI**: `cargo install tauri-cli`.

## 1. Setup the Daemon (Mojo via Pixi)

The daemon is the core orchestrator. We use `pixi` to manage the Mojo/MAX environment.

```bash
# Navigate to daemon package
cd packages/vlt-daemon

# Install dependencies
pixi install

# Run the daemon locally
pixi run start
```

## 2. Setup the Server (FastAPI)

The server acts as the authority.

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.api.main:app --reload --port 8000
```

## 3. Setup the Desktop App (Tauri)

The native client interface.

```bash
# Navigate to desktop app
cd desktop-app

# Install frontend deps
npm install

# Run in development mode
npm run tauri dev
```

## 4. Verify Connection

1.  Open the Desktop App.
2.  Go to **Settings > Daemon**.
3.  Enter `ws://localhost:9000` (or your remote server URL).
4.  Click **Connect**.
5.  Status should change to "Connected".

## 5. Run Your First Agent Task

1.  Open the **Terminal** tab in the Desktop App.
2.  Type `vlt status`.
3.  The command should execute on the daemon and stream output back instantly.