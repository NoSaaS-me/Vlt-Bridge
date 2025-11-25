# Deploying to Hugging Face Spaces

This guide walks through deploying the Document-MCP application to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co/join
2. **OAuth Application**: Created at https://huggingface.co/settings/connected-applications
3. **Git with HF CLI** (optional but recommended): `pip install huggingface_hub`

## Step 1: Create HF Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Space name**: `Document-MCP` (or your preferred name)
   - **License**: `apache-2.0` or `mit`
   - **Select the SDK**: Choose **Docker**
   - **Space hardware**: Start with **CPU basic** (free tier)
   - **Visibility**: Public or Private

3. Click **Create Space**

## Step 2: Configure OAuth Application

**IMPORTANT**: HF Spaces run at a different URL than the management page. Your app's actual URL will be detected automatically via request headers.

If you haven't already created an OAuth app:

1. Go to https://huggingface.co/settings/connected-applications
2. Click **Create new application**
3. Fill in:
   - **Application Name**: `Documentation-MCP`
   - **Homepage URL**: `https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP`
   - **Redirect URI**: `https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP/auth/callback`
   - **Scopes**: Select `openid` and `profile`

4. Click **Create**
5. **Save the Client ID and Client Secret** - you'll need these for environment variables

**Note**: The OAuth redirect URI will automatically adjust to your Space's actual running URL (e.g., `https://YOUR_USERNAME-document-mcp.hf.space/auth/callback`) via HTTP headers. The configured `HF_SPACE_URL` is only used as a fallback.

## Step 3: Set Environment Variables

In your HF Space settings, add these secrets:

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP`
2. Click **Settings** tab
3. Scroll to **Repository secrets**
4. Add the following secrets:

```
JWT_SECRET_KEY=<generate-a-random-32-char-string>
HF_OAUTH_CLIENT_ID=<your-oauth-client-id>
HF_OAUTH_CLIENT_SECRET=<your-oauth-client-secret>
HF_SPACE_URL=https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP
```

**Note on HF_SPACE_URL**: This is only a fallback. The application automatically detects its actual running URL using `X-Forwarded-Host` and `X-Forwarded-Proto` headers set by HF Spaces proxy. You can set this to the Space management URL for simplicity.

### Generating JWT_SECRET_KEY

Run this to generate a secure random key:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Or:

```bash
openssl rand -hex 32
```

## Step 4: Push Code to HF Space

### Option A: Using Git (Recommended)

```bash
# Clone your HF Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP
cd Document-MCP

# Copy all project files (excluding .git)
cp -r /path/to/Document-MCP/{backend,frontend,Dockerfile,.dockerignore} .

# Add and commit
git add .
git commit -m "Initial deployment"

# Push to HF Space
git push
```

### Option B: Using HF CLI

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload repository
cd /path/to/Document-MCP
huggingface-cli upload YOUR_USERNAME/Document-MCP . --repo-type=space
```

### Option C: Manual Upload

1. Go to **Files** tab in your Space
2. Click **Add file** → **Upload files**
3. Upload:
   - `Dockerfile`
   - `.dockerignore`
   - `backend/` directory (all contents)
   - `frontend/` directory (all contents)

## Step 5: Wait for Build

1. HF Spaces will automatically build your Docker container
2. Check the **Logs** tab to monitor build progress
3. Build typically takes 5-10 minutes for first deployment
4. Once complete, your app will be available at: `https://YOUR_USERNAME-Document-MCP.hf.space`

## Step 6: Test the Deployment

1. **Open the Space URL** in your browser
2. You should see the Login page
3. Click **Sign in with Hugging Face**
4. Authorize the OAuth app
5. You'll be redirected back to the app with a JWT token
6. Browse the demo vault with pre-seeded notes

## Step 7: MCP HTTP Access

The Space exposes the MCP server over HTTPS so tools like Claude Desktop can connect without running anything locally. Every JWT maps to an isolated vault directory (`data/vaults/<user_id>`), so each Hugging Face account sees only its own notes.

1. **Get your JWT token**:
   - Go to the in-app Settings page
   - Click **Copy token** (or generate a new one)

2. **Configure your MCP client** (example for Claude Desktop):

```json
{
  "mcpServers": {
    "obsidian-docs": {
      "transport": "http",
      "url": "https://YOUR_USERNAME-Document-MCP.hf.space/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_JWT_TOKEN_HERE"
      }
    }
  }
}
```

3. **Local development (optional)**: If you want to run the MCP server via STDIO on your laptop, use the "Local Development" snippet from the app's Settings page.

## Troubleshooting

### Build Fails

**Check logs** in the Space's "Logs" tab. Common issues:

- **npm install fails**: Check `frontend/package.json` dependencies
- **Python install fails**: Check `backend/pyproject.toml` dependencies
- **Out of memory**: Upgrade to a paid tier with more RAM

### OAuth Redirect Loop or 404 After Login

**The app now auto-detects its URL** from request headers. If OAuth fails:

- Check HF Space **application logs** for "OAuth base URL" messages to see what URL was detected
- Verify your OAuth app's Redirect URI matches one of:
  - The Space management URL: `https://huggingface.co/spaces/YOUR_USERNAME/Document-MCP/auth/callback`
  - The running app URL (shown in logs): `https://YOUR_USERNAME-document-mcp.hf.space/auth/callback`
- Ensure OAuth app scopes include `openid` and `profile`
- Try updating your OAuth app's Redirect URI to match the URL shown in the logs

### 500 Internal Server Error

- Check application logs in the Space's "Logs" tab
- Verify all environment variables are set correctly
- Ensure `JWT_SECRET_KEY` is at least 16 characters

### Data Not Persisting

**This is expected** - the demo uses ephemeral storage. Data resets on container restart. The app seeds demo content on every startup.

For persistent storage, you'd need to:
- Upgrade to HF Spaces Pro with persistent storage
- Or use external database (Supabase, PlanetScale, etc.)

## Updating Your Deployment

To deploy updates:

```bash
cd /path/to/Document-MCP
git add .
git commit -m "Update: description of changes"
git push
```

HF Spaces will automatically rebuild and redeploy.

## Monitoring

- **View logs**: Space Logs tab
- **Check build status**: Space "Building" indicator
- **Test health endpoint**: `https://YOUR_USERNAME-Document-MCP.hf.space/health`

## Cost Considerations

- **CPU basic**: Free tier, sufficient for demo/personal use
- **Persistent storage**: Requires paid tier ($5-20/month)
- **More CPU/RAM**: Upgrade if you experience slow performance

## Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **OAuth Docs**: https://huggingface.co/docs/hub/oauth
- **Docker SDK Guide**: https://huggingface.co/docs/hub/spaces-sdks-docker

