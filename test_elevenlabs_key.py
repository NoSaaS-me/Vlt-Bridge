#!/usr/bin/env python3
"""Test script to verify ElevenLabs API key."""

import os
import httpx

# Read API key from environment
api_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

print("=" * 60)
print("ElevenLabs API Key Test")
print("=" * 60)

if not api_key:
    print("❌ ELEVENLABS_API_KEY is not set!")
    print("\nSet it with:")
    print("  export ELEVENLABS_API_KEY='your-key-here'")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
print(f"✓ Voice ID: {voice_id}")
print()

# Test API key by calling the voices endpoint (free endpoint)
print("Testing API key with /voices endpoint...")
try:
    response = httpx.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key},
        timeout=10.0
    )

    if response.status_code == 200:
        print("✅ API key is VALID!")
        data = response.json()
        voices = data.get("voices", [])
        print(f"\nYou have access to {len(voices)} voices:")
        for voice in voices[:5]:  # Show first 5
            print(f"  - {voice['name']}: {voice['voice_id']}")
        if len(voices) > 5:
            print(f"  ... and {len(voices) - 5} more")
    elif response.status_code == 401:
        print("❌ API key is INVALID (401 Unauthorized)")
        print("\nPossible reasons:")
        print("  1. The API key is incorrect or has a typo")
        print("  2. The API key has been revoked or expired")
        print("  3. You're using the wrong ElevenLabs account")
        print("\nTo fix:")
        print("  1. Go to https://elevenlabs.io/app/settings/api-keys")
        print("  2. Generate a new API key")
        print("  3. Update ELEVENLABS_API_KEY in your .env or Space secrets")
    else:
        print(f"⚠️  Unexpected status code: {response.status_code}")
        print(f"Response: {response.text[:200]}")

except Exception as e:
    print(f"❌ Request failed: {e}")

print("=" * 60)
