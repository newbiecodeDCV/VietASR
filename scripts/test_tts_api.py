#!/usr/bin/env python3
"""Test VCC TTS API với một sample text."""

import os
import sys
import base64
import requests
from pathlib import Path

# TTS API Configuration
TTS_API_URL = "https://speech.aiservice.vn/tts/api/v1/synthesis"
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"

def test_tts_api(text: str, output_path: str = "/tmp/tts_test.wav") -> bool:
    """Test TTS API with sample text."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": PRIVATE_TOKEN
    }
    
    payload = {
        "text": text,
        "speaker_id": "hn-quynhanh",  # Giọng nữ Hà Nội
        "voice": "news",
        "output_format": "wav",
        "speed": 1.0,
        "return_binary": 1  # Trả về base64
    }
    
    print(f"📤 Calling TTS API...")
    print(f"   URL: {TTS_API_URL}")
    print(f"   Text: {text}")
    print(f"   Speaker: {payload['speaker_id']}")
    
    try:
        response = requests.post(
            TTS_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response keys: {result.keys()}")
            
            # Check for audio data
            if "audio" in result:
                audio_data = base64.b64decode(result["audio"])
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                file_size = os.path.getsize(output_path)
                print(f"✅ Success! Audio saved to: {output_path}")
                print(f"   File size: {file_size:,} bytes")
                return True
            elif "url" in result:
                print(f"   Audio URL: {result['url']}")
                return True
            else:
                print(f"   Full response: {result}")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


if __name__ == "__main__":
    # Test với câu chứa từ CS
    test_text = "Bạn có thể upload file lên cloud không"
    
    print("=" * 60)
    print("  VCC TTS API Test")
    print("=" * 60)
    
    success = test_tts_api(test_text)
    
    print("=" * 60)
    print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
