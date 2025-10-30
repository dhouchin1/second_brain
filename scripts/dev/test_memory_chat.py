#!/usr/bin/env python3
"""
Simple test script for memory-augmented chat
Run after starting the server: .venv/bin/python -m uvicorn app:app --reload --port 8082
"""

import requests
import json
import time

BASE_URL = "http://localhost:8082"

def test_chat():
    """Test basic chat with memory"""

    print("=" * 70)
    print("Memory-Augmented Chat Test")
    print("=" * 70)

    # First message - establish preference
    print("\n1️⃣  Establishing user preference...")
    print("-" * 70)
    response1 = requests.post(f"{BASE_URL}/api/chat/query", json={
        "user_id": 1,
        "message": "I really prefer Python over JavaScript for my projects. I use it for data analysis.",
        "use_memory": True
    })

    if response1.status_code != 200:
        print(f"❌ Error: {response1.status_code}")
        print(response1.text)
        return

    result1 = response1.json()
    print(f"✅ Response: {result1['response'][:300]}...")
    print(f"📝 Session ID: {result1['session_id']}")
    print(f"🧠 Memory used: {result1['memory_used']}")
    session_id = result1['session_id']

    # Wait a bit for memory extraction to potentially occur
    print("\n⏳ Waiting 3 seconds for memory extraction...")
    time.sleep(3)

    # Second message - test if preference is remembered
    print("\n2️⃣  Testing memory recall...")
    print("-" * 70)
    response2 = requests.post(f"{BASE_URL}/api/chat/query", json={
        "user_id": 1,
        "message": "What programming language should I use for my next data science project?",
        "session_id": session_id,
        "use_memory": True
    })

    if response2.status_code != 200:
        print(f"❌ Error: {response2.status_code}")
        print(response2.text)
        return

    result2 = response2.json()
    print(f"✅ Response: {result2['response'][:300]}...")
    print(f"🧠 Memory used: {result2['memory_used']}")
    print(f"🤖 Model used: {result2['model_used']}")

    # Wait for memory consolidation
    print("\n⏳ Waiting 5 seconds for memory consolidation...")
    time.sleep(5)

    # Check memory profile
    print("\n3️⃣  Checking memory profile...")
    print("-" * 70)
    response3 = requests.get(f"{BASE_URL}/api/chat/memory/profile/1")

    if response3.status_code != 200:
        print(f"❌ Error: {response3.status_code}")
        print(response3.text)
        return

    profile = response3.json()

    print(f"📊 Semantic facts count: {profile['semantic_facts_count']}")
    print(f"📊 Recent episodes count: {profile['recent_episodes_count']}")

    if profile['semantic_facts_count'] > 0:
        print("\n🧠 Semantic facts:")
        for fact in profile['semantic_facts'][:5]:
            print(f"  • {fact['fact']} (category: {fact['category']}, confidence: {fact['confidence']})")
    else:
        print("\n⚠️  No semantic facts extracted yet (may take time)")

    if profile['recent_episodes_count'] > 0:
        print("\n📖 Recent episodes:")
        for ep in profile['recent_episodes'][:3]:
            summary = ep.get('summary', ep.get('content', ''))[:100]
            print(f"  • {summary}... (importance: {ep.get('importance', 'N/A')})")

    # Check queue status
    print("\n4️⃣  Checking consolidation queue status...")
    print("-" * 70)
    response4 = requests.get(f"{BASE_URL}/api/chat/queue/status")

    if response4.status_code != 200:
        print(f"❌ Error: {response4.status_code}")
        print(response4.text)
        return

    queue_status = response4.json()
    print(f"📋 Queue size: {queue_status['queue_size']}")
    print(f"📋 Queue status: {queue_status['status']}")

    # Check available models
    print("\n5️⃣  Checking available models...")
    print("-" * 70)
    response5 = requests.get(f"{BASE_URL}/api/chat/models")

    if response5.status_code != 200:
        print(f"❌ Error: {response5.status_code}")
        print(response5.text)
        return

    models = response5.json()
    print(f"🤖 Current assignments: {models['current_assignments']}")
    print(f"🤖 Recommended chat models: {[m['name'] for m in models['recommended_models']['chat']]}")

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)

def test_manual_fact_addition():
    """Test manually adding semantic facts"""
    print("\n" + "=" * 70)
    print("Manual Fact Addition Test")
    print("=" * 70)

    print("\n➕ Adding manual semantic fact...")
    response = requests.post(
        f"{BASE_URL}/api/chat/memory/semantic/add",
        params={
            "user_id": 1,
            "fact": "User has 5 years of experience in machine learning",
            "category": "context",
            "confidence": 1.0
        }
    )

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return

    result = response.json()
    print(f"✅ Added fact with ID: {result['fact_id']}")

    print("\n✅ Manual fact addition complete!")

def main():
    print("\n🚀 Starting memory-augmented chat tests...")
    print("\n⚠️  Make sure the server is running on http://localhost:8082")
    print("   Run: .venv/bin/python -m uvicorn app:app --reload --port 8082\n")

    input("Press Enter to continue...")

    try:
        # Test basic connectivity
        print("\n🔍 Testing server connectivity...")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is reachable")
        else:
            print(f"⚠️  Server returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the server is running!")
        return

    # Run tests
    test_chat()
    test_manual_fact_addition()

    print("\n\n🎉 All tests completed!")
    print("\nNext steps:")
    print("  • Check memory_system.log for detailed logs")
    print("  • Try more conversations to build up memory")
    print("  • Query /api/chat/memory/profile/1 to see extracted memories")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
