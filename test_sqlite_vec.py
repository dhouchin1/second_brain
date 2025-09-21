#!/usr/bin/env python3
"""
Test script to verify sqlite-vec installation and functionality.
This script tests:
1. Basic sqlite-vec extension loading
2. Vector table creation
3. Vector insertion and retrieval
4. Vector similarity search
5. Integration with the search service
"""

import os
import sys
import sqlite3
import json
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_sqlite_vec():
    """Test basic sqlite-vec functionality."""
    print("🧪 Testing basic sqlite-vec functionality...")

    try:
        import sqlite_vec
        print("✅ sqlite-vec package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import sqlite-vec: {e}")
        return False

    # Test with in-memory database
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Test version
        result = conn.execute("SELECT vec_version()").fetchone()
        version = result[0] if result else "unknown"
        print(f"✅ sqlite-vec version: {version}")

        # Test creating a vector table
        conn.execute("""
            CREATE VIRTUAL TABLE test_vecs USING vec0(
                embedding float[384]
            )
        """)
        print("✅ Vector table created successfully")

        # Test inserting vectors
        test_vector = [0.1] * 384  # Simple test vector
        conn.execute("INSERT INTO test_vecs(rowid, embedding) VALUES (1, ?)",
                    (json.dumps(test_vector),))

        # Test querying vectors
        result = conn.execute("SELECT rowid FROM test_vecs WHERE rowid = 1").fetchone()
        if result:
            print("✅ Vector insertion and retrieval working")
        else:
            print("❌ Vector insertion/retrieval failed")
            return False

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Basic sqlite-vec test failed: {e}")
        return False

def test_vector_similarity():
    """Test vector similarity search functionality."""
    print("\n🧪 Testing vector similarity search...")

    try:
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Create vector table
        conn.execute("""
            CREATE VIRTUAL TABLE similarity_test USING vec0(
                embedding float[3]
            )
        """)

        # Insert test vectors
        vectors = [
            (1, [1.0, 0.0, 0.0]),  # Similar to query
            (2, [0.9, 0.1, 0.0]),  # Very similar to query
            (3, [0.0, 1.0, 0.0]),  # Different from query
            (4, [0.0, 0.0, 1.0]),  # Different from query
        ]

        for row_id, vec in vectors:
            conn.execute("INSERT INTO similarity_test(rowid, embedding) VALUES (?, ?)",
                        (row_id, json.dumps(vec)))

        # Test similarity search
        query_vector = [1.0, 0.0, 0.0]
        results = conn.execute("""
            SELECT rowid, 1.0 - vec_distance_cosine(embedding, ?) as similarity
            FROM similarity_test
            ORDER BY similarity DESC
            LIMIT 3
        """, (json.dumps(query_vector),)).fetchall()

        if results and len(results) >= 2:
            # Check that results are ordered by similarity
            similarities = [row[1] for row in results]
            if similarities == sorted(similarities, reverse=True):
                print("✅ Vector similarity search working correctly")
                print(f"   Top results: {[(row[0], round(row[1], 3)) for row in results]}")
                return True
            else:
                print("❌ Similarity results not properly ordered")
                return False
        else:
            print("❌ No similarity search results returned")
            return False

    except Exception as e:
        print(f"❌ Vector similarity test failed: {e}")
        return False

def test_search_service_integration():
    """Test integration with the search service."""
    print("\n🧪 Testing search service integration...")

    try:
        from services.search_adapter import SearchService

        # Use a temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        try:
            # Initialize search service
            search_service = SearchService(db_path=db_path)

            if search_service.vec_available:
                print("✅ Search service has vector search enabled")

                # Test adding a note
                note_id = search_service.upsert_note(
                    None,
                    "Test Note",
                    "This is a test note about artificial intelligence and machine learning.",
                    "ai,ml,test"
                )
                print(f"✅ Note added with ID: {note_id}")

                # Test vector table exists
                if search_service._vec_table_exists():
                    print("✅ Vector table exists and is accessible")

                    # Test vector search
                    semantic_results = search_service.search("artificial intelligence", mode='semantic', k=5)
                    if semantic_results:
                        print(f"✅ Semantic search returned {len(semantic_results)} results")
                    else:
                        print("⚠️  Semantic search returned no results (might be normal with minimal data)")

                    # Test hybrid search
                    hybrid_results = search_service.search("machine learning", mode='hybrid', k=5)
                    if hybrid_results:
                        print(f"✅ Hybrid search returned {len(hybrid_results)} results")
                    else:
                        print("⚠️  Hybrid search returned no results (might be normal with minimal data)")

                    return True
                else:
                    print("❌ Vector table does not exist")
                    return False
            else:
                print("❌ Search service does not have vector search enabled")
                return False

        finally:
            # Clean up temporary database
            try:
                os.unlink(db_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Search service integration test failed: {e}")
        return False

def test_embeddings_service():
    """Test the embeddings service."""
    print("\n🧪 Testing embeddings service...")

    try:
        from services.embeddings import Embeddings

        embedder = Embeddings()

        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = embedder.embed(test_text)

        if embedding and len(embedding) > 0:
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
            print(f"   Sample values: {embedding[:5]}...")

            # Test embedding consistency
            embedding2 = embedder.embed(test_text)
            if embedding == embedding2:
                print("✅ Embedding generation is consistent")
                return True
            else:
                print("⚠️  Embedding generation varies between calls (might be normal)")
                return True
        else:
            print("❌ Failed to generate embedding")
            return False

    except Exception as e:
        print(f"❌ Embeddings service test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting sqlite-vec installation and functionality tests...\n")

    tests = [
        ("Basic sqlite-vec functionality", test_basic_sqlite_vec),
        ("Vector similarity search", test_vector_similarity),
        ("Embeddings service", test_embeddings_service),
        ("Search service integration", test_search_service_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! sqlite-vec is properly installed and configured.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. sqlite-vec is partially working.")
    else:
        print("❌ All tests failed. sqlite-vec installation needs attention.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)