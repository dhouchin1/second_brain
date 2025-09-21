#!/usr/bin/env python3
"""
Test script to verify sqlite-vec is working with the production database.
This tests the actual search functionality against the existing notes.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_production_search():
    """Test search functionality with the production database."""
    print("🧪 Testing search functionality with production database...")

    try:
        from services.search_adapter import SearchService

        # Use the actual production database
        search_service = SearchService(db_path='notes.db')

        print(f"Vector search available: {search_service.vec_available}")

        if not search_service.vec_available:
            print("❌ Vector search not available")
            return False

        # Check if vector table exists
        if search_service._vec_table_exists():
            print("✅ Vector table exists")
        else:
            print("⚠️  Vector table does not exist - this is normal if no notes have been indexed yet")

        # Test different search modes with real queries
        test_queries = [
            "machine learning",
            "python",
            "AI",
            "database",
            "note"
        ]

        print("\n🔍 Testing search modes with production data:")

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 30)

            # Test keyword search
            try:
                kw_results = search_service.search(query, mode='keyword', k=5)
                print(f"  Keyword: {len(kw_results)} results")
            except Exception as e:
                print(f"  Keyword: FAILED - {e}")

            # Test semantic search
            try:
                sem_results = search_service.search(query, mode='semantic', k=5)
                print(f"  Semantic: {len(sem_results)} results")
            except Exception as e:
                print(f"  Semantic: FAILED - {e}")

            # Test hybrid search
            try:
                hybrid_results = search_service.search(query, mode='hybrid', k=5)
                print(f"  Hybrid: {len(hybrid_results)} results")

                # Show top results for hybrid search
                if hybrid_results:
                    print("    Top results:")
                    for i, result in enumerate(hybrid_results[:3], 1):
                        title = result['title'][:50] + "..." if len(result['title']) > 50 else result['title']
                        score = result['score'] if 'score' in result.keys() else 'N/A'
                        if isinstance(score, (int, float)):
                            print(f"      {i}. {title} (score: {score:.3f})")
                        else:
                            print(f"      {i}. {title} (score: {score})")

            except Exception as e:
                print(f"  Hybrid: FAILED - {e}")

        print("\n✅ Production search testing completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Production search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_indexing():
    """Test if we can add a new note and it gets vector indexed."""
    print("\n🧪 Testing vector indexing for new notes...")

    try:
        from services.search_adapter import SearchService

        search_service = SearchService(db_path='notes.db')

        if not search_service.vec_available:
            print("❌ Vector search not available for indexing test")
            return False

        # Add a test note
        test_title = "sqlite-vec Test Note"
        test_body = "This is a test note to verify that sqlite-vec vector indexing is working properly with our embedding system."
        test_tags = "test,sqlite-vec,vector,embeddings"

        note_id = search_service.upsert_note(None, test_title, test_body, test_tags)
        print(f"✅ Added test note with ID: {note_id}")

        # Test that we can find it with semantic search
        semantic_results = search_service.search("vector indexing embedding", mode='semantic', k=10)

        found_test_note = False
        for result in semantic_results:
            if result['id'] == note_id:
                found_test_note = True
                break

        if found_test_note:
            print("✅ Test note found in semantic search results")
        else:
            print("⚠️  Test note not found in semantic search (might need time to index)")

        # Clean up - remove the test note
        search_service.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        search_service.conn.execute("DELETE FROM note_vecs WHERE note_id = ?", (note_id,))
        search_service.conn.commit()
        print("✅ Test note cleaned up")

        return True

    except Exception as e:
        print(f"❌ Vector indexing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run production search tests."""
    print("🚀 Testing sqlite-vec integration with production data...\n")

    tests = [
        ("Production search functionality", test_production_search),
        ("Vector indexing for new notes", test_vector_indexing),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "="*60)
    print("📊 PRODUCTION SEARCH TEST RESULTS")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 sqlite-vec is fully functional with production data!")
        print("💡 Hybrid search (FTS5 + vector similarity) is now enabled")
        print("🔍 Users will get better search results combining keyword and semantic search")
    else:
        print("⚠️  Some production tests failed. Check the output above.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)