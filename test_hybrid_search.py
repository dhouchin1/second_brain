#!/usr/bin/env python3
"""
Test hybrid search functionality with realistic data.
This script tests the integration of FTS5 keyword search and vector similarity search.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hybrid_search_with_real_data():
    """Test hybrid search with realistic note data."""
    print("🧪 Testing hybrid search with realistic data...")

    try:
        from services.search_adapter import SearchService

        # Use a temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        try:
            # Initialize search service
            search_service = SearchService(db_path=db_path)

            if not search_service.vec_available:
                print("❌ Vector search not available")
                return False

            # Add realistic test notes
            test_notes = [
                {
                    "title": "Machine Learning Fundamentals",
                    "body": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Key concepts include supervised learning, unsupervised learning, and reinforcement learning.",
                    "tags": "ml,ai,learning"
                },
                {
                    "title": "Python Programming Tips",
                    "body": "Python is a versatile programming language. Here are some tips: use list comprehensions for cleaner code, leverage virtual environments for dependency management, and follow PEP 8 style guidelines.",
                    "tags": "python,programming,tips"
                },
                {
                    "title": "Database Design Principles",
                    "body": "Good database design involves normalization, proper indexing, and understanding relationships between tables. SQLite is great for lightweight applications while PostgreSQL handles complex enterprise needs.",
                    "tags": "database,sql,design"
                },
                {
                    "title": "Neural Networks Introduction",
                    "body": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions.",
                    "tags": "ai,neural,networks"
                },
                {
                    "title": "Web Development Best Practices",
                    "body": "Modern web development involves responsive design, security considerations, performance optimization, and accessibility. Frameworks like FastAPI and React help streamline development.",
                    "tags": "web,development,best-practices"
                }
            ]

            # Insert test notes
            note_ids = []
            for note in test_notes:
                note_id = search_service.upsert_note(
                    None,
                    note["title"],
                    note["body"],
                    note["tags"]
                )
                note_ids.append(note_id)
                print(f"✅ Added note: {note['title']} (ID: {note_id})")

            print(f"\n📊 Added {len(note_ids)} test notes to database")

            # Test different search modes
            search_queries = [
                "machine learning",
                "python programming",
                "database SQLite",
                "neural networks AI",
                "web development"
            ]

            print(f"\n🔍 Testing search modes with {len(search_queries)} queries...\n")

            for query in search_queries:
                print(f"Query: '{query}'")
                print("-" * 40)

                # Keyword search
                kw_results = search_service.search(query, mode='keyword', k=3)
                print(f"  Keyword ({len(kw_results)} results):")
                for i, result in enumerate(kw_results[:2], 1):
                    score = result['kw_rank'] if 'kw_rank' in result.keys() else 'N/A'
                    print(f"    {i}. {result['title']} (score: {score})")

                # Semantic search
                sem_results = search_service.search(query, mode='semantic', k=3)
                print(f"  Semantic ({len(sem_results)} results):")
                for i, result in enumerate(sem_results[:2], 1):
                    score = result['score'] if 'score' in result.keys() else 'N/A'
                    if isinstance(score, (int, float)):
                        print(f"    {i}. {result['title']} (score: {score:.3f})")
                    else:
                        print(f"    {i}. {result['title']} (score: {score})")

                # Hybrid search
                hybrid_results = search_service.search(query, mode='hybrid', k=3)
                print(f"  Hybrid ({len(hybrid_results)} results):")
                for i, result in enumerate(hybrid_results[:2], 1):
                    score = result['score'] if 'score' in result.keys() else 'N/A'
                    if isinstance(score, (int, float)):
                        print(f"    {i}. {result['title']} (score: {score:.3f})")
                    else:
                        print(f"    {i}. {result['title']} (score: {score})")

                print()

            # Test edge cases
            print("🧪 Testing edge cases...")

            # Test with special characters
            special_results = search_service.search("AI & ML", mode='hybrid', k=3)
            print(f"✅ Special characters: {len(special_results)} results")

            # Test with short query
            short_results = search_service.search("AI", mode='hybrid', k=3)
            print(f"✅ Short query: {len(short_results)} results")

            # Test with long query
            long_query = "artificial intelligence machine learning neural networks deep learning"
            long_results = search_service.search(long_query, mode='hybrid', k=3)
            print(f"✅ Long query: {len(long_results)} results")

            # Test with no matches
            no_match_results = search_service.search("blockchain cryptocurrency", mode='hybrid', k=3)
            print(f"✅ No match query: {len(no_match_results)} results")

            return True

        finally:
            # Clean up temporary database
            try:
                os.unlink(db_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_performance():
    """Test search performance with vector and hybrid modes."""
    print("\n🧪 Testing search performance...")

    try:
        import time
        from services.search_adapter import SearchService

        # Use a temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        try:
            search_service = SearchService(db_path=db_path)

            if not search_service.vec_available:
                print("❌ Vector search not available for performance test")
                return False

            # Add more notes for performance testing
            for i in range(20):
                search_service.upsert_note(
                    None,
                    f"Test Note {i}",
                    f"This is test note number {i} with various keywords like machine learning, artificial intelligence, python programming, and database design. Content varies to test search performance.",
                    f"test,note{i},performance"
                )

            print("✅ Added 20 test notes for performance testing")

            # Test search performance
            query = "machine learning artificial intelligence"
            modes = ['keyword', 'semantic', 'hybrid']

            print("\nSearch performance (3 runs each):")
            for mode in modes:
                times = []
                for _ in range(3):
                    start_time = time.time()
                    results = search_service.search(query, mode=mode, k=10)
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                print(f"  {mode:>8}: {avg_time:.4f}s avg ({len(results)} results)")

            return True

        finally:
            try:
                os.unlink(db_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run hybrid search tests."""
    print("🚀 Starting hybrid search functionality tests...\n")

    tests = [
        ("Hybrid search with realistic data", test_hybrid_search_with_real_data),
        ("Search performance testing", test_search_performance),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "="*60)
    print("📊 HYBRID SEARCH TEST RESULTS")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All hybrid search tests passed! Vector + FTS5 integration is working perfectly.")
    else:
        print("⚠️  Some hybrid search tests failed. Check the output above for details.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)