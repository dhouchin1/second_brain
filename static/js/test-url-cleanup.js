/**
 * Browser Console Test for URL Cleanup System
 * Run these tests in the browser console on /dashboard/v2
 */

// Test runner that can be executed in browser console
window.runUrlCleanupTests = function() {
    console.log('🧪 Starting URL Cleanup Tests...');

    const tests = [];
    let passed = 0;
    let failed = 0;

    function test(name, description, testFn) {
        tests.push({ name, description, testFn });
    }

    function assert(condition, message) {
        if (condition) {
            console.log(`✅ ${message}`);
            return true;
        } else {
            console.error(`❌ ${message}`);
            return false;
        }
    }

    // Test 1: State Manager exists and is initialized
    test('State Manager Initialization', 'Check if StateManager is available', () => {
        const exists = typeof window.stateManager !== 'undefined';
        const hasState = exists && typeof window.stateManager.getState === 'function';
        const state = hasState ? window.stateManager.getState() : null;

        return assert(exists && hasState && state,
            `StateManager available: ${exists}, has getState: ${hasState}, state keys: ${state ? Object.keys(state).length : 0}`);
    });

    // Test 2: Navigation Manager exists
    test('Navigation Manager Initialization', 'Check if NavigationManager is available', () => {
        const exists = typeof window.navigationManager !== 'undefined';
        const hasRoutes = exists && window.navigationManager.routes && window.navigationManager.routes.size > 0;

        return assert(exists && hasRoutes,
            `NavigationManager available: ${exists}, routes count: ${hasRoutes ? window.navigationManager.routes.size : 0}`);
    });

    // Test 3: Clean functions exist
    test('Clean Functions Available', 'Check if clean replacement functions exist', () => {
        const cleanFunctions = [
            'performSearchClean',
            'viewNoteClean',
            'navigateClean',
            'handleNoteSubmitClean'
        ];

        const available = cleanFunctions.filter(fn => typeof window[fn] === 'function');
        const allAvailable = available.length === cleanFunctions.length;

        return assert(allAvailable,
            `Clean functions available: ${available.length}/${cleanFunctions.length} [${available.join(', ')}]`);
    });

    // Test 4: URL Cleaning Function
    test('URL Cleaning', 'Test URL parameter removal', () => {
        const originalUrl = window.location.href;

        // Add test parameters
        const testUrl = new URL(window.location);
        testUrl.searchParams.set('test', 'value');
        testUrl.searchParams.set('q', 'search');
        testUrl.searchParams.set('id', '123');

        window.history.replaceState({}, document.title, testUrl.toString());
        const pollutedUrl = window.location.href;

        // Clean the URL
        if (window.stateManager && window.stateManager.cleanUrl) {
            window.stateManager.cleanUrl();
        }

        const cleanedUrl = window.location.href;
        const isClean = new URL(cleanedUrl).search === '';

        // Restore original URL
        window.history.replaceState({}, document.title, originalUrl);

        return assert(isClean,
            `URL cleaned successfully. Before: ${pollutedUrl}, After: ${cleanedUrl}`);
    });

    // Test 5: State Management
    test('State Operations', 'Test state setting and getting', () => {
        if (!window.stateManager) return assert(false, 'StateManager not available');

        const testData = {
            testKey: 'testValue',
            searchQuery: 'test-query-' + Date.now(),
            customObject: { nested: true, value: 42 }
        };

        window.stateManager.setState(testData);
        const retrievedState = window.stateManager.getState();

        const testKeyMatch = retrievedState.testKey === testData.testKey;
        const searchQueryMatch = retrievedState.searchQuery === testData.searchQuery;
        const objectMatch = retrievedState.customObject &&
                           retrievedState.customObject.nested === true &&
                           retrievedState.customObject.value === 42;

        return assert(testKeyMatch && searchQueryMatch && objectMatch,
            `State operations work. TestKey: ${testKeyMatch}, SearchQuery: ${searchQueryMatch}, Object: ${objectMatch}`);
    });

    // Test 6: Search without URL pollution
    test('Clean Search', 'Test search functionality without URL parameters', () => {
        if (!window.stateManager) return assert(false, 'StateManager not available');

        const originalUrl = window.location.href;
        const testQuery = 'test-search-' + Date.now();

        // Perform search using state management
        window.stateManager.setSearch(testQuery, { type: 'audio' });

        const state = window.stateManager.getState();
        const currentUrl = window.location.href;

        const queryStored = state.searchQuery === testQuery;
        const filtersStored = state.searchFilters && state.searchFilters.type === 'audio';
        const urlUnchanged = currentUrl === originalUrl;

        return assert(queryStored && filtersStored && urlUnchanged,
            `Search state managed correctly. Query: ${queryStored}, Filters: ${filtersStored}, URL unchanged: ${urlUnchanged}`);
    });

    // Test 7: Note Selection without URL exposure
    test('Clean Note Selection', 'Test note selection without URL parameters', () => {
        if (!window.stateManager) return assert(false, 'StateManager not available');

        const originalUrl = window.location.href;
        const testNote = {
            id: 'test-note-' + Date.now(),
            title: 'Test Note',
            content: 'Test content'
        };

        // Select note
        window.stateManager.selectNote(testNote);

        const state = window.stateManager.getState();
        const currentUrl = window.location.href;

        const noteSelected = state.selectedNote && state.selectedNote.id === testNote.id;
        const modalOpen = state.noteModalOpen === true;
        const urlUnchanged = currentUrl === originalUrl;

        // Close note
        window.stateManager.closeNote();
        const modalClosed = window.stateManager.getState('noteModalOpen') === false;

        return assert(noteSelected && modalOpen && urlUnchanged && modalClosed,
            `Note selection works. Selected: ${noteSelected}, Modal opened: ${modalOpen}, URL unchanged: ${urlUnchanged}, Modal closed: ${modalClosed}`);
    });

    // Test 8: Persistence
    test('State Persistence', 'Test localStorage persistence of appropriate data', () => {
        if (!window.stateManager) return assert(false, 'StateManager not available');

        const testHistory = ['search1', 'search2', 'search3'];
        const testPrefs = { theme: 'dark', notifications: true };

        window.stateManager.setState({
            searchHistory: testHistory,
            preferences: testPrefs
        });

        window.stateManager.savePersistentState();

        const savedHistory = localStorage.getItem('second_brain_searchHistory');
        const savedPrefs = localStorage.getItem('second_brain_preferences');

        const historyPersisted = savedHistory && JSON.parse(savedHistory).length === 3;
        const prefsPersisted = savedPrefs && JSON.parse(savedPrefs).theme === 'dark';

        // Cleanup test data
        localStorage.removeItem('second_brain_searchHistory');
        localStorage.removeItem('second_brain_preferences');

        return assert(historyPersisted && prefsPersisted,
            `Persistence works. History: ${historyPersisted}, Preferences: ${prefsPersisted}`);
    });

    // Run all tests
    console.log(`\n🚀 Running ${tests.length} tests...\n`);

    tests.forEach((testCase, index) => {
        console.log(`\n--- Test ${index + 1}: ${testCase.name} ---`);
        console.log(`Description: ${testCase.description}`);

        try {
            const result = testCase.testFn();
            if (result) {
                passed++;
            } else {
                failed++;
            }
        } catch (error) {
            console.error(`❌ Test failed with error: ${error.message}`);
            failed++;
        }
    });

    // Summary
    console.log(`\n📊 Test Summary:`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📈 Success Rate: ${Math.round((passed / tests.length) * 100)}%`);

    if (failed === 0) {
        console.log(`\n🎉 All tests passed! URL cleanup system is working correctly.`);
    } else {
        console.log(`\n⚠️  Some tests failed. Check the details above.`);
    }

    return { passed, failed, total: tests.length };
};

// Quick test function for manual verification
window.quickUrlTest = function() {
    console.log('🔍 Quick URL Test');

    const before = window.location.href;
    console.log('Before:', before);

    // Add some test parameters
    const url = new URL(window.location);
    url.searchParams.set('q', 'test-search');
    url.searchParams.set('id', '123');
    url.searchParams.set('action', 'edit');

    window.history.replaceState({}, document.title, url.toString());
    console.log('With params:', window.location.href);

    // Clean it
    if (window.stateManager && window.stateManager.cleanUrl) {
        window.stateManager.cleanUrl();
        console.log('After cleanup:', window.location.href);
        console.log('Is clean:', new URL(window.location).search === '');
    } else {
        console.log('❌ StateManager.cleanUrl not available');
    }

    // Restore
    window.history.replaceState({}, document.title, before);
    console.log('Restored:', window.location.href);
};

// Auto-run basic check when script loads
if (typeof window !== 'undefined') {
    console.log('🔧 URL Cleanup Test Suite loaded. Run runUrlCleanupTests() to test.');
}