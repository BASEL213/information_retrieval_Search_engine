// DOM Elements
const welcomePage = document.getElementById('welcome-page');
const searchPage = document.getElementById('search-page');
const startSearchButton = document.getElementById('start-search');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const resultsContainer = document.getElementById('results-container');
const suggestionsContainer = document.getElementById('suggestions-container');
const themeToggle = document.getElementById('theme-toggle');
const modelSelect = document.getElementById('model-select');

// Backend API URL
const API_URL = '/search';

// Transition to search page
startSearchButton.addEventListener('click', () => {
    welcomePage.style.display = 'none';
    searchPage.style.display = 'flex';
    searchInput.focus();
});

// Search Functionality
async function performSearch(query, model = 'bm25') {
    resultsContainer.innerHTML = '<div class="spinner"></div>';
    suggestionsContainer.innerHTML = ''; // Clear suggestions at the start

    if (query.trim() === '') {
        resultsContainer.innerHTML = '<p class="text-gray-400 text-center">Please enter a search query.</p>';
        return;
    }

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, model })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Display results
        resultsContainer.innerHTML = '';
        if (data.results.length === 0) {
            resultsContainer.innerHTML = '<p class="text-gray-400 text-center">No results found.</p>';
            return;
        }

        data.results.forEach(result => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card p-4 rounded-lg shadow-md';
            resultCard.innerHTML = `
                <h2 class="text-lg font-semibold text-blue-400">
                    <a href="${result.url}" target="_blank" class="hover:underline">${result.title}</a>
                </h2>
                <p class="text-sm text-gray-400">${result.url}</p>
                <p class="text-gray-200">${result.description}</p>
            `;
            resultsContainer.appendChild(resultCard);
        });

        // Do not show suggestions after search is performed
    } catch (error) {
        console.error('Error:', error);
        resultsContainer.innerHTML = '<p class="text-red-400 text-center">An error occurred while fetching results.</p>';
    }
}

// Query Expansion (Suggestions)
function showSuggestions(suggestions, originalQuery) {
    suggestionsContainer.innerHTML = '';
    if (!suggestions || suggestions.length === 0) return;

    suggestions.forEach(suggestion => {
        const suggestionItem = document.createElement('div');
        suggestionItem.className = 'suggestion-item p-2 rounded-lg shadow-sm text-gray-200';
        suggestionItem.tabIndex = 0;
        suggestionItem.innerHTML = suggestion;
        suggestionItem.addEventListener('click', () => {
            searchInput.value = suggestion;
            suggestionsContainer.innerHTML = ''; // Clear suggestions on selection
            performSearch(suggestion, modelSelect.value);
        });
        suggestionItem.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchInput.value = suggestion;
                suggestionsContainer.innerHTML = ''; // Clear suggestions on selection
                performSearch(suggestion, modelSelect.value);
            }
        });
        suggestionsContainer.appendChild(suggestionItem);
    });

    // Apply fade-in animation to suggestions
    suggestionsContainer.querySelectorAll('.suggestion-item').forEach((item, index) => {
        item.style.animation = `fadeIn 0.3s ease-in ${index * 0.1}s both`;
    });
}

// Event Listeners
searchButton.addEventListener('click', () => {
    suggestionsContainer.innerHTML = ''; // Clear suggestions on search button click
    performSearch(searchInput.value, modelSelect.value);
});
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        suggestionsContainer.innerHTML = ''; // Clear suggestions on Enter
        performSearch(searchInput.value, modelSelect.value);
    }
});
searchInput.addEventListener('input', async () => {
    const query = searchInput.value;
    if (query.trim() === '') {
        suggestionsContainer.innerHTML = '';
        return;
    }
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, model: modelSelect.value })
        });
        const data = await response.json();
        showSuggestions(data.suggestions, query);
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
});

// Update search results when model changes
modelSelect.addEventListener('change', () => {
    const query = searchInput.value;
    if (query.trim() !== '') {
        suggestionsContainer.innerHTML = ''; // Clear suggestions on model change
        performSearch(query, modelSelect.value);
    }
});

// Theme Toggle (Dark to Light)
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    themeToggle.innerHTML = `<i class="fas ${isLight ? 'fa-moon' : 'fa-sun'}"></i>`;
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
});

// Load saved theme
if (localStorage.getItem('theme') === 'light') {
    document.body.classList.add('light-mode');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
}