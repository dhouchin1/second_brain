// Second Brain Premium - Main JavaScript
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('notes-container');
    if (!container) return;

    fetch('/notes')
        .then(res => res.json())
        .then(notes => {
            if (!notes.length) {
                container.innerHTML = '<p>No notes yet. Start capturing your thoughts!</p>';
                return;
            }

            notes.forEach(note => {
                const card = document.createElement('div');
                card.className = 'note-card border-b border-gray-200 py-5 last:border-b-0';
                card.innerHTML = `
                    <h3 class="text-lg text-gray-800 mb-2">${note.title}</h3>
                    <p class="text-gray-600 mb-2">${note.summary}</p>
                    <span class="text-indigo-600 text-sm">${note.tags}</span>
                `;
                container.appendChild(card);
                requestAnimationFrame(() => card.classList.add('show'));
            });
        })
        .catch(() => {
            container.innerHTML = '<p>Error loading notes.</p>';
        });
});
