const chatBox = document.getElementById('chat-box');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

function setQuery(text) {
    queryInput.value = text;
    handleSend();
}

queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});

sendBtn.addEventListener('click', handleSend);

async function handleSend() {
    const text = queryInput.value.trim();
    if (!text) return;

    // Add user message
    appendMessage(text, 'user');
    queryInput.value = '';

    // Add typing indicator map
    const aiMessageDiv = document.createElement('div');
    aiMessageDiv.className = 'message ai';
    
    aiMessageDiv.innerHTML = `
        <div class="avatar-bot"></div>
        <div class="msg-content" style="min-width: 300px;">
            <div class="typing-indicator" id="typing-${Date.now()}">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <div class="agent-steps" style="display:none;"></div>
            <div class="final-answer" style="display:none; margin-top: 15px;"></div>
        </div>
    `;
    chatBox.appendChild(aiMessageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    const stepsContainer = aiMessageDiv.querySelector('.agent-steps');
    const answerContainer = aiMessageDiv.querySelector('.final-answer');
    const typingInd = aiMessageDiv.querySelector('.typing-indicator');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        
        const data = await response.json();
        
        // Hide typing, show steps container
        typingInd.style.display = 'none';
        stepsContainer.style.display = 'flex';

        // Animate agent steps one by one
        for (let i = 0; i < data.steps.length; i++) {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'agent-step';
            stepDiv.innerHTML = `<div class="step-dot"></div> ${data.steps[i]}`;
            stepsContainer.appendChild(stepDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Artificial delay to make it look like it's thinking agentically
            await new Promise(r => setTimeout(r, 600));
        }

        // Render final markdown answer
        answerContainer.style.display = 'block';
        
        // Highlighting citations
        let parsedAnswer = marked.parse(data.answer);
        parsedAnswer = parsedAnswer.replace(/\[Document: (.*?), Page: (.*?)\]/g, 
            '<span style="background: rgba(139, 92, 246, 0.2); color: #c4b5fd; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; border: 1px solid rgba(139, 92, 246, 0.4);">📄 $1, Pg $2</span>');

        answerContainer.innerHTML = parsedAnswer;
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        typingInd.style.display = 'none';
        answerContainer.style.display = 'block';
        answerContainer.innerHTML = `<p style="color: #ef4444;">Connection error to API. Please ensure FastAPI is running.</p>`;
    }
}

function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    const avatar = sender === 'user' ? '' : '<div class="avatar-bot"></div>';
    
    div.innerHTML = `
        ${avatar}
        <div class="msg-content">
            <p>${text}</p>
        </div>
    `;
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}
