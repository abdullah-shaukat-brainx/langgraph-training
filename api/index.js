const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
require('dotenv').config();

const { setupLangGraph } = require('./langGraph');

const app = express();
const port = process.env.PORT || 5002;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, '../public')));

// Serve index.html for the root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Initialize LangGraph
const graph = setupLangGraph();

// POST endpoint for chat
app.post('/api/chat', async (req, res) => {
    try {
        const { message, threadId } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        const inputs = { 
            messages: [{ role: "human", content: message }] 
        };
        
        const config = { 
            configurable: { thread_id: threadId || Date.now().toString() }
        };

        const responses = [];
        const stream = await graph.stream(inputs, config);

        for await (const event of stream) {
            // Handle the nested structure
            const agentName = Object.keys(event)[0];
            if (event[agentName] && event[agentName].messages) {
                const messages = event[agentName].messages;
                messages.forEach(msg => {
                    responses.push({
                        type: msg.role || 'ai',
                        name: agentName || 'Bot',
                        content: msg.content
                    });
                });
            }
        }

        res.json({ responses });

    } catch (error) {
        console.error('Error processing chat:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
}); 