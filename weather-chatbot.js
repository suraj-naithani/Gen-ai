require('dotenv').config();
const { OpenAI } = require('openai');
const { tavily } = require('@tavily/core');
const rl = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

// Initialize OpenAI and Tavily clients
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const client = tavily({ apiKey: process.env.TAVILY_API_KEY });

// In-memory conversation history as array of {user, ai} objects
let conversationHistory = [];

// Define tools
const tools = [
    {
        type: 'function',
        function: {
            name: 'getWeather',
            description: 'Get current weather information for a city',
            parameters: {
                type: 'object',
                properties: {
                    city: {
                        type: 'string',
                        description: 'Name of the city to get the weather for'
                    }
                },
                required: ['city']
            }
        }
    },
    {
        type: 'function',
        function: {
            name: 'summarize',
            description: 'Summarize the given text into a short summary',
            parameters: {
                type: 'object',
                properties: {
                    text: {
                        type: 'string',
                        description: 'Text to summarize'
                    }
                },
                required: ['text']
            }
        }
    }
];

async function getWeather(city) {
    try {
        const response = await client.search(`current weather in ${city}`);
        const data = response.results[0];
        return {
            city: city,
            content: data.content,
            source: data.url || 'Unknown source',
        };
    } catch (error) {
        return { error: error.message || 'Failed to fetch weather data' };
    }
}

async function summarize(text) {
    try {
        const completion = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: [
                { role: 'system', content: 'You summarize long text into a short sentence.' },
                { role: 'user', content: `Summarize this: ${text}` }
            ]
        });
        return completion.choices[0].message.content.trim();
    } catch (error) {
        return `Error summarizing text: ${error.message}`;
    }
}

async function chatWithOpenAI(userInput) {
    try {
        // Prepare messages with conversation history
        const messages = [
            {
                role: 'system',
                content: 'You are a helpful assistant that can provide weather information and summarize text. Use the conversation history to maintain context and provide relevant responses based on previous interactions.'
            },
            // Include previous conversation history
            ...conversationHistory.map(chat => [
                { role: 'user', content: chat.user },
                { role: 'assistant', content: chat.ai }
            ]).flat(),
            // Add current user input
            { role: 'user', content: userInput }
        ];

        const response = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages,
            tools,
            tool_choice: 'auto',
        });

        const message = response.choices[0].message;

        let reply;

        if (message.tool_calls && message.tool_calls.length > 0) {
            const toolCall = message.tool_calls[0];

            messages.push({
                role: 'assistant',
                content: null,
                tool_calls: message.tool_calls,
            });

            if (toolCall.function.name === 'getWeather') {
                const args = JSON.parse(toolCall.function.arguments);
                const weatherData = await getWeather(args.city);
                const fixedContent = weatherData.content.replace(/'/g, '"');
                let functionResponse;
                try {
                    const data = JSON.parse(fixedContent);
                    if (data.error) {
                        functionResponse = `Error fetching weather data: ${data.error}`;
                    } else {
                        functionResponse = `Weather in ${data.location?.name}: ${data.current.temp_c}°C, ${data.current.condition.text}, humidity: ${data.current.humidity}%, wind speed: ${data.current.wind_kph} kph`;
                    }
                } catch (parseError) {
                    functionResponse = `Error parsing weather data: ${parseError.message}`;
                }

                messages.push({
                    role: 'tool',
                    content: JSON.stringify({ result: functionResponse }),
                    tool_call_id: toolCall.id
                });

                const finalResponse = await openai.chat.completions.create({
                    model: 'gpt-3.5-turbo',
                    messages,
                });

                reply = finalResponse.choices[0].message.content.trim();
            } else if (toolCall.function.name === 'summarize') {
                const args = JSON.parse(toolCall.function.arguments);
                const summary = await summarize(args.text);

                messages.push({
                    role: 'tool',
                    content: JSON.stringify({ result: summary }),
                    tool_call_id: toolCall.id
                });

                const finalResponse = await openai.chat.completions.create({
                    model: 'gpt-3.5-turbo',
                    messages,
                });

                reply = finalResponse.choices[0].message.content.trim();
            }
        } else {
            reply = message.content.trim();
        }

        // Store the conversation in memory
        conversationHistory.push({ user: userInput, ai: reply });

        console.log(`Chatbot: ${reply}`);
        return reply;
    } catch (error) {
        console.error("Error communicating with OpenAI:", error);
        const errorReply = "Sorry, I encountered an error. Please try again.";
        conversationHistory.push({ user: userInput, ai: errorReply });
        console.log(`Chatbot: ${errorReply}`);
        return errorReply;
    }
}

function askQuestion() {
    rl.question('You: ', async (userInput) => {
        if (userInput.toLowerCase() === 'exit') {
            rl.close();
            console.log("Chatbot stopped.");
            return;
        }

        await chatWithOpenAI(userInput);
        askQuestion();
    });
}

console.log('Chatbot started! Type your message (or "exit" to quit):');
askQuestion();