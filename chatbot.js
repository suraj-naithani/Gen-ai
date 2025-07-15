require('dotenv').config();
const { OpenAI } = require('openai');

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
})

const rl = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

const chatWithOpenAI = async (userInput) => {
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: userInput }
            ]
        })
        console.log(response)
        const reply = response.choices[0].message.content.trim();
        console.log(`AI: ${reply}`);
    } catch (error) {
        console.error('Error communicating with OpenAI:', error);
    }
}

function askQuestion() {
    rl.question('You: ', async (userInput) => {
        if (userInput.toLowerCase() === 'exit') {
            rl.close();
            return;
        }

        await chatWithOpenAI(userInput);
        askQuestion();
    })
}

askQuestion();