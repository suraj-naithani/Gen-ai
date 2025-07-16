require('dotenv').config();
const { OpenAI } = require('openai');
const fs = require("fs");
const readline = require("readline");

// Create OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// 1. Load notes.txt
const rawText = fs.readFileSync("data.txt", "utf8");
const chunks = rawText.split("\n").filter(Boolean);

// 2. Embed chunks using OpenAI embeddings
async function embedChunks(textArray) {
    const vectors = [];

    for (const text of textArray) {
        const embedding = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: text
        });
        vectors.push({
            text,
            embedding: embedding.data[0].embedding,
        })
    }
    return vectors;
}

// 3. Cosine similarity function
function cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (normA * normB);
}

// 4. Find most similar chunk
function findMostRelevantChunk(questionEmbedding, embeddedChunks) {
    let bestScore = -1;
    let bestChunk = "";

    for (const chunk of embeddedChunks) {
        const score = cosineSimilarity(questionEmbedding, chunk.embedding);
        if (score > bestScore) {
            bestScore = score;
            bestChunk = chunk.text;
        }
    }

    return bestChunk;
}

// 5. Ask question in loop
async function main() {
    const embeddedChunks = await embedChunks(chunks);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const askQuestion = async () => {
        rl.question("Ask something: ", async (question) => {
            if (question.toLowerCase() === "exit") {
                rl.close();
                return;
            }

            const questionEmbedding = await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: question,
            });

            const relevantChunk = findMostRelevantChunk(
                questionEmbedding.data[0].embedding,
                embeddedChunks
            );

            const prompt = `Context: ${relevantChunk}\n\nAnswer the question: ${question}`;

            const response = await openai.chat.completions.create({
                model: "gpt-3.5-turbo",
                messages: [{ role: "user", content: prompt }],
            });

            console.log("Answer:", response.choices[0].message.content);
            askQuestion(); // loop again
        });
    };

    askQuestion();
}

main();