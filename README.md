# Generative AI

## Table of Contents

- **🔍 What is Generative AI?**  
  - [💡 Simple Definition](#simple-definition)  
  - [🤖 Real-Life Analogy](#real-life-analogy)  
  - [🧠 How Does It Work?](#how-does-it-work)  
  - [🎨 Examples of What Generative AI Can Do](#examples-of-what-generative-ai-can-do)  
  - [🧑‍💻 Example (Text)](#example-text)  
  - [🔁 Types of GenAI Models](#types-of-genai-models)  
  - [🤔 Where is it Used Today?](#where-is-it-used-today)  
  - [🧪 Quick Fun Exercise](#quick-fun-exercise)  
  - [✅ Summary](#summary)  

- **🔑 Important Keywords in GenAI**  
  - [1️⃣ Token](#token)  
  - [2️⃣ Prompt](#prompt)  
  - [3️⃣ Completion / Output](#completion-output)  
  - [4️⃣ Temperature](#temperature)  
  - [5️⃣ Top-k Sampling](#top-k-sampling)  
  - [6️⃣ Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)  
  - [7️⃣ Max Tokens](#max-tokens)  
  - [8️⃣ Stop Sequence](#stop-sequence)  
  - [9️⃣ Fine-tuning](#fine-tuning)  
  - [🔟 Embeddings](#embeddings)  
  - [1️⃣1️⃣ RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)  
  - [1️⃣2️⃣ Context Window](#context-window)  
  - [1️⃣3️⃣ Memory (Long-Term Brain)](#memory-long-term-brain)  
  - [1️⃣5️⃣ Context (Short-Term Brain)](#context-short-term-brain)  
  - [1️⃣6️⃣ Looping (Function Calling / Tool Use)](#looping-function-calling-tool-use)  
  - [🧠 Zero-shot Learning](#zero-shot-learning)  
  - [🧠 Few-shot Learning](#few-shot-learning)  
  - [🧠 Chain-of-Thought (CoT) Reasoning](#chain-of-thought-cot-reasoning)  
  - [🧠 LLM (Large Language Model)](#llm-large-language-model)  

- **🤖 What is a Chatbot?**  
  - [🛠️ What We’ll Build First](#what-well-build-first)  

- **📡 What is Streaming in OpenAI?**  
  - [What is Streaming in OpenAI?](#what-is-streaming-in-openai)  

- **🤖 Function Calling**  
  - [📖 Chapter 1: What Is Function Calling?](#chapter-1-what-is-function-calling)  
  - [📖 Chapter 2: Why Use Function Calling?](#chapter-2-why-use-function-calling)  
  - [📖 Chapter 3: The 3 Players](#chapter-3-the-3-players)  
  - [📖 Chapter 4: A Simple Example](#chapter-4-a-simple-example)  
  - [📖 Chapter 5: Handling the Function](#chapter-5-handling-the-function)  
  - [📖 Chapter 6: Summary (TL;DR)](#chapter-6-summary-tldr)  
  - [📖 Chapter 7: Real-Life Use Cases](#chapter-7-real-life-use-cases)  
  - [📖 Chapter 8: Example](#chapter-8-example)  

- **🤖 RAG (Retrieval-Augmented Generation)**  
  - [🔍 What is RAG?](#what-is-rag)
  - [🧱 Key Components of RAG](#key-components-of-rag)
  - [🔍 Understanding the Start of the RAG Process](#understanding-the-start-of-the-rag-process)
  - [📚 Step-by-Step Theory (with Examples)](#step-by-step-theory-with-examples)
    - [Step 1: Prepare Your Data](#step-1-prepare-your-data-docs-faqs-etc)
    - [Step 2: Chunking 🍕](#step-2-chunking)
    - [Step 3: Embedding 📐](#step-3-embedding)
    - [Step 4: Store in Vector DB 🧠](#step-4-store-in-vector-db)
    - [Step 5: Ask a Question 🗣️](#step-5-ask-a-question)
    - [Step 6: Generate Answer using OpenAI 💡](#step-6-generate-answer-using-openai)
  - [🔁 Flow Diagram with Arrows](#flow-diagram-with-arrows)
  - [RAQ Q&A Flow](#raq-qa-flow)
  - [RAG Example](#rag-example)
 

## 🔍 What is Generative AI? <a id="what-is-generative-ai"></a>

### 💡 Simple Definition <a id="simple-definition"></a>
**Generative AI** is a type of **artificial intelligence** that can create new content — like text, images, videos, code, music, etc. — just like humans do.

Instead of just analyzing or recognizing data, it actually generates something new from what it has learned.

### 🤖 Real-Life Analogy  <a id="real-life-analogy"></a>
Imagine a super smart robot that has read millions of books and can now:

- Write a new story
- Answer your questions like a teacher
- Draw pictures like an artist
- Compose music like a musician
- Write code like a developer

That robot is Generative AI.

### 🧠 How Does It Work? <a id="how-does-it-work"></a>
Generative AI learns patterns from vast amounts of data, including:

- Text (books, websites, articles)
- Images (cats, cars, mountains)
- Videos, code, audio, and more

When given a prompt, such as a question or task, it uses these learned patterns to generate new, original content.

### 🎨 Examples of What Generative AI Can Do <a id="examples-of-what-generative-ai-can-do"></a>
| Use Case            | Example                                           |
|---------------------|--------------------------------------------------|
| ✍️ Text Generation   | ChatGPT writes a blog or story for you           |
| 🖼️ Image Generation | DALL·E creates a picture of “a cat flying in space” |
| 🎥 Video Generation | Sora creates a video from a text prompt          |
| 👩‍🏫 Tutoring        | An AI agent explains math problems              |
| 🎵 Music Creation   | AI makes a new song in the style of your favorite artist |
| 💻 Code Generation  | Copilot writes code based on what you want to build |

## 🧑‍💻 Example (Text): <a id="example-text"></a>

**Prompt:**  
>“Write a funny poem about a banana going to space.”

**GenAI Output:**  
>There once was a banana named Jake,  
Who dreamt of a galactic milkshake.  
He flew to the stars,  
Passed Venus and Mars,  
And danced on the moon with a cake. 🧁🚀

This is original content created by the AI — not copy-pasted.

## 🔁 Types of GenAI Models <a id="types-of-genai-models"></a>

| Type   | Famous Models                                 | What they Generate       |
|--------|-----------------------------------------------|---------------------------|
| Text   | GPT, Claude, LLaMA, Gemini                    | Chat, writing, code       |
| Images | DALL·E, Midjourney, Stable Diffusion          | Photos, art               |
| Videos | Sora, Runway, Pika                            | Short video clips         |
| Music  | Suno, MusicGen                                | Songs, beats              |
| Code   | GitHub Copilot, Code LLaMA                    | Programming code          |

## 🤔 Where is it Used Today? <a id="where-is-it-used-today"></a>

- Chatbots (Customer support)  
- Virtual assistants (Like Siri, Alexa, ChatGPT)  
- Design tools (Auto-generate logos, images)  
- Marketing (Write product descriptions, ads)  
- Education (Personalized tutoring)  
- Coding (Generate functions, fix bugs)  
- Healthcare (Suggest diagnoses, generate reports)

## 🧪 Quick Fun Exercise: <a id="quick-fun-exercise"></a>

**You type:**  
>"Create a startup idea that involves pizza and robots"

**GenAI replies:**  
>"A smart pizza-making robot called RoboSlice that takes custom orders via voice and delivers hot pizza using drones!"

✨ *See? It creates ideas on the fly!*

## ✅ Summary <a id="summary"></a>

| Term       | Meaning                          |
|------------|----------------------------------|
| GenAI      | AI that creates new things       |
| Trained on | Text, images, videos, etc.       |
| Uses       | Writing, drawing, coding, more   |
| Examples   | ChatGPT, DALL·E, Sora, Copilot   |


---

# 🔑 Important Keywords in GenAI (Explained Simply) <a id="important-keywords-in-genai"></a>

## 1. Token <a id="token"></a>

📘 **What is it?**  
A token is a piece of text — could be a word, part of a word, or even punctuation.

💡 **Example:**  
The sentence:  
>*"ChatGPT is smart!"*  

Breaks down into tokens like:  
- "Chat", "G", "PT", " is", " smart", "!"

Each model uses its own tokenizer. GPT usually breaks words into sub-words.

🧠 **Why it matters?**  
- Models process tokens, not raw text.
- More tokens = higher cost and slower response.  
- There are limits (e.g., GPT-4 can handle ~128k tokens max).


## 2. Prompt <a id="prompt"></a>

📘 **What is it?**  
A prompt is the input or question you give to the AI.

💡 **Example:**  
>“Write a story about a robot who makes coffee.”

The AI takes your prompt and generates a response.


## 3. Completion / Output <a id="completion-output"></a>

📘 **What is it?**  
The AI’s response to your prompt.

💡 **Example:**  
If prompt is:  
>“Tell me a joke”

The completion might be:  
>“Why did the computer go to therapy? Because it had too many bugs!”


## 4. Temperature? <a id="temperature"></a>

It is a float value (e.g., 0.0 to 2.0) that adjusts the probability distribution over possible next words when the model is generating text.


### 💡 Simple Explanation:
Think of temperature like a creativity knob:

- **Low temperature** → the model plays it safe (predictable, accurate).  
- **High temperature** → the model becomes more creative (or chaotic!).

### 🎯 How It Works (in simple terms):

Language models generate the next word by picking from many possible words, each with a probability.  
Temperature changes how sharp or flat that probability curve is.

### 🧪 Examples:

Let’s say the model is trying to generate the next word after:  
>**"The cat sat on the"**

1. **temperature = 0.0** (deterministic)  
   - Always picks the highest probability word.  
   - 👉 Output: `"mat"`

2. **temperature = 0.7** (balanced)  
   - A bit of randomness.  
   - 👉 Output: `"mat"`, `"sofa"`, or `"floor"`

3. **temperature = 1.5** (high creativity)  
   - Very random.  
   - 👉 Output: `"rocket"`, `"cloud"`, or `"spoon"`


### ⚙️ Common Settings:

| Temperature | Behavior        | Use Case                    |
|-------------|------------------|-----------------------------|
| 0.0         | Deterministic    | Facts, math, code generation |
| 0.5         | Balanced         | General-purpose conversation |
| 1.0         | Creative         | Storytelling, poem generation |
| >1.2        | Very creative    | Wild ideas, brainstorming     |

### 🧠 In Short:
>Temperature controls how boring or bold your AI's response is.

## 5. Top-k Sampling <a id="top-k-sampling"></a>

Top-k sampling is a method where the model:
>Only considers the **top k most likely next words**, and randomly picks one from them based on their probabilities.


### 🤔 Why do we use it?

To control randomness and reduce weird outputs by **not letting the model choose from all possible words** (some of which have tiny, junky probabilities).


### 🧠 How It Works:

Imagine the model predicts the next word in a sentence, and it gives probabilities for **50,000 possible words**.

- **Without Top-k:** it can choose from all 50,000, even if some are very unlikely.
- **With Top-k = 5:** it picks only from the top 5 most likely words, and samples randomly among those.


### 📊 Example:

The model is generating the next word for:

>**"The pizza tastes"**

Top predicted probabilities:

| Word      | Probability |
|-----------|-------------|
| delicious | 0.45        |
| great     | 0.20        |
| amazing   | 0.15        |
| awful     | 0.10        |
| burnt     | 0.05        |
| wooden    | 0.01        |
| spicy     | 0.01        |
| ...       | ...         |

- Top-k = 3 → consider only: delicious, great, amazing
- Pick one of them randomly, weighted by their probabilities.
- "awful" or "burnt" will not be considered.


### ⚙️ When to use:

| Top-k Value | Behavior                    |
|-------------|-----------------------------|
| k = 1       | Always picks the top choice (deterministic) |
| k = 10      | Balanced randomness          |
| k = 50+     | More creative or surprising  |


### 🔁 Bonus: Often used with Temperature

- First apply **Top-k** to get a shortlist.
- Then apply **Temperature** to adjust randomness **within that shortlist**.


### 🧠 In Simple Words:

> Top-k sampling = “Only pick from the top k best options”, then choose one based on probability.**

## 6. Top-p (Nucleus) Sampling <a id="top-p-nucleus-sampling"></a>

Top-p sampling picks from the smallest set of words whose total probability adds up to **p** (like 0.9), and samples randomly from that set.

So instead of saying “pick top 5 words” (like in top-k), we say:

> “Keep adding the most likely words until their combined probability hits **p** — then sample from those.”

### 🤯 Sounds complex? Let’s make it simple.

Imagine your model wants to complete:

>**"The weather is"**

And it gives the following words with probabilities:

| Word   | Probability |
|--------|-------------|
| nice   | 0.40        |
| sunny  | 0.30        |
| rainy  | 0.15        |
| hot    | 0.05        |
| chilly | 0.03        |
| weird  | 0.02        |
| ...    | ...         |


### 🔍 Now, let’s say:  
✅ **Top-p = 0.9**

We start adding the top words until the sum ≥ 0.9:

- nice (0.40) → total: 0.40  
- sunny (0.30) → total: 0.70  
- rainy (0.15) → total: 0.85  
- hot (0.05) → total: 0.90 ✅  

So now, we randomly pick from:  
**nice, sunny, rainy, hot**

❌ Everything else (like “chilly” or “weird”) is ignored.

### 🔁 How it's different from Top-k:

| Feature          | Top-k             | Top-p (Nucleus)           |
|------------------|-------------------|----------------------------|
| Fixed number     | Yes (e.g. top 5)  | No — picks as many as needed |
| Based on         | Count of words    | Total probability mass     |
| Flexibility      | Less (fixed size) | More dynamic and adaptive  |


### ⚙️ When to Use:

| Top-p Value | Behavior                     |
|-------------|------------------------------|
| 0.9         | Common, balanced randomness  |
| 1.0         | No restriction — full vocab  |
| 0.7         | More focused and safe        |

### 🧠 In One Line:

> Top-p Sampling** picks from a dynamic shortlist of most likely words whose **combined probability ≥ p**, and samples randomly from there.

## 7. Max Tokens <a id="max-tokens"></a>

**Max Tokens** controls how long the output can be from a language model like GPT.


### ✅ In Simple Terms:

>“**Max tokens**” = the maximum number of **words or pieces (tokens)** the model is allowed to generate.


### 🤔 Wait... What is a Token?

A token is **not exactly a word** — it's a piece of text.

| Text             | Tokens |
|------------------|--------|
| Hello            | 1      |
| ChatGPT          | 1      |
| unbelievable     | 2      |
| I love pizza.    | 4      |
| 😊 (emoji)        | 1      |
| 2025-07-02       | 4      |

So `max_tokens` limits the number of **tokens**, not characters or full words.

### 📏 How It Works

If you set `max_tokens = 50`, the model will **stop generating after 50 tokens**, even if it hasn’t finished its sentence.

This helps:

- 🚫 Avoid super long or endless outputs  
- 💰 Control costs (API pricing is often token-based)  
- 📦 Fit within token limits (e.g., 4096 or 8192 total)

### 📌 Important:

The **input + output tokens** together must stay within the model’s total token limit:

| Model      | Token Limit         |
|------------|---------------------|
| GPT-3.5    | ~4,096 tokens        |
| GPT-4      | ~8,192 to 32,768     |

### 🧪 Example

**Prompt:**  
>"Write a short poem about cats."

And you set max_tokens = 20, the output might be:

**Possible Output:**  
>"Cats in sunbeams play,  
Softly purring through the day..."

✋ Then it stops — even if the poem isn’t finished — because it hit the 20-token limit.

### 💡 Use Cases

| Use Case              | Recommended Max Tokens |
|-----------------------|------------------------|
| Short answers (FAQs)  | 10–50                  |
| Chatbots              | 50–200                 |
| Story/essay generation| 200–1000+              |
| Code generation       | Depends, usually 100–800 |

### 🧠 In One Line:

> Max Tokens** limits how much the model can say — like cutting it off after a certain number of words/pieces.

## 8. Stop Sequence <a id="stop-sequence"></a>

A **stop sequence** is a custom string or token that tells the language model:

🗣️ *"Stop generating text once you see this."*

It’s like saying:  
> “As soon as you see this word/phrase, cut off the output!”

### ✅ Why Use Stop Sequences?

- To control where the output ends  
- To avoid unnecessary or repeated text  
- To simulate structured conversation (like ending after one message)

### 🧪 Example 1: Chatbot Message

You give this prompt:

```bash
User: What's your name?
AI:
```
And set stop = ["User:"]

The model might generate:

```bash
AI: I'm ChatGPT, your assistant.

```

It **stops before printing** "User:" again — avoiding generating the next turn in the conversation.

### 🧪 Example 2: Multi-Choice Question

Prompt:

```bash
Q: What is 2 + 2?
A:
```
Set stop = ["\n"]
```bash
A: 4
```
It stops as soon as it hits the first newline (\n) — short, sweet answer ✅

### 🧪 Example 3: JSON Completion

Prompt:

```bash
{
  "name": "Alice",
  "age":
```
Set stop = ["}"]

Output:
```bash
{
  "name": "Alice",
  "age": 30
}
```
It stops right before closing brace — useful for **structured outputs**.

## 🧠 Summary Table

| Feature        | What it Does                                             |
|----------------|----------------------------------------------------------|
| Stop Sequence  | Halts generation when the model outputs a match          |
| Type           | String or list of strings (e.g., `["User:", "\n"]`)      |
| Common Use     | Chatbots, JSON, code, Q&A, structured text               |

## 💡 In One Line

**Stop Sequence** tells the model:  
> *“Stop Sequence tells the model: “Stop writing when you hit this word or phrase.”*

# 9. Fine-tuning  <a id="fine-tuning"></a>

**Fine-tuning** is the process of training a pre-trained language model on your own custom dataset, so it learns to give more specific, domain-relevant, or personalized responses.

### 🪄 In Simple Words:
>You're teaching a smart AI a special skill or style, on top of what it already knows.

### 🔧 Analogy

Imagine GPT is like a chef who can cook all kinds of food.

With fine-tuning, you're teaching the chef to cook **your grandma’s secret recipes** perfectly. 🍲

Now, the chef (GPT) still knows everything — but becomes super good at your **specific style**.

### 🤖 Why Fine-tune a Model?

To make it:

- 🗣️ Talk in your brand voice  
- 🧠 Answer in domain-specific knowledge (e.g., medicine, law, finance)  
- 🧾 Follow specific response formats  
- 🌍 Speak in a different language or tone  
- 👨‍💼 Act like a custom assistant or bot  

### 🏗️ How Fine-tuning Works (Step-by-Step)

1. Start with a base model (like GPT-3.5 or LLaMA)  
2. Prepare a dataset of input-output pairs (called *prompts and completions*)  
3. Train the model on this data using a few passes (called *epochs*)  
4. The model updates its internal weights slightly to favor your examples  

### 📚 Example Dataset

```json
{"prompt": "User: What's your return policy?\n", "completion": "Bot: You can return any item within 30 days with a receipt.\n"}
{"prompt": "User: How long does shipping take?\n", "completion": "Bot: Shipping usually takes 3–5 business days.\n"}
```
✅ After fine-tuning, your model will always respond in this style and tone — even to similar but not identical questions.

### 🔬 Fine-tuning vs Prompt Engineering

| Feature            | Fine-tuning                             | Prompt Engineering                    |
|--------------------|------------------------------------------|----------------------------------------|
| Changes model?     | ✅ Yes — updates internal weights         | ❌ No — just changes the prompt         |
| Custom training?   | ✅ Needs your dataset                     | ❌ Just uses clever wording             |
| Cost?              | 💰 Higher (training & hosting)            | 💸 Lower (just inference)               |
| Flexibility        | ✅ More control over behavior             | 🟡 Limited, but easier                  |

### ⚠️ When to NOT Fine-tune

- If you just want **minor tweaks** → use **prompt engineering** or **function calling**
- If **data is confidential** → be careful about what you upload
- If your use case is **simple or short-lived**

### 💡 In One Line

> *Fine-tuning** teaches a pre-trained AI model to act like an expert in your specific use case, using your own training data.

## 10. Embeddings  <a id="embeddings"></a>

An **embedding** is a way to convert text into numbers so that a machine can understand it — but not just random numbers:  

It captures the **meaning and context** of the text in **vector form**.


### 🗣️ In Simple Words

Imagine you want to teach a computer that:

- "Dog" and "Puppy" are related ✅  
- "Dog" and "Carrot" are not ❌

You can’t just give it the words — you need to give it **meaningful numbers**.

That’s what an **embedding** does.

### 🔢 Example (3D vector for illustration)

| Text     | Embedding          |
|----------|--------------------|
| Dog      | [0.91, 0.21, 0.55] |
| Puppy    | [0.89, 0.20, 0.56] |
| Carrot   | [0.12, 0.99, 0.33] |

Now the computer can "see" that:

- 🟢 **Dog** and **Puppy** are close together  
- 🔴 **Dog** and **Carrot** are far apart

✅ This lets AI reason about **meaning**, **similarity**, and **relationships**.

### 🧠 Where Are Embeddings Used?

| Use Case                     | Why Embeddings Matter                                     |
|-----------------------------|-----------------------------------------------------------|
| Semantic Search             | Find similar documents even if wording is different       |
| Recommendation Systems      | Suggest similar movies, products, etc.                    |
| Clustering                  | Group similar items or topics                             |
| Text Classification         | Label content (e.g., spam detection, sentiment)           |
| RAG (Retrieval-Augmented Generation) | Fetch relevant info from a database            |


### 📚 Real-Life Example: Semantic Search

Suppose a user searches:
> `"How to cook pasta?"`

You embed that query as a vector:  You embed that query: `[0.45, 0.67, 0.88, ...]`

Then you embed all your documents (once), and compare them using **cosine similarity**.

You’ll get results like:

- ✅ **"Best pasta cooking techniques"** → High similarity  
- ❌ **"How to fix a car engine"** → Low similarity

💡 This is how AI search engines and chatbots with memory work under the hood.

### 🛠️ Tools You Can Use

- 🔗 **OpenAI Embeddings API** (e.g., `text-embedding-3-small`)
- 🤗 **HuggingFace Sentence Transformers**
- 📦 **FAISS** / **Weaviate** / **Pinecone** for vector storage & similarity search

### 🤯 Fun Fact

The **embedding space** has **hundreds or thousands of dimensions** (not just 3), so it's like mapping meaning into a huge invisible **galaxy of concepts** 🌌

### 💡 In One Line

> **Embeddings** = turning words or sentences into numbers that represent their meaning, so machines can **compare** and **reason** about them.

## 🔎11. RAG (Retrieval-Augmented Generation)<a id="rag-retrieval-augmented-generation"></a>

**RAG** is a technique where a language model (like GPT) is given access to **external information** (retrieval) before it generates answers (generation).

So instead of the model guessing from memory, it can **look things up first**.

### 📦 In Simple Words

Think of RAG like a smart student with Google:

- It retrieves notes from the internet (or a database)  
- Then uses them to answer your question

### 🔁 The RAG Process (Step-by-Step)

### 🔍 1. Retrieve
- Convert the user's question into embeddings  
- Search a vector database for relevant documents or chunks  
  (This could be PDFs, websites, notes, etc.)

### ✍️ 2. Augment
- Feed the retrieved info **plus** the original question into the LLM

### 🧠 3. Generate
- The model uses both inputs to generate a more **accurate, grounded answer**

### 📚 Example

You ask:  
>**"What are the side effects of Ibuprofen?"**

🧠 Without RAG (standard GPT):
>It tries to remember from its training data (may be outdated)

🔍 With RAG:
1. It searches a medical knowledge base  
2. Finds a chunk that says:  
>*“Common side effects of Ibuprofen include nausea, dizziness, and stomach pain.”*

3. Then it generates:  
>**"Ibuprofen may cause nausea, dizziness, and stomach pain, according to the medical database."**

✅ Now it’s **factual + grounded**

### 🤯 Why Use RAG?

| Without RAG                | With RAG                         |
|----------------------------|----------------------------------|
| May hallucinate info       | Grounded in real data            |
| Limited to training data   | Can access live, updatable info |
| Static knowledge           | Dynamic, real-time knowledge     |

### 💡 Where RAG is Used

| Use Case               | How RAG Helps                               |
|------------------------|---------------------------------------------|
| AI Chatbots            | Pull info from your docs or helpdesk        |
| Internal Knowledge Bots| Use your company wiki or Notion pages       |
| Customer Support       | Provide product answers from manuals        |
| Legal / Medical AI     | Pull accurate info from trusted docs        |
| Research Assistants    | Pull live papers or academic content        |

### 🛠️ How to Build a RAG System

1. Embed your documents using **OpenAI** or **HuggingFace**
2. Store them in a vector DB like **Pinecone**, **Weaviate**, **FAISS**, or **Chroma**
3. When a user asks something:
   - Convert query → embeddings  
   - Search DB → get relevant chunks  
   - Feed both into GPT using context + prompt

🔥 Boom — Smart, grounded answer!

### 💡 In One Line

> **RAG** = Letting AI *"look up"* relevant info before answering,so it’s more **accurate**, **up-to-date**, and **useful**.

## 12. Context Window <a id="context-window"></a>

The **context window** is the **maximum number of tokens** (words, parts of words, or symbols) a language model can "see" at once during a conversation or prompt.

Think of it as the model’s **short-term memory**.

### 🗣️ In Simple Words

It’s how much **text** the model can remember and pay attention to at a time —  
both your input and its own previous responses.

### 🧠 Example

Let’s say a model has a context window of **4,096 tokens**:

- 📝 You send: 1,000 tokens  
- 🤖 Model responds: 500 tokens  
- 🔢 Total used: 1,000 + 500 = 1,500 tokens

You can continue chatting until you hit the **4,096-token limit**.

> After that, older messages might be forgotten or cut off (unless managed manually).

### 🔢 How Big Are Context Windows?

| Model            | Max Context Window         |
|------------------|----------------------------|
| GPT-3.5 Turbo    | 4,096 or 16,385 tokens      |
| GPT-4            | 8,192 to 32,768 tokens 😮   |
| Claude 3 Opus    | 200,000+ tokens             |
| Gemini 1.5 Pro   | 1,000,000 tokens 🧠         |

### 🔁 What Happens When It’s Full?
If your input + history exceeds the limit:

- The **oldest tokens** are truncated or dropped  
- The model may “forget” earlier parts of the conversation  
- It might lose track of context, leading to weird or wrong answers

### 📦 Analogy

Imagine the model is reading through a **sliding window**:  
- It can only see a portion of the text at a time.  
- As new text comes in, **old text slides out**.

### 💡 Why Context Window Matters

| Use Case              | Importance of Context Window                      |
|------------------------|---------------------------------------------------|
| Long conversations     | Needed to keep full chat history                 |
| Legal/Medical docs     | Must fit large document content                  |
| Code generation        | Helps maintain variables/functions               |
| RAG systems            | Must insert enough relevant context              |

### 🔍 Reminder

- **Tokens ≠ Words**  
- Typically: `1 token ≈ ¾ word` in English  
- So: `100 tokens ≈ 75 words`

Use [OpenAI’s tokenizer tool](https://platform.openai.com/tokenizer) to check token counts.

### 🧠 In One Line

> The **context window** is how much text (in tokens) an AI model can **"see and remember" at once** — like its **short-term brain space**.

## 13.Memory (Long-Term Brain) <a id="memory-long-term-brain"></a>
Think of **Memory** like a notebook where Gen AI writes down important stuff to remember for future chats.

- **Without memory**: It's like talking to a goldfish. You say: “My name is Suraj,” Next second? It forgets. 🐠

- **With memory**: You say: “My name is Suraj,” and next time it greets you like:  
  _“Hey Suraj, back to break more code, huh?”_

In tools like ChatGPT, memory is optional and usually NOT active by default. But some platforms let the AI store and recall facts, tasks, notes, etc.

### 📌 Simple Example:
```bash
You: My dog's name is Bruno.
(With memory ON, the AI saves: dog name = Bruno)

You (next day): What’s my dog’s name?
AI: Bruno!

```
>✅ Memory = stored knowledge across conversations.

## 15. Context (Short-Term Brain) <a id="context-short-term-brain"></a>
**Context** is like the current conversation history. It's what the AI remembers *right now* — like short-term working memory.

- You ask: “What’s 2+2?” → AI says: “4”
- You say: “And add 5 more?” → It needs the previous message to know you're referring to "4".

If too much stuff is happening, older context gets forgotten — like a brain with a limited number of sticky notes.

### Limits:
- GPT-4 has a 128k token context window (~300 pages of text).
- Beyond that, old messages may be dropped or compressed.

>🧠 Context = chat history the model can “see” right now.

## 16. Looping (Function Calling / Tool Use) <a id="looping-function-calling-tool-use"></a>

Now let’s get to looping, which is kinda like Gen AI calling itself or doing tasks again and again (with variations) until something is “done.”

There are two flavors:

a. Prompt-based Looping (You simulate it)
You write:

> “Generate 5 blog titles. If none sound catchy, improve and try again.”

The model follows your prompt and might try multiple outputs in one go — but it’s still one big request. You're faking a loop inside the prompt.

b. Real Looping (Using Tools / Code)
This is where you tell the AI:
>**“If X happens, call this function again.”**

Like:
```bash
while (quality < 90%) {
    call GPT with new prompt;
}
```
You can combine this with function calling (OpenAI or LangChain style):

### Example:
**User**: “Summarize this document.”  
**AI**: Calls a summarize() function  
**System checks length** → If it’s still too long, it loops again.



## 🧠 Zero-shot Learning <a id="zero-shot-learning"></a>

> Ask the model to do something **without giving it any examples**.

Basically saying:  
>**“Hey GPT, figure this out based on your general knowledge.”**

### 📚 Example

**Prompt:**

>Translate to French:  
"I love pizza."

**✅ Output:**
> "J'aime la pizza."

🔍 You didn’t teach it anything — it just knew what to do.

## 🧠 Few-shot Learning <a id="few-shot-learning"></a>

> Ask the model to do something by giving it **a few examples first**, then a new input.

This helps the model understand the **task pattern.**

### 📚 Example

**Prompt:**

```bash
Translate to French:
English: I love pizza → French: J'aime la pizza
English: Good morning → French: Bonjour
English: How are you → French:
```
**✅ Output:**
>Comment ça va

🧠 This is **few-shot** because you showed **a few examples**, then gave a new case.

## 🧠 Chain-of-Thought (CoT) Reasoning <a id="chain-of-thought-cot-reasoning"></a>
You ask the model to **think step by step**, not just give the answer.

Useful for **math, logic, or complex problems** where reasoning is needed.


### 📚 Example

**Prompt:**

>Q: If you have 3 apples and you buy 2 more, how many do you have in total?  
A: Let's think step by step.

**✅ Output:**
>First, I have 3 apples.
Then I buy 2 more, so now I have 3 + 2 = 5 apples.
Answer: 5

### 💡 Why It Matters

| Method     | Use Case                             | Strength                              |
|------------|---------------------------------------|----------------------------------------|
| Zero-shot  | Quick general tasks                  | Fast, no prep needed                   |
| Few-shot   | Custom format, tone, or domain logic | Helps model learn your pattern         |
| CoT        | Math, logic, reasoning               | Improves accuracy via step-by-step     |

### 🧠 Combine Them?
YES! You can mix **few-shot + chain-of-thought** for even better results:

Example:

```bash
Q: There are 5 cars. Each car has 4 wheels. How many wheels total?
A: Let's think step by step.
Step 1: Each car has 4 wheels.
Step 2: 5 cars × 4 wheels = 20 wheels.
Answer: 20
```
✅ This improves reasoning drastically!

### 🧠 In One Line:
>**- Zero-shot = Just ask**
>**- Few-shot = Give a few examples**
>**- Chain-of-Thought = Ask it to explain its reasoning step-by-step**


## 🧠 LLM (Large Language Model) <a id="llm-large-language-model"></a>

An LLM is an AI model trained on a **huge amount of text data** that can understand, generate, and manipulate human language.

It powers tools like **ChatGPT, Claude, Gemini**, and even AI coding assistants like **GitHub Copilot**.

### 🗣️ In Simple Words

An LLM is a **super smart text engine** that learned language by reading the entire internet  
(books, articles, Wikipedia, code, etc.)

Now it can:
- Talk like a human  
- Write essays  
- Translate  
- Summarize  
- Code  
...and more!

### 🏗️ Why is it called "Large"?

Because it:

- Has **billions (or trillions) of parameters**
- Was trained on **terabytes of text**
- Can handle **long and complex tasks**

#### 📊 Examples:

| Model         | Parameters         |
|---------------|--------------------|
| GPT-3         | 175 billion         |
| GPT-4         | unknown (larger)    |
| LLaMA 3       | up to 70 billion    |
| Claude 3 Opus | ~200B+ (estimated)  |

### 🔧 What Can LLMs Do?

✅ Text generation  
✅ Translation  
✅ Summarization  
✅ Code generation  
✅ Q&A  
✅ Sentiment analysis  
✅ Chatbots  
✅ Reasoning & math (limited)  
✅ Image descriptions (if multimodal)

## 🔧 How Does an LLM Work? — Step-by-Step

### 🔥 Step 1: Learn from the Internet (Training Phase)
Imagine giving the AI a **massive reading assignment:**

>“Hey AI, read **books, Wikipedia, blogs, Reddit, news, code**, and more!”

It reads **billions of sentences** and learns patterns like:

- Which words go together  
- Sentence structures  
- Logic and reasoning  
- Coding styles  
- How people ask and answer questions  


#### ⚠️ **Important:**  
It doesn’t memorize everything.It learns **patterns using numbers**

### 📦 Step 2: Break Text into Tokens
Before training, text is broken into **tokens** (small pieces of words).  

Example:  
>"I love pizza" → `["I", " love", " pizza"]`  

The model doesn’t see words — it sees **numbers representing these tokens**.

### ⚙️ Step 3: Predict the Next Token (Core Idea)
LLMs are basically **super-smart guessing machines**.

**For example:**  
>Prompt: “I love eating hot...”  
LLM guesses: _dogs_, _pizza_, _chocolate_, etc.  
It picks the **most likely next word** using math.

This is called:

>🧠 **"Language Modeling = Predict the next word."**

Over and over again. That's it!

### 🔁 Step 4: Use Transformers (The Engine Behind the Magic)
LLMs use a special architecture called the Transformer (created in 2017).

Transformers introduced the concept of Attention, meaning:

>"Focus on the important parts of the sentence."

Example:

>"The cat that the dog chased was fluffy."
To understand "was fluffy," the model needs to pay attention to "cat," not "dog."

🧠 Transformers help the AI understand context better than older models.

### 🏁 Step 5: You Ask, It Generates (Inference Time)
After training is done, you can chat with the LLM.

You give a prompt, like:  
> “Write a story about a robot who loves pizza.”

The LLM:

- Breaks your prompt into **tokens**  
- Runs it through its **trained neural network**  
- Predicts the **next token**, then next, then next...  
- Combines them to generate a full response  

### ⚡ **LLM = Pattern Predictor**  
LLMs don't actually think or know facts.  
They generate words based on:

- Training data (**patterns**)  
- Your prompt (**context**)  
- Temperature settings (**randomness**)  

They're just **giant pattern-predicting machines**.

🎯 **Simple Example: Fill in the Blank**  

>**Prompt:** “Ronaldo is a famous football ___”

The LLM sees:  
- Based on training, “player” is a common next word.  

>✅ **Output:** “player”

If you change the prompt:

>“Ronaldo is a famous football... from Portugal who plays for Al-Nassr.”

Now it predicts a **more detailed** answer.

### 🎥 **Visual Analogy**  
Imagine this as a movie:

| Scene              | What’s Happening                  |
|--------------------|-----------------------------------|
| 📖 Reading phase    | AI reads billions of words        |
| 🧩 Tokenizing        | Breaks all words into tokens      |
| 🧠 Learning patterns | Figures out “how language works” |
| 🤖 Chat time         | You ask it something              |
| 🧮 Neural net kicks in | It processes your input          |
| 🎯 Predicts & generates | One token at a time             |

### 🛠️ **Tools LLM Uses Internally (Behind the Scenes)**

| Term        | Meaning                                   |
|-------------|--------------------------------------------|
| Token       | Smallest unit of text it understands       |
| Embedding   | Converts tokens into numbers               |
| Attention   | Figures out what's important               |
| Transformer | The brain architecture                     |
| Decoder     | Predicts next word/token                   |
| Weights     | Trained values that store knowledge        |

### 🔍 **Summary**

| Concept     | Explained Simply                              |
|-------------|------------------------------------------------|
| LLM         | A big AI that predicts the next word           |
| Trained on  | Internet text (books, websites, etc.)          |
| Learns      | Language patterns using math                   |
| Uses        | Transformers & Attention                       |
| Goal        | Understand your prompt and generate output     |


### 🧠 How Do LLMs Work (Simplified)?

#### 🔁 Training:
- Reads billions of text examples  
- Learns to **predict the next word** in a sentence  
- Over time, becomes great at understanding and generating language

#### 🚀 Inference (Usage):
- You give it a **prompt** (e.g., “Tell me a joke”)
- It generates a **response**, word by word

### 📚 Example:

**Prompt:**

>"Once upon a time,"

**LLM Output:**
> “there was a dragon who loved to bake cookies.”

“there was a dragon who loved to bake cookies.”

### 💬 LLM vs Traditional AI

| Feature                    | Traditional AI | LLM                         |
|----------------------------|----------------|------------------------------|
| Rule-based logic           | ✅ Yes         | ❌ No (learns from data)     |
| Needs task-specific training | ✅ Yes       | ❌ No (general-purpose)      |
| Understands language       | ❌ Limited     | ✅ Yes (very well)           |

### 🔐 Limitations of LLMs

⚠️ Can **hallucinate** (make things up)  
⚠️ Doesn’t understand like a human (no real consciousness)  
⚠️ Needs a lot of **computing power**  
⚠️ Can be **biased** (reflects training data)  
⚠️ Sensitive to **prompt phrasing**

### 🧠 In One Line

>**A Large Language Model is a powerful AI trained on tons of text, capable of reading, writing, and understanding human language — like a digital brain for words.**

# 🧠 What is a Chatbot? <a id="what-is-a-chatbot"></a>

A **chatbot** is a software app that talks to users, like me (ChatGPT).  
You give it a question, it gives you an answer.

A **GenAI chatbot** means we use **Generative AI** (like OpenAI GPT models) to power the brain of the bot.

### 🛠️ What We’ll Build First <a id="what-well-build-first"></a>

A simple chatbot using:

- **Node.js** (backend server)  
- **OpenAI GPT-4 API** (AI brain)

```bash
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
```

## 📡 What is Streaming in OpenAI? <a id="what-is-streaming-in-openai"></a>

In OpenAI, **streaming** refers to delivering responses from models (like ChatGPT) **incrementally** instead of all at once. 

This is especially useful for apps like:
- 🤖 Chatbots  
- 🧑‍💻 Code editors  
- 💬 Live assistants  

Where users expect fast, real-time feedback.

### 🧠 How It Works (Simple Explanation)

#### ❌ Without Streaming:
1. You send a prompt.
2. You wait...
3. The model thinks...
4. You get the **entire** answer at once.

#### ✅ With Streaming:
1. You send a prompt.
2. The model **starts sending tokens immediately**.
3. You see the response **"typing out" live**, like a human writing.


### 📦 Why Use Streaming?

| Without Streaming                            | With Streaming                                   |
|---------------------------------------------|--------------------------------------------------|
| Full response comes **after** generation    | Tokens start arriving **during** generation     |
| Feels slower                                | Feels faster and more

###  ⚙️ Technical Example (JavaScript + fetch)
When using the OpenAI API with stream: true:

```bash
const response = await fetch("https://api.openai.com/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer YOUR_API_KEY`
  },
  body: JSON.stringify({
    model: "gpt-4",
    messages: [{ role: "user", content: "Tell me a joke." }],
    stream: true // 👈 key part
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value, { stream: true });
  console.log(chunk); // Logs the streamed tokens
}
```

### 🧪 Use Cases

- 💬 **Chat apps** (real-time feel)
- 🧠 **Live coding assistants**
- 🎙️ **Voice assistants**
- 🖥️ **Streaming to terminal apps or dashboards**


### 📝 Notes

- You must handle the **chunked response** properly.
- You'll receive data as **Server-Sent Events (SSE)**.
- Each chunk contains a `data:` payload that includes **part of the response**.
- You'll need to **parse and append** these chunks in your frontend to build the full response.


## 🤖 Function Calling <a id="function-calling"></a>

Let’s break down OpenAI Function Calling in the easiest way possible. Think of it as giving GPT **superpowers** by letting it use **your real code** to get stuff done!

### 📘 Chapter 1: What Is Function Calling? <a id="chapter-1-what-is-function-calling"></a>

Imagine you’re building a robot (like me 😄). You give it tasks like:

> “Book a flight.”

Now instead of teaching the robot everything about flights, you simply say:

> “When I say ‘book a flight’, go call **this function** I already made.”

#### ✅ So, function calling means:
- You give **GPT** the ability to use **your functions**.
- It can **decide when to use them** based on your instructions.
- It returns **function arguments**, and **you** run the function in your backend.

### 📘 Chapter 2: Why Use Function Calling? <a id="chapter-2-why-use-function-calling"></a>

#### 🤔 Without Function Calling:
> You: “What’s the weather?”  
> GPT: “I think it’s sunny.” *(based on old training data)*

#### ✅ With Function Calling:
> You: “What’s the weather?”  
> GPT: “Let me check…” *(calls your real-time weather API)*  
> GPT: “It's 29°C and sunny in Dehradun!”

#### 🧠 Perfect for:
- 🔴 **Live data** (weather, prices, stock updates)
- 🛠️ **Actions** (bookings, sending emails, running workflows)
- 🧩 **Custom logic** (databases, filtering, dynamic content)

### 📘 Chapter 3: The 3 Players <a id="chapter-3-the-3-players"></a>

| Role     | Description |
|----------|-------------|
| 🧑 You   | Define the function and decide what GPT can use. |
| 🧠 GPT   | Reads user input, decides when to call your function. |
| 🖥️ Backend | Actually runs the function and returns the result to GPT. |


Want a hands-on example next? Like calling a real-time weather API using Node.js and OpenAI function calling? Just say the word. 🌦️

### 📘 Chapter 4: A Simple Example <a id="chapter-4-a-simple-example"></a>

Let's say you have a function to get the weather:

```bash
function getWeather(city) {
  return `It’s sunny in ${city}`;
}
```

Now you tell GPT about it:

```bash
const functions = [
  {
    name: "getWeather",
    description: "Get the weather for a city",
    parameters: {
      type: "object",
      properties: {
        city: {
          type: "string",
          description: "The city name",
        },
      },
      required: ["city"],
    },
  }
];
```

Then you call GPT:

```bash
const completion = await openai.chat.completions.create({
  model: "gpt-4-0613",
  messages: [{ role: "user", content: "What’s the weather in Delhi?" }],
  functions,
});
```

Now GPT might say:

```bash
{
  "function_call": {
    "name": "getWeather",
    "arguments": "{ \"city\": \"Delhi\" }"
  }
}
```

It’s telling you: “Hey, I want you to run `getWeather("Delhi")`.”

### 📘 Chapter 5:  Handling the Function <a id="chapter-5-handling-the-function"></a>

So you now run that function:

```bash 
const result = getWeather("Delhi");
```

Then you tell GPT the result:

```bash 
const finalResponse = await openai.chat.completions.create({
  model: "gpt-4-0613",
  messages: [
    { role: "user", content: "What’s the weather in Delhi?" },
    {
      role: "assistant",
      function_call: {
        name: "getWeather",
        arguments: JSON.stringify({ city: "Delhi" }),
      },
    },
    {
      role: "function",
      name: "getWeather",
      content: result, // the answer from your real function
    },
  ],
});
```
GPT will now say:
>“It’s sunny in Delhi!”

### 📘 Chapter 6: Summary (TL;DR) <a id="chapter-6-summary-tldr"></a>

-  You define what GPT is allowed to call
- ✅ GPT picks the right one and sends input
- ✅ You run it and give the output back
- ✅ GPT continues the convo based on real data

### 📘 Chapter 7: Real-Life Use Cases <a id="chapter-7-real-life-use-cases"></a>

| 💡 Use Case         | 🧩 Function Signature                          |
|---------------------|-----------------------------------------------|
| 🌦️ Weather          | `getWeather(city)`                            |
| 🪙 Crypto Price     | `getCryptoPrice(coin)`                        |
| ✈️ Book Flight       | `bookFlight(name, date, from, to)`            |
| ⏰ Set Reminder      | `createReminder(text, date)`                  |
| 📡 Call Your API     | `fetchUserData(userId)`                       |

### 📘Chapter 8: Example <a id="chapter-8-example"></a>

This code shows how to use **OpenAI function calling** to make a chatbot smart — it can call real functions like `getWeather` or `summarize` when needed.  
It also uses **memory** by saving previous chat messages, so the bot remembers the conversation context.

🔗 [View on GitHub](https://github.com/your-username/your-repo/blob/main/index.js)

## 🔍 What is RAG? <a id="what-is-rag"></a>

Imagine you’re writing an answer to a question. But instead of relying only on your memory, you Google the latest articles, pick the most helpful ones, and then write your reply.

That’s exactly what RAG does!

>RAG = Retrieval (search) + Generation (write)

It first retrieves relevant documents.

Then it generates an answer using those documents.

### 🧱 Key Components of RAG <a id="key-components-of-rag"></a>

| Component | Role in RAG | Analogy |
|:---------|:------------|:--------|
| 🔍 Retriever | Finds relevant chunks from your documents | Like Googling answers |
| 🧩 Chunking | Splits large texts into small pieces | Like cutting a pizza into slices |
| 📐 Embeddings | Converts text into numbers for comparison | Like turning words into coordinates |
| 📚 Vector Store | Stores and indexes these number-versions | Like a giant smart filing cabinet |
| 🧠 LLM (ChatGPT) | Reads the retrieved chunks and answers | Like a smart student writing essays |

## 🔍 Understanding the Start of the RAG Process <a id="understanding-the-start-of-the-rag-process"></a>

> 📝 *Let us look at how the RAG process begins — the chart above is just for illustration purposes only.*

### 📄 Input Document (PDF with 10M Tokens)

The document, which contains a large number of tokens (e.g., words or characters), is broken down into smaller, manageable chunks.  
Each chunk typically contains around **1,000 tokens**. This makes it easier to process and analyze large texts.

### ✂️ Chunking Process

The large text document is divided into smaller “chunks” of text.  
For instance, if you have a PDF document, it could be split into several sections or paragraphs — each considered a **chunk**.

### 🧠 Embedding Generation

Each chunk is then processed by a **Language Model (LLM) embedder**,  
which converts the text into an **embedding** — a numerical representation of the text.

This embedding captures the **semantic meaning** of the text and is represented as a **vector** (a list of numbers).

**Example:**
```bash
"Dog" → [1, 2, 4, 1]
"Cat" → [1, 2, 3, 2]
"House" → [0, 3, 7, 9]
```

### 🗃️ Vector Store

The embeddings (numerical representations) of all the chunks are stored in a **vector store**.

This is a database optimized for storing and retrieving **high-dimensional vectors**.

The vector store is used to **quickly find relevant chunks based on a query** made during the Q&A process.


## 📚 Step-by-Step Theory (with Examples) <a id="step-by-step-theory-with-examples"></a>

### Step 1: Prepare Your Data (Docs, FAQs, etc.) <a id="step-1-prepare-your-data-docs-faqs-etc"></a>
You have some documents like this:

```bash
"Bananas are rich in potassium and are good for heart health."
"Apples contain antioxidants and fiber."
"Mangoes are tropical fruits high in Vitamin C."
```

### Step 2: Chunking 🍕 <a id="step-2-chunking"></a>
If a document is too big, we break it into smaller overlapping chunks.

✅ **Why?**
- LLMs can only read a few tokens (words) at a time.
- Overlapping helps keep context between chunks.

✂️ **Example:**
```makefile
Original:
"Bananas are rich in potassium. They are good for the heart and digestion."

Chunk 1 (0-50 chars): "Bananas are rich in potassium. They are good"
Chunk 2 (20-70 chars): "They are good for the heart and digestion."
```
So you slide a window over the text with some overlap.

### Step 3: Embedding 📐 <a id="step-3-embedding"></a>
We convert each chunk into a vector (array of numbers). These represent the meaning of the text.

Use OpenAI Embeddings API or something like `@pinecone-database/doc-splitter`.

**Example using OpenAI:**
```js
const { OpenAI } = require('openai');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function embedText(text) {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });

  return res.data[0].embedding; // returns a vector of floats
}
```

### Step 4: Store in Vector DB 🧠 <a id="step-4-store-in-vector-db"></a>
Use vector DBs like:

- Pinecone
- Weaviate
- Chroma
- FAISS (for local use)

They allow semantic search: Find the most similar chunks based on meaning, not keywords.

We’ll use Chroma (local) for this example to keep it simple.

### Step 5: Ask a Question 🗣️ <a id="step-5-ask-a-question"></a>
Let’s say you ask:

```bash
"What fruit is good for the heart?"
```

- Convert your question into an embedding
- Search similar chunks in the vector DB
- Get top 3 matching texts

### Step 6: Generate Answer using OpenAI 💡 <a id="step-6-generate-answer-using-openai"></a>
Pass the retrieved texts + question into gpt-4:

```js
const prompt = `
Context:
1. Bananas are rich in potassium and good for heart health.
2. Apples contain antioxidants.

Question: What fruit is good for the heart?

Answer:
`;

const chatRes = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: prompt }],
});
```

The LLM reads the context and writes a **smart answer**.

### 🔁 Flow Diagram with Arrows <a id="flow-diagram-with-arrows"></a>

```bash
[User Asks Question]
         ↓
  [Retriever (Vector DB)]
         ↓
[Find Relevant Chunks]
         ↓
[Send Chunks + Question to LLM]
         ↓
     [LLM Generates Answer]
         ↓
     [Return Answer to User]

```

### RAQ Q&A Flow <a id="raq-qa-flow"></a>

- **Question Input**: The process begins with a user posing a question (RAQ — Retrieval-Augmented Question Answering).
- **Retrieval Step**: The question is processed by a retriever model, which checks the stored chunks of text (from the vector store) to find the most relevant information. This step is based on similarity scoring (e.g., scores between 0.0 to 1.0).
- **Chunk Retrieval**: The retriever pulls out the most relevant chunks of text that are likely to contain the answer to the user’s question.
- **AI Processing**: These retrieved chunks, along with the original question, are passed to an AI model like ChatGPT. The AI model processes the input and generates an appropriate response.
- **Response to User**: Finally, the AI’s response is sent back to the user, completing the Q&A flow.

Combining the entire Q&A Flow along with the Vector Store looks like this.

### RAG Example <a id="rag-example"></a>

This code shows how to use Retrieval-Augmented Generation (RAG) to make a chatbot smarter — it finds and uses real text chunks from data.txt to answer questions.
It also uses embeddings to compare the user's question with saved notes and respond with context-aware answers.

🔗 [View on GitHub](https://github.com/suraj-naithani/Gen-ai/blob/main/rag.js)
