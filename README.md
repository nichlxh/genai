# Singapore Budget 2024 Chatbot 
This repository shares the approach for the RAG-powered Chatbot, focusing on the Singapore Budget 2024. 

#### Deployment Guide:

- Cloud Hosted: https://htxdigital-poc.streamlit.app/
- Docker:
  1) Download the Dockerfile in the repository `app/DockerFile`.
  2) Run the below command within the same directory of the downloaded DockerFile to build the `streamlit` image.
     - `docker build --no-cache -t streamlit . -f dockerFile`
     - This should take around 2mins.
  4) Run the command below to run the image (please input your OpenAI API key in environment variable: `OPENAI_API_KEY`):
     - `docker run -e OPENAI_API_KEY='' -p 8501:8501 streamlit `
  5) Enter `https://0.0.0.0:8501` on browser to view the Streamlit interface.

---
![Architecture 1](https://github.com/nichlxh/genai/blob/main/images/a1.svg?raw=True)

#### Approach: 
With the rising interest in Agentic LLMs, such as [Autogen](https://microsoft.github.io/autogen/), where it deploys a Multi-Agent Conversation Framework for agents of different roles to collaboratively solve a real-world problem, my approach to the assessment takes inspiration from this research direction. Although my approach does not involve the use of multi-turn Agent-Agent conversations in [Autogen](https://microsoft.github.io/autogen/), I have instead, used a total of $4$ agents of different roles, collaborating to generate the text output. As per the above figure, the $4$ agents span across: 
1. **Input and Output Guardrail Agent:** Safeguards against (a) User Input and (b) Main Conversation Agent's output
2. **History-aware Rephrasing Agent:** Summarize the chat history and current user prompt to rephrase the current user prompt to better retrieve relevant chunks.
3. **Retrieval Filtering Agent:** A 2nd layer "filter" of the chunks retrieved via vanilla RAG e.g., from 6 chunks reduced to 3 most relevant chunks.
4. **Main Conversation Agent:** The main conversational agent to correspond with the user.

In the following short sections, I will share more about the system developed, with the goal of providing an intuition of how each component is deployed.

---
#### Data:
![Dataset](https://github.com/nichlxh/genai/blob/main/images/data.svg?raw=True)

For simplicity, I have downloaded the PDF files offline and parsed them as input, accordingly chunking them and storing them in the vector database for RAG usage.

---

#### List of Prompts:
Below is the list of all prompts tested (corresponding screenshot outputs will be in the further sections):
1. Am I eligible for the Majulah Package?
2. What are the payouts I can expect to receive in December 2024?
3. What are the Key reasons for high inflation over the last two years?
4. 

1. ![Prompt 1](https://github.com/nichlxh/genai/blob/main/images/p1.svg?raw=True)
---
2. ![Prompt 2](https://github.com/nichlxh/genai/blob/main/images/p2.svg?raw=True)
---
3. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
---

#### Input and Output Guardrails (Agent):
4. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
5. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
6. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
7. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
---

#### History-aware Rephrasing (Agent):
8. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
9. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
10. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
11. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
---

#### Retrieval Filtering (Agent):
12. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
13. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
14. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
---

#### Main Conversation (Agent):
15. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
16. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)

---

#### Citation Processing:
17. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
18. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)

##### References Integrity:
19. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
20. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)

---

#### Chunking Strategies:
21. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
22. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)
23. ![Prompt 3](https://github.com/nichlxh/genai/blob/main/images/p3.svg?raw=True)


---
