# 🩺 Medical AI Chatbot 🤖

This project provides a chatbot interface powered by AI to answer medical questions. It leverages a Pinecone vector store for efficient information retrieval and a language model (LLM) to generate informative and helpful responses. The chatbot filters offensive language and can handle casual conversation, providing a user-friendly experience.

## 🚀 Key Features

- **Medical Question Answering:** Answers user's medical questions based on a knowledge base stored in a Pinecone vector store.
- **Contextual Understanding:** Uses a language model (LLM) to understand the context of the question and provide relevant answers.
- **Offensive Language Filtering:** Detects and filters offensive language to ensure a safe and respectful environment.
- **Casual Conversation Handling:** Can handle simple greetings and casual conversation.
- **Efficient Information Retrieval:** Uses Pinecone for fast and accurate retrieval of relevant medical information.
- **Customizable Prompts:** Uses prompt engineering to guide the LLM's responses and ensure accuracy.

## 🛠️ Tech Stack

*   **Frontend:** (Likely a simple HTML/JS interface, not explicitly defined in provided files)
*   **Backend:** Flask
*   **Vector Database:** Pinecone
*   **Language Model:** OpenAI (or similar, via `langchain_openai`)
*   **Embeddings:** Hugging Face Transformers ('sentence-transformers/all-MiniLM-L6-v2')
## Tech Stack

| Area          | Technology |
|---------------|------------|
| Backend       | Flask (Python) |
| LLM           | Google Gemini 2.5 Flash |
| Vector DB     | Pinecone v8 |
| Embeddings    | SentenceTransformers / HuggingFace |
| Frontend      | HTML, Bootstrap, jQuery |
| Environment   | Python 3.9+ |

---
## 📦 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.7+
- Pip package manager
- Pinecone API key
- OpenAI API key (or API key for the LLM you are using)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd medical-ai-project
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    *   Create a `.env` file in the project root.
    *   Add the following variables, replacing the placeholders with your actual API keys:

        ```
        PINECONE_API_KEY=<your_pinecone_api_key>
        OPENAI_API_KEY=<your_openai_api_key> # Or API key for your LLM
        ```

### Running Locally

1.  **Prepare the Pinecone index:**

    *   Run `store_index.py` to create the Pinecone index and populate it with data from the PDF files. Make sure you have PDF files in a directory specified in `store_index.py`.

        ```bash
        python store_index.py
        ```

2.  **Run the Flask application:**

    ```bash
    python app.py
    ```

    This will start the Flask development server. You can then access the chatbot interface in your web browser (likely at `http://127.0.0.1:5000`).

## 💻 Usage

1.  Open your web browser and navigate to the address where the Flask application is running (e.g., `http://127.0.0.1:5000`).
2.  You should see the chatbot interface.
3.  Type your medical question in the input field and press Enter or click the "Send" button.
4.  The chatbot will process your question and display the answer.

## 📂 Project Structure

```
medical-ai-project/
├── app.py                    # Flask backend + RAG pipeline
├── store_index.py            # PDF ingestion + Pinecone indexing
├── templates/
│   └── chat.html             # UI
├── static/assets/            # Project screenshots and images
├── src/
│   ├── helper.py             # Embeddings and preprocessing
│   ├── prompt.py             # System prompt design
│   └── ...
├── requirements.txt
└── README.md

```

## 📸 Screenshots

<img src="assets/Screenshot 2025-12-11 115939.png" width="400px">
<img src="assets/Screenshot 2025-12-11 120518.png" width="400px">
<img src="assets/Screenshot 2025-12-11 123443.png" width="400px">
<img src="assets/Screenshot 2025-12-11 123524.png" width="400px">




## 🏗️ High Level Architecture

```mermaid
flowchart TD
    A[User Message] --> B[Flask Backend]
    B --> C[Pinecone v8 Retriever]
    C --> D[Top-k Chunks]
    B --> E[Gemini 2.5 Flash LLM]
    D --> E
    E --> F[Final Answer]
    F --> G[Frontend Chat UI]

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and concise messages.
4.  Submit a pull request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

Thanks for checking out this project! I hope it's helpful.

This is written by [readme.ai](https://readme-generator-phi.vercel.app/).
