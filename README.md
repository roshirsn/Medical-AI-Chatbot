# Medical-AI-Chatbot

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-green)

## 📝 Description

Develop a Medical-AI-Chatbot using Python to provide users with quick and reliable medical information. This chatbot will leverage advanced natural language processing techniques to understand user queries and provide accurate responses based on a comprehensive medical knowledge base. Key features include symptom checking, medication information, and general health advice.

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
langchain: latest
flask: latest
pypdf: latest
sentence-transformers: 2.2.2
python-dotenv: latest
pinecone[grpc]: latest
langchain-pinecone: latest
langchain_community: latest
langchain_openai: latest
langchain_experimental: latest
-e .   #this code used full adding medical-ai-project to pip list: latest
```

## 📁 Project Structure

```
.
├── Data
│   └── medical_book.pdf
├── LICENSE
├── app.py
├── requirements.txt
├── research
│   └── trials.ipynb
├── setup.py
├── src
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
├── static
│   ├── chat.css
│   └── style.css
├── store_index.py
├── template.py
├── templates
│   ├── app.png
│   ├── chat.html
│   ├── doctor.png
│   ├── index.html
│   ├── untitled_image.png
│   └── user.png
└── test.py
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/roshirsn/Medical-AI-Chatbot.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

Please ensure your code follows the project's style guidelines and includes tests where applicable.

## 📜 License

This project is licensed under the MIT License.
