
# Mental Health Chatbot

This repository contains a **Mental Health Chatbot** designed to provide users with mental health resources, guidance, and support. The chatbot leverages advanced AI technologies, including **LangChain**, **Chainlit**, and **OpenAI models**, to deliver accurate and contextual responses. It uses the **RAG (Retrieval-Augmented Generation)** concept to enhance its capabilities.

---

## Features

- **Conversational Interface**: Interact with the chatbot for mental health-related queries.
- **RAG Architecture**: Combines pre-trained language models with a vector database for accurate and contextually relevant answers.
- **Customizable**: Tailored for mental health topics but can be adapted for other domains.
- **Local Deployment**: Execute the application locally with minimal setup.

---

## Setup Instructions

Follow these steps to set up and run the chatbot on your local machine.

### Prerequisites

- Python 3.8 or higher installed
- Install the required Python packages listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

<!--- **Vector Index**: Download the **vector_faiss index** file from the GitHub repository and place it in the `./model/` directory.-->

### Running the Chatbot

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mental-health-chatbot.git
   cd mental-health-chatbot
   ```

2. Run the chatbot application:
   ```bash
   chainlit run app.py
   ```

3. Open the browser at the displayed local address to interact with the chatbot.

---

## Project Structure

```
mental-health-chatbot/
│
├── app.py                  # Main application script
├── vector_faiss/           # Directory to store the vector index
│     
│
├── requirements.txt        # Required Python libraries
├── README.md               # Project documentation
├── screenshots/            # Screenshots of chatbot responses
│   ├── chatbot_ui.png
│   ├── sample_response.png
│
└── other_files/            # Any additional support files
```

---

## Screenshots

### Chatbot User Interface
![Chatbot UI](https://github.com/user-attachments/assets/6ed5d3b9-5792-49a8-9143-409da96485d1)

### Sample Response

---![Sample Response](https://github.com/user-attachments/assets/3ce5dac8-d2bb-4a0d-ab2b-4318cbd256bf)


## Technical Details

1. **LangChain**: Used for structuring conversational AI workflows.
2. **Chainlit**: Provides a clean and interactive user interface for the chatbot.
3. **OpenAI Model**: Powers the core language understanding and response generation.
4. **RAG (Retrieval-Augmented Generation)**: Integrates a FAISS vector index for contextual data retrieval.

---

## Contributions

Contributions are welcome! If you have suggestions for improvement or additional features, feel free to create a pull request or raise an issue.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [Chainlit](https://github.com/Chainlit/chainlit)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

