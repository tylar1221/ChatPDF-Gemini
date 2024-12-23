# ChatPDF-Gemini

**ChatPDF-Gemini** is a conversational AI tool that allows users to interact with PDF files by asking questions based on the content. The tool processes the PDFs, extracts the text, and provides detailed answers using the power of Google Generative AI and langchain. It also saves the chat history, which can be downloaded in either Word or PDF format.

## Features
- Upload multiple PDF files and ask questions based on the content.
- Store chat history and download it in Word or PDF format.
- Uses Google Generative AI (Gemini) for context-based question answering.
- Process and split large text into smaller chunks for efficient search and retrieval.
- Save the vector store for future interactions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ChatPDF-Gemini.git
    cd ChatPDF-Gemini
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment:
    - Create a `.env` file and add your Google API key:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```

4. Run the app:
    ```bash
    streamlit run app.py
    ```

## How to Use

1. Upload the PDF documents in the sidebar.
2. Ask questions related to the content of the uploaded PDFs.
3. View and download the chat history in Word or PDF format.
4. Clear chat history with the button provided in the sidebar.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to your forked repository.
5. Create a pull request.



