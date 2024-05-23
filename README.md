<<<<<<< HEAD
# VoiceCloning-App

Welcome to the VoiceCloning-App repository! This project enables you to clone voices using state-of-the-art machine learning models.



### Prerequisites

- Python 3.9  or strictly below 3.11
- [Git](https://git-scm.com/)
- [Virtualenv](https://virtualenv.pypa.io/)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/MathavanSG/VoiceCloning-App.git
    cd VoiceCloning-App
    ```

2. **Set Up a Virtual Environment**

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. **Install the Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download Embeddings**

    Due to a known bug, you must download the embeddings directly from Hugging Face. Please follow these instructions strictly to avoid errors:

    - Go to the [Hugging Face Embeddings](https://huggingface.co/) page.
    - Download the necessary embedding files.
    - Place the downloaded embedding files in the `bark` directory.

5. **Clone the Bark Repository**

    Ensure you maintain the folder hierarchy strictly:

    ```bash
    git clone https://github.com/suno-ai/bark.git
    ```

    Place the cloned `bark` directory within the `VoiceCloning-App` directory.

## Usage

1. **Activate the Virtual Environment**

    ```bash
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

2. **Run the Application**

    ```bash
    streamlit run app.py
    ```

    Follow the instructions in the Streamlit app to input the text you want to convert to speech.

## Folder Structure

To avoid errors, please maintain the following folder structure:
Bark_voices/speaker or in app.py--> Temp/Speaker
=======
![image](https://github.com/CyberSnowTeam/PitchDeck/assets/121884337/3ceddf14-c2db-4cc4-9008-95dedb9816a2)
# Pitch Deck PDF Analysis Chatbot

## Project Overview:
This project implements a chatbot capable of analyzing pitch deck PDF files uploaded by users. The chatbot extracts text from the PDFs, performs text analysis, and engages in conversation with users based on their queries regarding the pitch deck content.

## Technologies Used:
- **Flask:** Flask is a micro web framework for Python used to develop web applications.
- **Pytesseract:** Pytesseract is a Python wrapper for Google's Tesseract-OCR Engine, used for optical character recognition (OCR).
- **pdf2image:** This library is used to convert PDF files to images, facilitating text extraction from PDFs.
- **langchain:** langchain is a library for conversational AI and natural language processing tasks. It provides tools for text splitting, embeddings, vector stores, and conversation management.
- **OpenAIEmbeddings:** OpenAIEmbeddings is used for generating embeddings of text chunks.
- **FAISS:** FAISS is a library for efficient similarity search and clustering of dense vectors.
- **DuckDuckGoSearchAPIWrapper:** DuckDuckGoSearchAPIWrapper is utilized for fetching search results from DuckDuckGo search engine.
- **dotenv:** dotenv is used to load environment variables from a .env file.
- **Werkzeug:** Werkzeug is a WSGI (Web Server Gateway Interface) utility library for Python, providing necessary functionalities for handling file uploads.

## Setup Instructions:
1. Install Python and Flask if not already installed.
2. Install Tesseract-OCR and add its path to the system environment variables.
3. Clone the project repository
4. Navigate to the project directory.
5. Install the required Python packages by running: `pip install -r requirements.txt`.
6. Create a `.env` file and define the necessary environment variables.
7. Run the Flask application by executing `python app.py`.
8. Access the chatbot interface through a web browser.

## Usage:
- Visit the home route of the Flask application to access the chatbot interface.
- Upload a pitch deck PDF file for analysis.
- The chatbot will extract text from the PDF and initialize a conversation with the user.
- Users can ask queries related to the pitch deck content.
- The chatbot will respond based on the analyzed content of the uploaded PDF.

## Contributors:
- [Nithin](https://github.com/Nithin1522)-Project Lead & Developer
- [Mathavan S G](https://github.com/MathavanSG) - Machine Learning Engineer
- [Shruthi M] - Machine Learning Engineer

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
>>>>>>> 37006d7 (Add .gitignore)
