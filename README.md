REC Azamgarh Chatbot
A Streamlit-based chatbot designed to provide information about Rajkiya Engineering College (REC) Azamgarh using advanced natural language processing and vector search capabilities.
Developed By - Suraj Kumar Pandey

Overview
This application leverages a FAISS vector store with Hugging Face embeddings to efficiently retrieve relevant information from a pre-loaded text dataset. It uses a state-of-the-art language model to generate accurate and context-aware responses to user queries about REC Azamgarh.

Features
Answers questions related to REC Azamgarh based on provided text data.
Utilizes FAISS for fast similarity search and Hugging Face embeddings for text representation.
Offers a user-friendly interface powered by Streamlit.
Integrates a cutting-edge language model for natural language generation.
Prerequisites
Python 3.12 or higher
Required dependencies (listed in requirements.txt)
Usage
Ensure the rec_azamgarh_info.txt file is in the project directory, containing relevant REC Azamgarh information.
Run the Streamlit app:
bash

Copy
streamlit run bot.py
Open the provided local URL (e.g., http://localhost:8501) in your browser.
Type your question about REC Azamgarh in the text input field to receive a response!
Project Structure
bot.py: Main script containing the chatbot application logic.
rec_azamgarh_info.txt: Text file with information about REC Azamgarh.
requirements.txt: List of required Python packages.
.gitignore: Files and folders to exclude from version control.
Dependencies
Listed in requirements.txt:

streamlit
langchain-community
sentence-transformers
transformers
torch
faiss-cpu
Deployment
Local Deployment
Follow the usage instructions above.
Streamlit Cloud Deployment
Push your project to a GitHub repository.
Sign in to Streamlit Cloud and create a new app.
Link your GitHub repository and set bot.py as the main script.
Deploy the app (ensure all dependencies and data files are included).
Troubleshooting
Model Download Error: If the application fails to start due to model issues, ensure internet access and sufficient disk space. Pre-downloading required models locally may help.
Application Errors: Check the Streamlit logs for details and ensure all dependencies are correctly installed.
License
This project is licensed under the MIT License.
Developed By - Suraj Kumar Pandey
