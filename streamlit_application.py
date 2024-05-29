import streamlit as st
import streamlit_color
from PyPDF2 import PdfReader
from docx import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pandas as pd
import os

# Initialize streamlit_color
streamlit_color.main()

# Load OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = "sk-proj-77tDToQi6LUjoXbnRUHST3BlbkFJYsVLZsLn5ASXH74A1U2F"

st.markdown("<h1 style='text-align: center; color: grey;'>ResumeRank</h1>", unsafe_allow_html=True)

# Allow users to upload the first data file
cv1 = st.file_uploader("Upload Resume 1 :point_down:", type=["pdf", "docx"])
# Allow users to upload the second data file
cv2 = st.file_uploader("Upload Resume 2 :point_down:", type=["pdf", "docx"])

if cv1 is not None and cv2 is not None:
    st.write("Files successfully loaded!")

    def generate_text_from_pdf(pdf_file):
        try:
            pdfreader = PdfReader(pdf_file)
            raw_text = ''
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content
            return raw_text
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
            return None

    def generate_text_from_docx(docx_file):
        try:
            doc = Document(docx_file)
            raw_text = ''
            for para in doc.paragraphs:
                raw_text += para.text + '\n'
            return raw_text
        except Exception as e:
            st.error(f"An error occurred while reading the DOCX: {e}")
            return None

    def generate_document_search(file):
        try:
            if file.type == "application/pdf":
                raw_text = generate_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = generate_text_from_docx(file)
            else:
                st.error("Unsupported file type.")
                return None

            if raw_text:
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=800,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings()
                document_search = FAISS.from_texts(texts, embeddings)
                return document_search
            else:
                return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    candidates = ['candidate_1', 'candidate_2']
    st.title("Search Resumes for")
    Hr_question_1 = st.text_input("Enter your question below:", value='Which companies have you worked in and for how long?')
    Hr_question_2 = st.text_input("Enter your question below:", value='What are the Python libraries you know? Enumerate 3.')
    Hr_question_3 = st.text_input("Enter your question below:", value='Email and contact number.')

    # Initialize DataFrame to store results
    df = pd.DataFrame(columns=[Hr_question_1, Hr_question_2, Hr_question_3], index=candidates)

    # Generate FAISS embedding document search for each document
    document_search_1 = generate_document_search(file=cv1)
    document_search_2 = generate_document_search(file=cv2)

    if document_search_1 and document_search_2:
        document_search_list = [document_search_1, document_search_2]
        chain = load_qa_chain(OpenAI(model='gpt-3.5-turbo-instruct'), chain_type="stuff")

        for candidate, document_search in zip(candidates, document_search_list):
            for query in [Hr_question_1, Hr_question_2, Hr_question_3]:
                try:
                    docs = document_search.similarity_search(query)
                    df.loc[candidate, query] = chain.run(input_documents=docs, question=query)
                except Exception as e:
                    st.error(f"Error processing query '{query}' for {candidate}: {e}")

        st.title('Final output')
        st.write(df)
    else:
        st.error("Failed to generate document search for one or both CVs. Please check your input files and try again.")
