import os
import imaplib
import email
import json
import chromadb
from dotenv import load_dotenv
from google import genai
from chromadb.utils import embedding_functions

chroma_client = chromadb.Client()
# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

dictionary = {}
with open('password.json', 'r') as file:
    data = json.load(file)
# Connect to the Gmail IMAP server
imap_host = 'imap.gmail.com'
email_user = data['username']
email_pass = data['password']

mail = imaplib.IMAP4_SSL(imap_host)
mail.login(email_user, email_pass)
mail.select('inbox')

def store_email(frm, subject, body, count):
    frm = frm.split("<")
    frm = frm[0]
    if '0' not in dictionary:
        dictionary.setdefault(count, {}).setdefault('From', f"{frm}")
        dictionary.setdefault(count, {}).setdefault('Subject', f"{subject}")
        dictionary.setdefault(count, {}).setdefault('body', f"{body}")
    else:
        dictionary.setdefault(count, {}).setdefault('From', f"{frm}")
        dictionary.setdefault(count, {}).setdefault('Subject', f"{subject}")
        dictionary.setdefault(count, {}).setdefault('body', f"{body}")
    # Create a directory 'email_data'...'dict_file' will be the name of the file being created
    with open('./email_data/dict_file', 'w') as dictfile:
        json.dump(dictionary, dictfile, indent=4)

# Search and fetch emails
status, messages = mail.search(None, 'ALL')
email_ids = messages[0].split()
count = 0
for email_id in email_ids[-10:]:  # Limit to the first 10 emails
    status, msg_data = mail.fetch(email_id, '(RFC822)')
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            subject = msg['subject']
            from_email = msg['from']
            body = None

            # Extract the email body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain' and 'attachment' not in part.get('Content-Disposition', ''):
                        body = part.get_payload(decode=True).decode('utf-8')
                        break
            else:
                body = msg.get_payload(decode=True).decode('utf-8')

            print(f"From: {from_email}\nSubject: {subject}\nMessage: {body}\n")
            store_email(from_email, subject, body, count)
            count = count + 1

mail.logout()
# Initialize the Chroma client with persistence
chroma = chromadb.PersistentClient(path="chroma_persistent_storage")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=default_ef)

client = genai.Client(api_key=gemini_api_key)


# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename:
            with open(
                    os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./email_data"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i + 1}", "text": chunk})

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    for idx, document in enumerate(results["documents"][0]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        print(
            f" For the query: {question}, \n (ID: {doc_id}, Distance: {distance})"
        )
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    all = f'{prompt}\n\n{question}'

    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=all)

    answer = response
    return answer

def display():
    running = True
    while running:
        question = input('What do you want to ask?(N/n to exit)\n')
        if question.upper() == "N":
            running = False
        else:
            relevant_chunks = query_documents(question)
            answer = generate_response(question, relevant_chunks)
            text_content = answer.candidates[0].content.parts[0].text
            print(text_content)
display()