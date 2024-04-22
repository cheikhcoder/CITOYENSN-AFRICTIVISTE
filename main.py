import streamlit as st
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import asyncio
load_dotenv()

st.set_page_config(page_title="CitoyenSN", page_icon=":flag-sn:",layout="wide")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)



# st.title("Bienvenue a Djohodo")
st.markdown("<h2 style='text-align: center; '>Un Sénégal où chaque citoyen est informé, habilité <br> et capable d'exercer pleinement ses droits démocratiques</h2>", unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

col1,col2 = st.columns(2)
st.write("""Découvrez CitoyenSN - Prototype, votre portail pour des informations juridiques instantanées au Sénégal. Notre chatbot alimenté par l'intelligence artificielle vous offre un aperçu de notre solution, vous permettant d'explorer vos droits et la démocratie. Notez que ce site est un prototype et ne fournit pas d'assistance juridique officielle. Vos retours sont précieux pour nous aider à améliorer SénégaLoi et à le rendre plus utile pour la communauté sénégalaise..""")

 
def load_db(file, chain_type, k):
    async def load_and_process():
        # load documents 
        loader = PyPDFLoader(file)
        documents = loader.load()
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        # define embedding
        embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
        # create vector database from data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        # define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain. Memory is managed externally.
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY")),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa

    return asyncio.run(load_and_process()) 

async def chat_with_bot(cb, query):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cb, {"question": query, "chat_history": []})

def main():
    st.title("  Citoyen's bot ✨")
    
    col1,col2 = st.columns(2)
    with col1:
        st.write("Je votre assistant dédié. N'hésitez pas à me poser des questions sur les droits et la démocratie au Sénégal. Par exemple, vous pouvez me demander des informations sur les lois en vigueur, les procédures légales ou les organisations gouvernementales pertinentes. Je suis là pour vous fournir des réponses précises et utiles pour mieux comprendre vos droits.")
    with col2:
        original_image = Image.open("citoyen.png")
        st.image(original_image)

    # Load the database and chatbot
    cb = load_db("constition-senegal.pdf", "stuff", 4)

    query = st.text_input("Poser une question:")
    if st.button("Demander"):
        result = asyncio.run(chat_with_bot(cb, query))
        response = result["answer"]
        st.write("ChatBot:", response)

if __name__ == '__main__':
    main()
