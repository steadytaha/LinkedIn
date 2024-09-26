# tools
import re
import os
import glob
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from requests.cookies import cookiejar_from_dict
from tempfile import TemporaryDirectory
from urllib.parse import urlparse, parse_qs

# llamaindex
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex, Settings

# streamlit
import streamlit as st

# linkedin
from linkedin_api import Linkedin

# keys
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LINKEDIN_EMAIL = os.getenv("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.getenv("LINKEDIN_PASSWORD")


def response_generator(stream):
    """
    Generator that yields chunks of data from a stream response.
    
    Args:
        stream: The stream object from which to read data chunks.
    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

@st.cache_resource(show_spinner=False)
def load_data(documents: list[BytesIO], linkedin_data: dict = None) -> VectorStoreIndex:
    """
    Loads and indexes multiple PDF documents and optionally LinkedIn job information using Ollama and Llamaindex.
    
    Args:
        documents (list[BytesIO]): List of PDF documents to query.
        linkedin_data (dict, optional): Dictionary containing LinkedIn job information.
    
    Returns:
        VectorStoreIndex: An instance of VectorStoreIndex containing the indexed documents and embeddings.
    """

    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini")

    try:
        with TemporaryDirectory() as tmpdir:
            # Save all uploaded PDFs to the temporary directory
            for idx, document in enumerate(documents, start=1):
                temp_file_path = os.path.join(tmpdir, f'temp_{idx}.pdf')
                with open(temp_file_path, 'wb') as f:
                    f.write(document.getbuffer())

            with st.spinner(text="Loading and indexing the Streamlit docs. This may take a few minutes."):
                # Loading documents
                docs = SimpleDirectoryReader(tmpdir).load_data()

                # If LinkedIn data is provided, add it as a Document
                if linkedin_data:
                    linkedin_text = "\n".join([f"{key}: {value}" for key, value in linkedin_data.items()])
                    linkedin_doc = Document(text=linkedin_text, extra_info={"source": "LinkedIn Job Info"})
                    docs.append(linkedin_doc)

                # Embeddings | Query Container
                text_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=150)
                embed_model = OpenAIEmbedding(model_name="text-embedding-3-large", embed_batch_size=200)

                # Settings
                Settings.llm = llm
                Settings.embed_model = embed_model
                Settings.text_splitter = text_splitter
                Settings.transformations = [text_splitter]

                system_prompt = (
                    "Sen, işe alım konularında uzman, Türkçe konuşan ve "
                    "görevlerini iyi bilen bir işe alım asistanısın. "
                    "İnsan adaylarına CV'lerini analiz ederek geri bildirimler sağlıyorsun "
                    "ve işe alım sürecini destekliyorsun."
                )

                # Indexing DB
                index = VectorStoreIndex.from_documents(
                    docs,
                    embed_model=embed_model,
                    transformations=Settings.transformations
                )
    except Exception as e:
        st.error(f"An error occurred while processing the files: {e}")
        return None

    return index

def clean_html(html_text):
    if html_text:
        clean_text = re.sub(r'<[^>]+>', '', html_text)
        clean_text = re.sub(r'\n+', ' ', clean_text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        return clean_text
    return 'N/A'

def linkedin_info(url):
    """
    Fetches and returns LinkedIn job information based on the provided URL.
    
    Args:
        url (str): The LinkedIn Job URL.
    
    Returns:
        dict: A dictionary containing job information.
    """
    cookies = cookiejar_from_dict(
        {
            "liap": 'true',
            "li_at": 'AQEDATa5zJ0FYbp9AAABki2FqJsAAAGSUZIsm00AMFTRL4aG4ttkpYqd3MWsX4e_Ib8pDvLfele0UpwYH45l1X8BNKlnwuz8hPWF-Cj4bevn94bmKxWf9md8542I9x61zWk6Kiw5sRv_TLvZM7-P-8nc',
            "JSESSIONID": 'ajax:1102011791334138941',
        }
    )
    
    linkedin = Linkedin(LINKEDIN_EMAIL, LINKEDIN_PASSWORD, cookies=cookies)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    current_job_id = query_params.get('currentJobId', ['N/A'])[0]

    print(f"Current Job ID: {current_job_id}")

    job_dict = linkedin.get_job(current_job_id)
    job_info = {
        'Job Title': job_dict.get('title', 'N/A'),
        'Company Name': job_dict.get('companyDetails', {})
                         .get('com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany', {})
                         .get('companyResolutionResult', {}).get('name', 'N/A'),
        'Location': job_dict.get('formattedLocation', 'N/A'),
        'Workplace Type': job_dict.get('workplaceTypesResolutionResults', {})
                         .get('urn:li:fs_workplaceType:1', {}).get('localizedName', 'N/A'),
        'Remote Allowed': 'Yes' if job_dict.get('workRemoteAllowed') else 'No',
        'Description': clean_html(job_dict.get('description', {}).get('text', 'N/A')),
        'Apply URL': job_dict.get('applyMethod', {})
                        .get('com.linkedin.voyager.jobs.OffsiteApply', {})
                        .get('companyApplyUrl', 'N/A')
    }

    df = pd.DataFrame(job_info.items(), columns=['Field', 'Value'])
    print(df)
    
    return job_info

def main() -> None:
    """
    Controls the main chat application logic using Streamlit and Ollama.
    This function serves as the primary orchestrator of a chat application with the following tasks:

    1. Page Configuration: Sets up the Streamlit page's title, icon, layout, and sidebar using st.set_page_config.
    2. Model Selection: Manages model selection using st.selectbox and stores the chosen model in Streamlit's session state.
    3. Chat History Initialization: Initializes the chat history list in session state if it doesn't exist.
    4. Data Loading and Indexing: Calls the load_data function to create a VectorStoreIndex from the provided model name and LinkedIn info.
    5. Chat Engine Initialization: Initializes the chat engine using the VectorStoreIndex instance, enabling context-aware and streaming responses.
    6. Chat History Display: Iterates through the chat history messages and presents them using Streamlit's chat message components.
    7. User Input Handling:
          - Accepts user input through st.chat_input.
          - Appends the user's input to the chat history.
          - Displays the user's message in the chat interface.
    8. Chat Assistant Response Generation:
          - Uses the chat engine to generate a response to the user's prompt.
          - Displays the assistant's response in the chat interface, employing st.write_stream for streaming responses.
          - Appends the assistant's response to the chat history.
    """

    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with documents 💬")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    with st.sidebar:
        # LLM
        llm = OpenAI(model="gpt-4o-mini")

        url = st.sidebar.text_input("LinkedIn Job URL")

        # Data ingestion
        documents = st.file_uploader("Upload PDF files to query", type=['pdf'], accept_multiple_files=True)

        # LinkedIn toggle
        linkedin_toggle = st.toggle('Get LinkedIn Job Info', key='linkedin_toggle')

        # File processing                
        if st.button('Process files'):
            if documents:
                if linkedin_toggle:
                    if url:
                        linkedin_data = linkedin_info(url)
                        index = load_data(documents, linkedin_data=linkedin_data)
                    else:
                        st.warning("Please enter a LinkedIn Job URL in the sidebar.")
                        index = None
                else:
                    index = load_data(documents)
                
                if index is not None:
                    st.session_state.activate_chat = True
                else:
                    st.session_state.activate_chat = False
            else:
                st.warning("Please upload at least one PDF file before processing.")

    if st.session_state.activate_chat:
        # Initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize the chat engine if not already done
        if "chat_engine" not in st.session_state:
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("How can I help you?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Chat assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()
                with st.chat_message("assistant"):
                    stream = st.session_state.chat_engine.stream_chat(prompt)
                    response = st.write_stream(response_generator(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
    else:
        st.markdown(
            "<span style='font-size:15px;'><b>Upload PDF files to start chatting</b></span>",
            unsafe_allow_html=True
        )

if __name__=='__main__':
    main()
