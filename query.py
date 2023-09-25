pip install vertexai
pip install google-cloud-aiplatform google-cloud-discoveryengine langchain==0.0.236 pydantic==1.10.8 typing-inspect==0.8.0 typing_extensions==4.5.0 --upgrade --user

import streamlit as st
import os
import vertexai
from langchain.llms import VertexAI
from langchain.retrievers import GoogleCloudEnterpriseSearchRetriever
from langchain.chains import RetrievalQA

# Set the path to your service account JSON key file
SERVICE_ACCOUNT_KEY_PATH = "C:\\Users\\HP\\Downloads\\fl-datascience-research-stag-3321f553dcf8.json"

# Set your Google Cloud Project ID, Search Engine ID, Region, and Model
PROJECT_ID = "fl-datascience-research-stag"
SEARCH_ENGINE_ID = "fl-cogeco-podcast-transcri_1695652576173"
REGION = "us-central1"
MODEL = "text-bison@001"

# Initialize environment variables
os.environ["SEARCH_ENGINE_ID"] = SEARCH_ENGINE_ID
os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["REGION"] = REGION
os.environ["MODEL"] = MODEL

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH

# Initialize Vertex AI and Langchain
vertexai.init(project=PROJECT_ID, location=REGION)
llm = VertexAI(model_name=MODEL)

# Create a Streamlit app
st.title("Question Retrieval App")

# User input: Enter a question
search_query = st.text_input("Enter your question:")

# ...
if st.button("Retrieve Answer"):
    retriever = GoogleCloudEnterpriseSearchRetriever(
        project_id=PROJECT_ID, search_engine_id=SEARCH_ENGINE_ID
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    result = retrieval_qa.run(search_query)

    # Display the result without checking its type
    st.subheader("Answer:")
    st.write(result[1:])



# This will display the Streamlit app with an input box for questions and a button to retrieve answers.
