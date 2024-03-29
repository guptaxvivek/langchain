import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
st.title("LANGChain OPENAPI Tutorial")

input_text = st.text_input("Ask a query")

if input_text:
    llm = ChatOpenAI(temperature=0.8)
    st.write(llm.invoke(input_text).content)