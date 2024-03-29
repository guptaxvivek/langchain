import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
st.title("Celebrity Search Results")

# Custom Prompt template
prompt = ChatPromptTemplate.from_template(
    template="Tell me about {name}"
)


person_memory = ConversationBufferMemory(input_key='name', memory_key='name_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='person_history')
decr_memory = ConversationBufferMemory(input_key='dob', memory_key='desc_history')


input_text = st.text_input("Ask a query")
llm = ChatOpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key='person', memory=person_memory)

second_prompt = ChatPromptTemplate.from_template(
    template="When did {person} born?"
)
chain2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_prompt = ChatPromptTemplate.from_template(
    template="Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(llm=llm, prompt=third_prompt, verbose=True, output_key='description', memory=decr_memory)

parent_chain = SequentialChain(chains=[chain,chain2,chain3], input_variables=['name'], output_variables=['person','dob','description'], verbose=True)


if input_text:
    st.write(parent_chain({"name":input_text}))
    with st.expander("Person Name"):
        st.info(person_memory.buffer)
        
    with st.expander("Major Events"):
        st.info(decr_memory.buffer)