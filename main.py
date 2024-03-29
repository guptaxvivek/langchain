from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# print(llm.generate(text))
# template = "{text}"
# prompt = PromptTemplate.from_template(template)
prompt = PromptTemplate.from_template(
    # input_variables=["product"],
    "What is a good name for a company that makes {product}?"
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# print(llm_chain.run(text))
print(llm_chain.run("eco-friendly water bottles"))

