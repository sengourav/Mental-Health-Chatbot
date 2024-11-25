from langchain.chains import LLMChain
import chainlit as cl
import os
import openai
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper


openai_api_key = "sk-xxxx"
serper_api_key="yyyy"


VECTOR_STORE_PATH = "./vector_faiss/"

openai.api_key = openai_api_key
os.environ["SERPER_API_KEY"]=serper_api_key



@cl.cache
def instantiate():

    llm = ChatOpenAI()
    

    embeddings = OpenAIEmbeddings(show_progress_bar=True)

    # Load vector store
    # index=FAISS.load_local('https://drive.google.com/file/d/1-DgmC8rYns-IRmIGZZF1TcaHztX2i7vq/view?usp=sharing', embeddings, allow_dangerous_deserialization=True)
    index=FAISS.load_local(VECTOR_STORE_PATH, embeddings,allow_dangerous_deserialization=True)
    return llm , index

llm , vector_store= instantiate()

# Set up tools

search = GoogleSerperAPIWrapper()



tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
]


# Set up agent
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    verbose=True,
    memory=ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", k=5),
    return_intermediate_steps=True,
)


# Set up conversation chain
@cl.on_chat_start
def main():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are an empathetic mental health counselor focused on providing support and guidance."}],
    )
    

    prompt_template = """"Answer the question as a supportive and empathetic mental health counselor. Always maintain professional boundaries and encourage seeking professional help when appropriate.
    Context: {context}


    {chat_history}
    Human: {question}
    Assistant:"""

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template,
        # partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", k=5)
    handler = StdOutCallbackHandler()
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    cl.user_session.set("llm_chain", llm_chain)
    
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 10}
        )
    
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    retriever= cl.user_session.get("retriever")
    agent = cl.user_session.get("agent")
    
    # Get the user's message
    user_input = message.content
    docs = retriever.get_relevant_documents(user_input)
    msg = cl.Message(content="")
    await msg.send()
    # Generate response using LLMChain
    response = await llm_chain.arun(question=user_input, context=docs, callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)],agent=agent)

    
    for part in (response):
        if token := part or "":
            await msg.stream_token(token)

    await msg.update()

