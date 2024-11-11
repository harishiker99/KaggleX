import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load the model and tokenizer from Hugging Face
model_name = "harishnair04/gemma_instruct_medtr_2b"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the conversation memory
memory = ConversationBufferMemory()

# Set up the conversation chain with LangChain
conversation_chain = ConversationChain(llm=model, memory=memory)

# Streamlit UI
st.title("Medical Transcription Conversational AI")
st.write("Interact with the Gemma-2B model for medical transcription tasks.")

# Input for user message
user_input = st.text_input("You:", placeholder="Ask your medical question here...")

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

# Handle user input
if user_input:
    # Generate model's response
    inputs = tokenizer(user_input, return_tensors="tf")
    outputs = model.generate(inputs.input_ids)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add to memory and session history
    memory.add_message(user_input, response)
    st.session_state["conversation"].append({"user": user_input, "bot": response})

# Display conversation history
for chat in st.session_state["conversation"]:
    st.write(f"You: {chat['user']}")
    st.write(f"Gemma: {chat['bot']}")