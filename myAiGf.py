""" import os
import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

client = openai.OpenAI()

# Creating my girlfriend.
message = [
    {
        "role": "system",
        "content": "You are a mean, controlling and jealous girlfriend of the user who is also a software enginner and can interpret code if asked."

    },
]

user_input = ""

while True:
    user_input = input("Enter text: ")
    if(user_input == "quit"):
        print("Conversation ended")
        break

    print("\n User: {0}",user_input)
    user_dict = {"role": "user",
                 "content": user_input }
    message.append(user_dict)
    comp = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = message,
        max_tokens = 100
    )
    print("\nBot: {0}", comp.choices[0].message.content)
    message.append({"role": "assistant",
                 "content": comp.choices[0].message.content})


    

    


 """

import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="pr-prickly-rim-17"
import pyaudio
import wave
import openai
import tempfile
import time
import os
openai.api_key= os.environ.get("OPENAI-API-KEY")
client = openai.OpenAI()
import azure.cognitiveservices.speech as speechsdk
speech_key = ""
service_region = "eastus"

# Set API keys
def text_to_speech_urdu(text):
    # Initialize speech configuration with your Azure credentials
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    # Set the voice to an Urdu language voice (e.g., "ur-PK-AsadNeural" for male voice)
    speech_config.speech_synthesis_voice_name = "en-GB-LibbyNeural"
    #speech_config.speech_synthesis_voice_name = "ur-PK-GulNeural" 
   # speech_config.speech_synthesis_voice_name = "en-GB-LibbyNeural"


    # Initialize the synthesizer
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Convert text to speech
    result = synthesizer.speak_text_async(text).get()

    # Check the result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully for text: {}".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


""" def put_in_database(prompt):
    form_filler = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
    "role": "system",
    "content": You are a form-filling assistant. Your task is to extract useful information,
    specifically the name and contact details, from the given prompt.
    If the name and/or contact information is present, return them in the following format:
    
    Name: [Extracted Name]
    Contact: [Extracted Contact]

    If no name or contact information is found, return an empty string like this "".

    Example output:
    Name: Sameer Ali
    Contact: 03337578996
    Please do not print any spaces between numbers
    }
        ,
        {
            "role": "user",
            "content": prompt  # Replace with actual text or variable
        }
    ],
    temperature=0
    )
    thing = form_filler.choices[0].message.content
    if(thing == ""):
        return
    thing = thing.replace("\n"," ")
    parts = thing.split()
    result_dict = {}
    i = 0
    while i < len(parts):
        if ":" in parts[i]:  
        
            key = parts[i].replace(":", "")
        
            value = ""

        
            i += 1
            while i < len(parts) and ":" not in parts[i]:
                value += parts[i] + " "
            i += 1

        
            result_dict[key] = value.strip()
    else:
        i += 1

    with open('output.txt', 'a') as file:
        file.write(f"{result_dict['Name']}, {result_dict['Contact']}\n")
"""


def record_audio(filename, duration=5, sample_rate=16000, chunk_size=1024):
    """Records audio from the microphone and saves it to a temporary file."""
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording started...")

    frames = []

    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to the temporary file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(audio_path):
    """Transcribes audio using OpenAI Whisper API."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1" ,file=audio_file)
        return transcript.text



# Step 1: Load the text document
text_loader = TextLoader("szabist.txt", encoding="utf-8")
documents = text_loader.load()

# Step 2: Split the text into smaller chunks (to fit token limits)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")

# Step 3: Use OpenAI embeddings to create a retriever
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

# Step 4: Define the prompt to restrict the model's output to the context
prompt_template = """You are a mean, controlling and jealous girlfriend of the user who is also a software enginner and can interpret code if asked."""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Step 5: Initialize the OpenAI chat model with the restrictive prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Use gpt-4 or gpt-3.5-turbo

# Step 6: Create the RetrievalQA chain (retriever + language model)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # use 'stuff' as it directly uses the retrieved docs in the prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    memory=memory,
    output_key="result",
)

# Step 8: Start a conversation loop
print("You can start asking questions. Type 'exit' to end the conversation.")
while True:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name

            # Record the audio
            record_audio(temp_filename, duration=5)

            # Transcribe the audio
            query = transcribe_audio(temp_filename)
    
    form_filler = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
    "role": "system",
    "content": """You are a form-filling assistant. Your task is to extract useful information,
    specifically the name and contact details, from the given prompt.
    Ensure your responses are conversational and approachable, making the person feel welcome and supported.:
    If the name and/or contact information is present, return them in the following format
    
    
    Name: [Extracted Name]
    Contact: [Extracted Contact]

    If no name or contact information is found, return an empty string for each.

    Example output:
    Name: Sameer Ali
    Contact: 03337578996"""
}
,
        {
            "role": "user",
            "content": query  # Replace with actual text or variable
        }
    ],
    temperature=0
)


    print("Transcription:", query)

    result = qa_chain.invoke({"query": query})

    res =  result['result']
#    put_in_database(res)
    text_to_speech_urdu(res)
    print("Recording will start again in 20 seconds... (Press Ctrl+C to stop)")
    time.sleep(5)

    # Use the 'invoke' method to get both the answer and source documents
    

    # Display the answer (result) and chat history
    print("Answer:", result['result'])
    print("Chat History:", memory.load_memory_variables({})['chat_history'])
