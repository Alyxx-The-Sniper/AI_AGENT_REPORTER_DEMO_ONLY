import gradio as gr
import os
import requests
import uuid
from scipy.io.wavfile import write
import numpy as np
from typing import Annotated, Sequence, TypedDict, Optional, List

# --- Core Imports from Your Code ---
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

# --- Load Environment Variables ---
# Make sure you have a .env file with your OPENAI_API_KEY and DEEPINFRA_API_KEY
load_dotenv()

# ==============================================================================
# == YOUR LANGGRAPH AGENT CODE (with minor corrections and adjustments) ========
# ==============================================================================

# 1. State Definition
class AgentState(TypedDict):
    audio_path: str
    transcribed_text: str
    news_report: Annotated[Sequence[BaseMessage], add_messages]
    current_feedback: Optional[str]
    # 'approve' flag is handled by the UI buttons instead of within the state
    
# 2. Node and Tool Functions

def transcribe_fast(state: AgentState) -> AgentState:
    """Transcribes the audio file using the DeepInfra API."""
    print("---TRANSCRIBING AUDIO---")
    # Corrected key access from audio_path to 'audio_path'
    audio_path = state['audio_path']

    # Ensure API key is available
    DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
    if not DEEPINFRA_API_KEY:
        raise ValueError("DEEPINFRA_API_KEY not found in environment variables.")

    API_URL = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"
    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}"}
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"audio": (os.path.basename(audio_path), audio_file, 'audio/wav')}
            print(f"Uploading '{audio_path}' to DeepInfra for transcription...")
            
            response = requests.post(API_URL, headers=headers, files=files)
            response.raise_for_status()
            
            result = response.json()
            transcribed_text = result.get("text", "Transcription failed.")
            
            print("Transcription successful!")
            print(f"Result: {transcribed_text[:200]}...")

            # Update the state with the result
            state['transcribed_text'] = transcribed_text
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepInfra API: {e}")
        state['transcribed_text'] = f"Error during transcription: {e}"
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}")
        state['transcribed_text'] = "Error: Audio file not found."

    return state

# Define LLM and Tools
llm_openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

@tool
def revise(transcribed_text: str, last_report: str, feedback: str) -> str:
    """
    Revise the news report based on feedback.
    (Function signature modified for direct calls from Gradio)
    """
    print("---REVISING REPORT---")
    human_input = f"""
                You are a professional news editor.

                Here is the original transcribed text of an audio report:
                \"\"\"{transcribed_text}\"\"\"

                Here is the current draft of the news report that needs revision:
                \"\"\"{last_report}\"\"\"

                Here is the feedback on what to change:
                \"\"\"{feedback}\"\"\"

                Please provide a new, revised version of the news report that addresses the feedback.
                Ensure the revised report maintains clarity, correct grammar, and a professional style, while staying faithful to the original transcription.
                """
    
    response = llm_openai.invoke([HumanMessage(content=human_input)])
    return response.content

@tool
def save(final_text: str) -> str:
    """
    Save the latest news report to a text file.
    (Function signature modified for direct calls from Gradio)
    """
    print("---SAVING REPORT---")
    output_dir = "saved_reports"
    os.makedirs(output_dir, exist_ok=True)

    if not final_text:
        return "No report available to save."

    # Use a unique filename to avoid overwriting
    filename = os.path.join(output_dir, f"news_report_{uuid.uuid4().hex[:8]}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    print(f"Report saved to {filename}")
    return f"✅ Report successfully saved to: `{filename}`"


def ai_agent_reporter(state: AgentState) -> AgentState:
    """Generates the initial news report from the transcription."""
    print("---GENERATING INITIAL REPORT---")
    system_prompt = f"""
                    You are an expert news reporter. Your task is to write a clear, concise, and factual news report based on the following transcribed audio text.

                    Transcribed text:
                    \"\"\"{state['transcribed_text']}\"\"\"

                    Present the information as a professional news report.
                    """
    messages = [SystemMessage(content=system_prompt)]
    response = llm_openai.invoke(messages)
    
    # Append the new report to the message history
    state["news_report"].append(AIMessage(content=response.content))
    return state




# ==============================================================================
# == GRADIO APPLICATION ========================================================
# ==============================================================================

def process_audio(audio_input, chat_history):
    """
    Gradio handler: Transcribes audio and generates the first report.
    """
    if audio_input is None:
        return chat_history, None, "Please provide audio first.", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    chat_history = [] # Reset for new report
    
    # Save microphone input to a temporary WAV file
    if isinstance(audio_input, tuple):
        sample_rate, audio_data = audio_input
        # Use a unique filename for the temp audio
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, f"temp_mic_input_{uuid.uuid4().hex}.wav")
        write(audio_path, sample_rate, audio_data.astype(np.int16))
    else:
        audio_path = audio_input # Path is provided directly for file uploads

    # 1. Initialize state
    initial_state = AgentState(
        audio_path=audio_path,
        transcribed_text="",
        news_report=[],
        current_feedback=None
    )

    # 2. Run transcription
    state_after_transcription = transcribe_fast(initial_state)
    chat_history.append((None, f"**🎤 Transcription:**\n\n>_{state_after_transcription['transcribed_text']}_"))
    
    # 3. Generate the first report
    state_after_report = ai_agent_reporter(state_after_transcription)
    first_report = state_after_report["news_report"][-1].content
    chat_history.append((None, f"**✍️ First Draft:**\n\n{first_report}"))
    
    # Clean up temporary mic file
    if isinstance(audio_input, tuple) and os.path.exists(audio_path):
        os.remove(audio_path)

    return chat_history, state_after_report, "", gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def handle_revision(feedback_text, current_state, chat_history):
    """
    Gradio handler: Takes user feedback and calls the revise tool.
    """
    if not feedback_text:
        return chat_history, current_state, "Please enter your feedback for the revision."
    
    chat_history.append((feedback_text, None))

    # Get the necessary data from the state
    transcribed_text = current_state['transcribed_text']
    last_report = current_state['news_report'][-1].content
    
    # Call the revise tool directly
    revised_content = revise.func(transcribed_text, last_report, feedback_text)
    
    # Update state and chat
    current_state['news_report'].append(AIMessage(content=revised_content))
    chat_history.append((None, f"**🔄 Revised Draft:**\n\n{revised_content}"))
    
    return chat_history, current_state, "" # Clear feedback box

def handle_approval(current_state, chat_history):
    """
    Gradio handler: Saves the final report.
    """
    final_report_text = current_state['news_report'][-1].content
    
    # Call the save tool directly
    save_message = save.func(final_report_text)
    
    chat_history.append((None, save_message))
    
    # Disable buttons after saving
    return chat_history, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Audio News Reporter") as app:
    gr.Markdown("# Audio News Reporter Agent (Multi lagguage)")
    gr.Markdown("Upload or record audio, and the agent will generate a news report. You can then revise it with feedback or approve and save the final version.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Upload or Record Audio")
            generate_btn = gr.Button("Generate Report", variant="primary")
            gr.Markdown("---")
            gr.Markdown("### Controls")
            feedback_box = gr.Textbox(lines=3, label="Revision Feedback", placeholder="e.g., 'Make the tone more formal.' or 'Focus on the financial impact.'", visible=False)
            
            with gr.Row():
                revise_btn = gr.Button("Revise Report", variant="secondary", visible=False)
                approve_btn = gr.Button("Approve & Save", variant="primary", visible=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Agent Output", height=600)
            
    # State object to hold the agent's state across interactions
    agent_state = gr.State()
    
    # Wire up the components
    generate_btn.click(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, agent_state, feedback_box, feedback_box, revise_btn, approve_btn]
    )
    
    revise_btn.click(
        fn=handle_revision,
        inputs=[feedback_box, agent_state, chatbot],
        outputs=[chatbot, agent_state, feedback_box]
    )

    approve_btn.click(
        fn=handle_approval,
        inputs=[agent_state, chatbot],
        outputs=[chatbot, revise_btn, approve_btn, generate_btn]
    )

if __name__ == "__main__":
    app.launch(share=True)