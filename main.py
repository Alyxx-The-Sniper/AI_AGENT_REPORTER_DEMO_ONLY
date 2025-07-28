import gradio as gr
import os
import requests
import base64
from typing import TypedDict, Optional, Sequence, Annotated
from operator import add

# --- LangChain Core Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# Instantiate the language model with the API key
llm_openai = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# A separate LLM instance for vision tasks (same model, different settings)
vision_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)



class AgentState(TypedDict):
    """Defines the state of our agent."""
    audio_path: Optional[str]
    image_path: Optional[str]
    transcribed_text: Optional[str]
    image_description: Optional[str]
    news_report: Annotated[Sequence[BaseMessage], add]
    current_feedback: Optional[str]



import os
from openai import OpenAI
import requests

def transcribe_fast(state: dict) -> dict:
    """Transcribes the audio file with OpenAI Whisper."""
    print("---TRANSCRIBING AUDIO (OpenAI Whisper)---")
    audio_path = state.get("audio_path")
    if not audio_path:
        print("No audio path found in state.")
        return state

    client = OpenAI(api_key=OPENAI_API_KEY)
    if not client.api_key:
        print("OPENAI_API_KEY environment variable not set.")
        state["transcribed_text"] = "Error: Missing OPENAI_API_KEY."
        return state

    try:
        with open(audio_path, "rb") as audio_file:
            print(f"Uploading '{audio_path}' to OpenAI Whisper...")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            state["transcribed_text"] = transcription.strip()
            print("Transcription successful!")
    except Exception as e:
        print(f"Transcription error: {e}")
        state["transcribed_text"] = f"Error during transcription: {e}"

    return state


####################################################
@tool
def describe_image_node(state: AgentState) -> AgentState:
    """Describes the image like a news anchor, reporter, and journalist using a multimodal LLM if an image_path is present."""
    image_path = state.get('image_path')
    if not image_path:
        print("---NO IMAGE PROVIDED, SKIPPING DESCRIPTION---")
        state['image_description'] = None
        return state

    print("---DESCRIBING IMAGE---")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        message = HumanMessage(
            content=[
                {"type": "text", "text": "You are a new's anchor, reporter, and journalist. Describe this image for a news report. Focus on key objects, people, actions, and the overall setting. Be factual and objective."},
                # CORRECTED: The value for 'image_url' must be an object with a 'url' key.
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = vision_llm.invoke([message])
        description = response.content
        print(f"Image Description Generated: {description[:150]}...")
        state['image_description'] = description

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        state['image_description'] = "Error: Image file not found."
    except Exception as e:
        print(f"An error occurred during image description: {e}")
        state['image_description'] = f"Error during image description: {e}"

    return state


def ai_agent_reporter(state: AgentState) -> AgentState:
    """Generates the news report from transcription and/or image description."""
    print("---GENERATING INITIAL NEWS REPORT---")
    context_parts = ["You are an expert news reporter. Your task is to write a clear, concise, and factual news report based on the following information.\n"]

    transcribed_text = state.get('transcribed_text')
    image_description = state.get('image_description')

    if not transcribed_text and not image_description:
        report_content = "No input provided. Please provide an audio file or an image to generate a report."
    else:
        if transcribed_text:
            context_parts.append(f"--- Transcribed Audio ---\n\"{transcribed_text}\"\n")
        if image_description:
            context_parts.append(f"--- Image Description ---\n\"{image_description}\"\n")
        
        context_parts.append("Present the information as a professional news report. If you have both audio and an image description, synthesize them into a single, coherent story.")
        
        prompt = SystemMessage(content="\n".join(context_parts))
        response = llm_openai.invoke([prompt])
        report_content = response.content

    state["news_report"].append(AIMessage(content=report_content))
    return state

@tool
def revise(state: AgentState) -> AgentState:
    """Revise the news report based on the latest feedback."""
    print("---REVISING NEWS REPORT---")
    transcribed = state.get("transcribed_text", "Not available.")
    latest_report_msg = next((msg for msg in reversed(state["news_report"]) if isinstance(msg, AIMessage)), None)
    last_report = latest_report_msg.content if latest_report_msg else "No report yet."
    feedback = state.get("current_feedback", "No feedback provided.")

    prompt = f"""You are a professional news editor.
Revise the news report to address the feedback. Ensure clarity, grammar, and style are improved, while staying faithful to the original transcription.

**Original Transcription:**
"{transcribed}"

**Current Draft of News Report:**
"{last_report}"

**Latest Human Feedback:**
"{feedback}"

Provide only the full, revised news report as your response.
"""
    response = llm_openai.invoke([HumanMessage(content=prompt)])
    state["news_report"].append(response)
    return state

@tool
def save(state: AgentState) -> str:
    """Save the latest news report to a text file."""
    print("---SAVING REPORT---")
    output_dir = "saved_reports"
    os.makedirs(output_dir, exist_ok=True)

    latest_report_msg = next((msg for msg in reversed(state["news_report"]) if isinstance(msg, AIMessage)), None)
    if not latest_report_msg:
        return "No report available to save."

    final_text = latest_report_msg.content
    filename = os.path.join(output_dir, "news_report.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    print(f"Report saved to: {filename}")
    return f"✅ News report successfully saved to: {filename}"

# ==============================================================================
#  GRADIO UI AND HANDLER FUNCTIONS
# ==============================================================================

def get_latest_report(state: AgentState) -> str:
    """Helper to extract the most recent AI message from the state."""
    if not state or not state.get("news_report"):
        return "No report generated yet."
    
    latest_ai_messages = [msg for msg in state["news_report"] if isinstance(msg, AIMessage)]
    return latest_ai_messages[-1].content if latest_ai_messages else "Waiting for report generation..."

def generate_report_workflow(audio_filepath, image_filepath):
    """Handles the initial report generation."""
    if not audio_filepath and not image_filepath:
        # MODIFIED: Added another empty string for the new transcription output
        return None, "Upload a file to begin.", "", gr.update(visible=False), "Upload a file to begin."
    
    # 1. Initialize state
    state = AgentState(
        audio_path=audio_filepath,
        image_path=image_filepath,
        news_report=[],
    )

    # 2. Run sequential processing
    if state.get('audio_path'):
        state = transcribe_fast(state)
    if state.get('image_path'):
        state = describe_image_node.func(state)
    
    state = ai_agent_reporter(state)
    
    # 3. Get results and update UI
    report_text = get_latest_report(state)
    # NEW: Get the transcribed text from the state
    transcribed_text = state.get("transcribed_text", "No audio was provided for transcription.")
    
    # MODIFIED: Added transcribed_text to the return values
    return state, report_text, transcribed_text, gr.update(visible=True), "Initial report generated. Ready for feedback."

def revise_report_workflow(feedback: str, current_state: AgentState):
    """Handles the revision workflow."""
    if not feedback.strip():
        return current_state, get_latest_report(current_state), "⚠️ Please provide feedback to revise the report."
    
    current_state["current_feedback"] = feedback
    revised_state = revise.func(current_state)
    
    report_text = get_latest_report(revised_state)
    return revised_state, report_text, "✅ Report revised. You can provide more feedback or save."

def save_report_workflow(current_state: AgentState):
    """Handles the save workflow."""
    if not current_state:
        return "No report available to save."
    status_message = save.func(current_state)
    return status_message


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Multi-Modal News Reporter") as app:
    agent_state = gr.State(value=None)
    
    gr.Markdown("# 🤖 Multi-Modal News Reporter Agent")
    gr.Markdown("Upload an audio recording and/or an image to generate a news report. You can then provide feedback to revise the report.")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Field Reporter's Audio")
        image_input = gr.Image(type="filepath", label="Photojournalist's Image")

    generate_btn = gr.Button("Generate Initial Report", variant="primary", scale=1)
    
    gr.Markdown("---")
    
    # NEW: Added a Row to display transcription and report side-by-side
    with gr.Row():
        transcription_display = gr.Textbox(
            label="📝 Raw Transcription", 
            interactive=False, 
            lines=15,
            show_copy_button=True
        )
        report_display = gr.Textbox(
            label="📰 Generated News Report", 
            interactive=False, 
            lines=15, 
            show_copy_button=True
        )
    
    with gr.Group(visible=False) as revision_group:
        gr.Markdown("### ✍️ Provide Feedback to Revise Report")
        feedback_input = gr.Textbox(label="Your Feedback", placeholder="e.g., 'Make the tone more urgent.' or 'Clarify the second paragraph.'")
        with gr.Row():
            revise_btn = gr.Button("Revise Report")
            save_btn = gr.Button("Save & Finish", variant="stop")
            
    status_display = gr.Textbox(label="Status", interactive=False)
    
    # --- Event Handler Logic ---
    generate_btn.click(
        fn=generate_report_workflow,
        inputs=[audio_input, image_input],
        # MODIFIED: Added the new transcription_display to the outputs
        outputs=[agent_state, report_display, transcription_display, revision_group, status_display]
    )
    
    revise_btn.click(
        fn=revise_report_workflow,
        inputs=[feedback_input, agent_state],
        outputs=[agent_state, report_display, status_display]
    )
    
    save_btn.click(
        fn=save_report_workflow,
        inputs=[agent_state],
        outputs=[status_display]
    )

if __name__ == "__main__":
    app.launch()