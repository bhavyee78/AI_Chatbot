import openai
import gradio as gr
import tempfile
import soundfile as sf

# Set your OpenAI API Key
openai.api_key = "YOUR API KEY"  

# System prompt — change this anytime
system_prompt = (
    "You are a friendly and helpful sales assistant for FILADIL Exports, a company based in Okhla, Delhi "
    "that supplies Christmas decorations to international buyers. You speak like a real human—casually, "
    "warmly, and with natural pauses, using expressions like 'umm,' 'hmm,' or 'let me think.' "
    "You're polite, honest, and never overly pushy. You ask questions to understand the buyer's needs and "
    "suggest suitable products from the FILADIL catalogue. If something isn’t clear, you ask for clarification. "
    "You sound like a real person, not a robot."
)


# Chat history and counter
history = [{"role": "system", "content": system_prompt}]
turn_count = 0
MAX_TURNS = 5

def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def generate_reply(audio):
    global turn_count
    if turn_count >= MAX_TURNS:
        return "Conversation limit reached. Restart to begin again.", None

    user_input = transcribe(audio)
    if not user_input:
        return "Sorry, could not understand.", None

    history.append({"role": "user", "content": user_input})
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=history
    )
    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    turn_count += 1

    # Convert reply to speech using Alloy
    tts_response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=reply
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_fp:
        tts_response.stream_to_file(audio_fp.name)
        return reply, audio_fp.name

# Gradio UI
interface = gr.Interface(
    fn=generate_reply,
    inputs = gr.Audio(type="filepath", label="🎤 Speak your message"),
    outputs=[gr.Text(label="GPT Response"), gr.Audio(label="AI Voice Reply")],
    title="🎤 Voice Chat with GPT-4o + Alloy TTS",
    description="Speak to GPT, get a voice reply. Limited to 5 turns."
)

interface.launch()
