# Chat With Ollama
This an ollama GUI client based on pygame_gui, pygame, openai-whisper, coqui-ai/TTS.

User can voice talk to llm models deployed on ollama and get voice response.

# Screenshots
### Application Window
![Alt text](/doc/window.png?raw=true "application window")

# Usage

python3.11 or python3.10, first using python3.11 -m pip install ./requirements.txt

then, run python3.11 ./main.py to start the application.

Pressed on "Record" button and hold it down for recording,

Release "Record" button will trigger STT to generate text query, and the response will go through TTS, then output through speakers.

"Stop" button for stoping the audio play.

"Play" button for playing the audio again.

"Discard" button for remove this context from the chat.

You can select the target model for chating with.

"New Chat" for staring a new chat.
