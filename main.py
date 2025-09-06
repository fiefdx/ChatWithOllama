import os
import sys
import json
import math
import time
import signal
import random
import tempfile
import queue
from queue import Queue, Empty
import threading
from threading import Thread
from multiprocessing import Process
from multiprocessing import Queue as PQueue

import pygame
import pygame_gui
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write, read
import numpy
import whisper
from ollama import Client
from TTS.api import TTS
import emoji

__version__ = "0.0.1"

os.environ['SDL_VIDEO_CENTERED'] = '1'
Q = queue.Queue(100)
TQ = PQueue(10)
StopSignal = "stop_signal"
StopPlay = "stop_play"
PlayAgain = "play_again"


theme_data = {
    "drop_down_menu": {
        "misc": {
            "expand_direction": "up"
        },
        "colours": {
            "normal_bg": "#25292e",
            "hovered_bg": "#35393e"
        }
    },
    "drop_down_menu.#selected_option": {
        "misc": {
            "border_width": "1",
            "open_button_width": "10"
        }
    },
    "text_box": {
        "colours": {
            "normal_text": "#ffffff"
        }
    },
    "text_entry_box": {
        "colours": {
            "normal_text": "#ffffff"
        }
    },
}
theme_file_path = "theme.json"
with open(theme_file_path, "w") as fp:
    json.dump(theme_data, fp, indent = 4)


class AudioPlayer(Process):
    def __init__(self, task_queue):
        Process.__init__(self)
        self.task_queue = task_queue
        self.stream = None

    def sig_handler(self, sig, frame):
        print("Caught signal: %s" % sig)

    def run(self):
        try:
            signal.signal(signal.SIGTERM, self.sig_handler)
            signal.signal(signal.SIGINT, self.sig_handler)
            self.tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar = False).to("cpu") # ("cuda")
            while True:
                try:
                    task = None
                    try:
                        task = self.task_queue.get(block = False)
                    except Empty:
                        pass
                    if task != StopSignal:
                        if task == StopPlay:
                            sd.stop()
                        elif task == PlayAgain:
                            if self.stream and self.stream.active:
                                sd.stop()
                            if os.path.exists("./output.wav"):
                                samplerate, data = read("./output.wav")
                                self.stream = sd.play(data, samplerate)
                        elif task is None:
                            if self.stream is None:
                                time.sleep(0.1)
                            elif self.stream and self.stream.active:
                                time.sleep(0.1)
                            else:
                                sd.stop()
                        else:
                            if os.path.exists("./output.wav"):
                                os.remove("./output.wav")
                            task = emoji.replace_emoji(task, "")
                            self.tts.tts_to_file(text = task, file_path = "./output.wav")
                            samplerate, data = read("./output.wav")
                            self.stream = sd.play(data, samplerate)
                    else:
                        break
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)


class StoppableThread(Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ThinkThread(StoppableThread):
    def __init__(self):
        StoppableThread.__init__(self)
        Thread.__init__(self)
        self.main_thread = None
        self.query = None
        self.model = whisper.load_model("base.en", device = 'cpu') # base.en, small.en
        self.client = Client("http://localhost:11434")
        self.chat_model = 'llama3.2:3b'
        self.context = []
        self.context_length = 4096

    def run(self):
        try:
            while True:
                if not self.stopped():
                    try:
                        if self.query is not None:
                            if self.query == "input.wav":
                                r = self.model.transcribe("input.wav")
                                self.main_thread.message = r["text"].strip()
                                self.main_thread.update_query = True
                            self.context.append({"role": "user", "content": self.main_thread.message})
                            if len(self.context) > self.context_length:
                                self.context.pop(0)
                            r = self.client.chat(model = self.chat_model, messages = self.context, think = False)
                            self.query = None
                            self.main_thread.response = r
                            self.main_thread.update_reply = True
                            self.context.append({"role": r.message.role, "content": r.message.content})
                            if len(self.context) > self.context_length:
                                self.context.pop(0)
                            TQ.put(str(r.message.content))
                        else:
                            time.sleep(0.1)
                    except Exception as e:
                        print(e)
                        time.sleep(0.1)
                else:
                    break
        except Exception as e:
            print(e)


def callback(indata, frames, time, status):
    if status:
        print(status, file = sys.stderr)
    Q.put(indata.copy())


class UserInterface(object):
    def __init__(self, think_thread):
        self.think_thread = think_thread
        self.think_thread.main_thread = self
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()
        self.window = pygame.display.set_mode((1280, 640)) # pygame.FULLSCREEN | pygame.SCALED) # pygame.RESIZABLE | pygame.SCALED)
        pygame.display.set_caption("Chat - v%s" % __version__)
        # pygame.display.set_icon(pygame.image.load("assets/image/icon.png"))
        pygame.joystick.init()
        self.clock = pygame.time.Clock()
        self.running = True
        self.status = "idling"
        self.font_command = pygame.font.SysFont('Arial', 40)
        self.font = pygame.font.SysFont('Arial', 20)
        self.sf = None
        self.message = None
        self.samplerate = 352800 # 44100
        self.response = None
        self.models = []
        models = self.think_thread.client.list()
        if models and "models" in models:
            for model in models["models"]:
                self.models.append(model["model"])
        self.models.sort()
        self.manager = pygame_gui.UIManager((1280, 640), theme_path = "theme.json")
        self.query_box = pygame_gui.elements.ui_text_entry_box.UITextEntryBox(relative_rect = pygame.Rect(10, 10, 1190, 70), manager = self.manager)
        self.update_query = False
        self.send_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((1200, 10), (70, 70)), text = 'Send', manager = self.manager)
        self.reply_box = pygame_gui.elements.ui_text_box.UITextBox("", relative_rect = pygame.Rect(10, 90, 1260, 500), manager = self.manager)
        self.update_reply = False
        self.stop_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((10, 595), (100, 40)), text = 'Stop', manager = self.manager)
        self.play_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((120, 595), (100, 40)), text = 'Play', manager = self.manager)
        self.discard_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((230, 595), (100, 40)), text = 'Discard', manager = self.manager)
        self.chat_models = pygame_gui.elements.ui_drop_down_menu.UIDropDownMenu(options_list = self.models, starting_option = self.think_thread.chat_model, relative_rect = pygame.Rect((340, 595), (300, 40)), expansion_height_limit = 300, manager = self.manager)
        self.record_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((650, 595), (100, 40)), text = 'Record', manager = self.manager)
        self.new_chat_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((1170, 595), (100, 40)), text = 'New Chat', manager = self.manager)

    def quit(self):
        TQ.put(StopSignal)
        self.think_thread.stop()
        self.running = False

    def play(self):
        print("play")
        TQ.put(PlayAgain)

    def stop(self):
        print("stop")
        TQ.put(StopPlay)

    def new(self):
        print("new")
        self.think_thread.context.clear()
        self.query_box.set_text("")
        self.reply_box.set_text("")

    def discard(self):
        print("discard")
        self.think_thread.context.pop(-1)
        self.think_thread.context.pop(-1)
        self.query_box.set_text("")
        self.reply_box.set_text("")

    def change_chat_model(self, model):
        print("change model to: %s" % model)
        self.think_thread.chat_model = model

    def change_message(self, text):
        self.message = text

    def send(self):
        self.think_thread.query = "new_message"

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.play_button:
                    self.play()
                elif event.ui_element == self.stop_button:
                    self.stop()
                elif event.ui_element == self.discard_button:
                    self.discard()
                elif event.ui_element == self.new_chat_button:
                    self.new()
                elif event.ui_element == self.send_button:
                    self.send()
                elif event.ui_element == self.record_button:
                    self.status = "waiting"
                    self.record_button.set_text("Record")
            elif event.type == pygame_gui.UI_BUTTON_START_PRESS:
                if event.ui_element == self.record_button:
                    if os.path.exists("input.wav"):
                        os.remove("input.wav")
                    self.sf = sf.SoundFile("input.wav", mode = 'x', samplerate = self.samplerate, channels = 2)
                    self.sd = sd.InputStream(samplerate = self.samplerate, channels = 2, callback = callback)
                    self.sd.start()
                    self.status = "recording"
                    self.message = None
                    self.record_button.set_text("Recording")
            elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == self.chat_models:
                    self.change_chat_model(event.text)
            elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                if event.ui_element == self.query_box:
                    self.change_message(event.text)
            self.manager.process_events(event)

    def render(self):
        if self.status == "recording":
            d = None
            try:
                d = Q.get(block = False)
            except Empty:
                pass
            if d is not None:
                self.sf.write(d)
                self.sf.flush()
        else:
            if self.sf is None:
                pass
            else:
                self.sf.close()
                self.sd.close()
        if self.status == "waiting":
            if self.message is None:
                self.think_thread.query = "input.wav"
                self.message = ""
        self.window.fill(0)
        if self.message is not None and self.update_query:
            self.query_box.set_text(self.message)
            self.update_query = False
        if self.response is not None and self.update_reply:
            self.reply_box.set_text(self.response.message.content)
            self.update_reply = False

    def run(self):
        while self.running:
            self.process_input()
            self.render()
            time_delta = self.clock.tick(60) / 1000.0 # for test
            self.manager.update(time_delta)
            self.manager.draw_ui(self.window)
            pygame.display.update()
        pygame.joystick.quit()

if __name__ == "__main__":
    p = AudioPlayer(TQ)
    p.start()
    think = ThinkThread()
    think.start()
    UserInterface = UserInterface(think)
    UserInterface.run()
    think.join()
    p.join()
    pygame.quit()