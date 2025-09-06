import os
import sys
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

__version__ = "0.0.1"

os.environ['SDL_VIDEO_CENTERED'] = '1'
Q = queue.Queue(100)
TQ = PQueue(10)
StopSignal = "stop_signal"
StopPlay = "stop_play"
PlayAgain = "play_again"


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
            self.tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar = False).to("cuda")
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
                                time.sleep(0.5)
                            elif self.stream and self.stream.active:
                                time.sleep(0.5)
                            else:
                                sd.stop()
                        else:
                            if os.path.exists("./output.wav"):
                                os.remove("./output.wav")
                            self.tts.tts_to_file(text = task, file_path = "./output.wav")
                            samplerate, data = read("./output.wav")
                            self.stream = sd.play(data, samplerate)
                            # sd.wait()
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
        self.model = whisper.load_model("small.en") # base.en
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
                            r = self.model.transcribe("input.wav")
                            self.main_thread.message = r["text"].strip()
                            self.main_thread.update_query = True
                            self.context.append({"role": "user", "content": self.main_thread.message})
                            if len(self.context) > self.context_length:
                                self.context.pop(0)
                            r = self.client.chat(model = self.chat_model, messages = self.context)
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
        self.window = pygame.display.set_mode((1280, 640)) # 854-480, 746-420, 712-400, 640-360  pygame.FULLSCREEN | pygame.SCALED) # pygame.RESIZABLE | pygame.SCALED)
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
        self.manager = pygame_gui.UIManager((1280, 640))
        self.query_box = pygame_gui.elements.ui_text_box.UITextBox("", relative_rect = pygame.Rect(10, 10, 1100, 70), manager = self.manager)
        self.update_query = False
        self.reply_box = pygame_gui.elements.ui_text_box.UITextBox("", relative_rect = pygame.Rect(10, 90, 1260, 500), manager = self.manager)
        self.update_reply = False
        self.stop_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((10, 595), (100, 40)), text = 'Stop', manager = self.manager)
        self.play_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((120, 595), (100, 40)), text = 'Play', manager = self.manager)
        self.discard_button = pygame_gui.elements.UIButton(relative_rect = pygame.Rect((230, 595), (100, 40)), text = 'Discard', manager = self.manager)
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

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                elif event.key == pygame.K_SPACE:
                    if os.path.exists("input.wav"):
                        os.remove("input.wav")
                    self.sf = sf.SoundFile("input.wav", mode = 'x', samplerate = self.samplerate, channels = 2)
                    self.sd = sd.InputStream(samplerate = self.samplerate, channels = 2, callback = callback)
                    self.sd.start()
                    self.status = "recording"
                    self.message = None
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.status = "waiting"
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.play_button:
                    self.play()
                elif event.ui_element == self.stop_button:
                    self.stop()
                elif event.ui_element == self.discard_button:
                    self.discard()
                elif event.ui_element == self.new_chat_button:
                    self.new()
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
                # r = self.model.transcribe("input.wav")
                # self.message = r["text"]
                # self.think_thread.query = self.message

        self.window.fill(0)
        status = self.font_command.render(self.status, True, (255, 255, 255))
        self.window.blit(status, (1120, 20))
        if self.message is not None and self.update_query:
            # message = self.font.render("%s" % self.message, True, (255, 255, 255))
            # self.window.blit(message, (10, 60))
            self.query_box.set_text(self.message)
            self.update_query = False
        if self.response is not None and self.update_reply:
            self.reply_box.set_text(self.response.message.content)
            self.update_reply = False
            # lines = self.response.message.content.split("\n")
            # n = 0
            # for line in lines:
            #     striped = line.strip()
            #     if striped != "":
            #         if n == 0:
            #             reply = self.font.render("A: %s" % line, True, (255, 255, 255))
            #             self.window.blit(reply, (10, 90))
            #         else:
            #             reply = self.font.render("   %s" % line, True, (255, 255, 255))
            #             self.window.blit(reply, (10, 90 + n * 20))
            #         n += 1
        # pygame.display.update()

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