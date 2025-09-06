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


class AudioPlayer(Process):
    def __init__(self, task_queue):
        Process.__init__(self)
        self.task_queue = task_queue

    def sig_handler(self, sig, frame):
        print("Caught signal: %s" % sig)

    def run(self):
        try:
            signal.signal(signal.SIGTERM, self.sig_handler)
            signal.signal(signal.SIGINT, self.sig_handler)
            self.tts = TTS("tts_models/en/ljspeech/glow-tts", progress_bar = False).to("cuda")
            while True:
                task = self.task_queue.get()
                if task != StopSignal:
                    if os.path.exists("./output.wav"):
                        os.remove("./output.wav")
                    self.tts.tts_to_file(text = task, file_path = "./output.wav")
                    samplerate, data = read("./output.wav")
                    sd.play(data, samplerate)
                    sd.wait()
                else:
                    break
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
        self.context_length = 1024

    def run(self):
        try:
            while True:
                if not self.stopped():
                    try:
                        if self.query is not None:
                            r = self.model.transcribe("input.wav")
                            self.main_thread.message = r["text"].strip()
                            self.context.append({"role": "user", "content": self.main_thread.message})
                            if len(self.context) > self.context_length:
                                self.context.pop(0)
                            r = self.client.chat(model = self.chat_model, messages = self.context)
                            self.query = None
                            self.main_thread.response = r
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

    def quit(self):
        TQ.put(StopSignal)
        self.think_thread.stop()
        self.running = False

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
        status = self.font_command.render(self.status, True, (0, 0, 255))
        self.window.blit(status, (10, 10))
        if self.message is not None:
            message = self.font.render("Q: %s" % self.message, True, (255, 255, 255))
            self.window.blit(message, (10, 60))
        if self.response is not None:
            reply = self.font.render("A: %s" % self.response.message.content, True, (255, 255, 255))
            self.window.blit(reply, (10, 90))
        pygame.display.update()

    def run(self):
        while self.running:
            self.process_input()
            self.render()
            self.clock.tick(60) # for test
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