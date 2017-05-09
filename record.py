# Builtin modules
import glob
import math
import os
import time
import logging

# 3rd party modules
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import scipy.io.wavfile
from scipy import signal

WIDTH_PIXELS = 652
HEIGHT_PIXELS = 656

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

def write_raw_data(data, rate, file_prefix):
    scipy.io.wavfile.write('{}_{}.wav'.format(file_prefix, 'audio'), rate, data)
    f, t, Sxx = signal.spectrogram(data, fs=rate, noverlap=250)
    fig = plt.pcolormesh(t, f, 10 * np.log10(1 + Sxx), cmap='inferno')
    my_dpi = plt.gcf().get_dpi()
    plt.gcf().set_size_inches(WIDTH_PIXELS/my_dpi, HEIGHT_PIXELS/my_dpi)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('{}_{}.jpeg'.format(file_prefix, 'spec'), bbox_inches='tight', pad_inches = 0, dpi=my_dpi)

class AudioHandler(object):

    DATA_DIR = 'raw_data'
    RATE = 16000
    INPUT_BLOCK_TIME = 0.03  # 30 ms
    CHANNELS = 1
    INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
    SENTENCE_DELAY = 1.1  # seconds
    MAX_SILENT_BLOCKS = math.ceil(SENTENCE_DELAY / INPUT_BLOCK_TIME)
    THRESHOLD = 40  # dB

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.save_counter = 0
        self.silent_blocks = 0
        self.listening = False
        self.audio = []

        # debug
        logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.stream.close()

    def open(self):
        device_index = self.find_input_device()

        self.stream = self.pa.open(format=pyaudio.paInt16,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              input_device_index=device_index,
                              frames_per_buffer=self.INPUT_FRAMES_PER_BLOCK)

    def find_input_device(self):
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)

            for keyword in ['mic','input']:
                if keyword in devinfo['name'].lower():
                    logging.debug('Found an input: Device {} - {}'.format(i, devinfo['name']))
                    return i

        logging.debug('No preferred input found; using default input device.')

    def save_block(self, snd_block):
        self.audio.append(snd_block)
        flat_block = np.hstack(snd_block)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        np.savetxt('{}/block{:0>8}.txt'.format(self.DATA_DIR, self.save_counter), flat_block)
        self.save_counter += 1

    def listen(self):
        try:
            raw_block = self.stream.read(self.INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)
            snd_block = np.fromstring(raw_block, dtype=np.int16)
        except Exception as e:
            logging.debug('Error recording: {}'.format(e))
            return

        amplitude = get_rms(snd_block)
        if amplitude > self.THRESHOLD:
            self.listening = True
            self.silent_blocks = 0 # reset counter
        else:
            self.silent_blocks += 1

        if self.listening:
            self.save_block(snd_block)
        if self.silent_blocks > self.MAX_SILENT_BLOCKS and self.listening:
            # remove last stored silent blocks
            for i in range(int(self.save_counter) - 1, int(self.save_counter) - self.MAX_SILENT_BLOCKS, -1):
                self.audio.pop()
                i = str(i).zfill(8)
                os.remove('{}/block{}.txt'.format(self.DATA_DIR, i))
            self.listening = False
            return True # done speaking

    def save_all_audio(self):
        flat_audio = np.hstack(self.audio)
        file_prefix = '{}/all_data'.format(self.DATA_DIR)
        write_raw_data(flat_audio, self.RATE, file_prefix)

    def convert_fileblock(self):
        for block_counter, filename in enumerate(glob.glob('{}/*.txt'.format(self.DATA_DIR))):
            block = np.loadtxt(filename, dtype=np.int16)
            t0 = time.time()
            file_prefix = '{}/{:0>8}'.format(self.DATA_DIR, block_counter)
            write_raw_data(block, self.RATE, file_prefix)
            logging.debug('Time to process block{}: {}'.format(block_counter, time.time() - t0))
