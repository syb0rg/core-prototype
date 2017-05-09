import glob
import sys
import os
import shutil
import wave
import time
import re
from threading import Thread
from PIL import Image
from random import randint

import scipy.io.wavfile
import pyaudio

DATA_DIR = 'raw_data'
LABELED_DIR = 'labeled_data'
answer = None

def cycle_wave(wf, chunk):
    while True:
        data = wf.readframes(chunk)
        if data == b'':
            wf.rewind()
            time.sleep(1)
            data = wf.readframes(chunk)
        yield data

def classify_files(chunk=1024):
    global answer
    join = os.path.join

    p = pyaudio.PyAudio()
    for filename in glob.glob(join(DATA_DIR, '*.wav')):
        wf = wave.open(filename, 'rb')
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        for data in cycle_wave(wf, chunk):
            if answer is not None:
                break
            stream.write(data)

        # don't know how to classify, skip sample
        if answer == '.':
            answer = None
            continue

        # sort spectogram based on input
        spec_filename = '{}_spec.jpeg'.format(re.findall(r'\d+', filename)[0])
        os.makedirs(join(LABELED_DIR, answer), exist_ok=True)
        shutil.copyfile(
            join(DATA_DIR, spec_filename),
            join(LABELED_DIR, answer, '{}_spec.jpeg'.format(randint(0, sys.maxsize)))
        )

        # reset answer field
        answer = None

        #stop stream
        stream.stop_stream()
        stream.close()

    #close PyAudio
    p.terminate()

if __name__ == '__main__':
    join = os.path.join
    try:
        img = Image.open(join(DATA_DIR, 'all_data_spec.jpeg'))
        img.show()
        num_files = len(glob.glob(join(DATA_DIR, '*.wav')))
        Thread(target = classify_files).start()
        for _ in range(0, num_files):
            answer = input("Enter letter of sound heard: ")
    except KeyboardInterrupt:
        sys.exit()
