from record import AudioHandler
import shutil

if __name__ == '__main__':
    print('Initializing recording device...')
    invalid = 0
    with AudioHandler() as audio:
        shutil.rmtree(audio.DATA_DIR, ignore_errors=True)
        print('Initialzed, ready to record')
        while not invalid:
            try:
                invalid = audio.listen()
            except KeyboardInterrupt:
                break
        print('Starting conversion to spectrograms')
        audio.convert_fileblock()
        audio.save_all_audio()
        print('Save successful')
