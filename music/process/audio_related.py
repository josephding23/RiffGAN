import pygame
import pyaudio
import wave


def export_as_wav(midi_path, wav_path):

    sample_rate = 44100  # Sample rate used for WAV/MP3
    channels = 2  # Audio channels (1 = mono, 2 = stereo)
    buffer = 1024  # Audio buffer size
    input_device = 2  # Which recording device to use. On my system Stereo Mix = 1
    bitsize = -16  # unsigned 16 bit

    pygame.mixer.init(sample_rate, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(1.0)

    # Init pyAudio
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True,
                        input_device_index=input_device, frames_per_buffer=buffer)

    try:
        pygame.mixer.music.load(midi_path)

    except pygame.error:
        print("Couldn't play %s! (%s)" % (midi_path, pygame.get_error()))
        return

    pygame.mixer.music.play()

    frames = []

    # Record frames while the song is playing
    while pygame.mixer.music.get_busy():
        frames.append(stream.read(buffer))

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(wav_path, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(sample_rate)

    wave_file.writeframes(b''.join(frames))
    wave_file.close()


def play_music(path):
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(path)
    except:
        import traceback
        print(traceback.format_exc())
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)