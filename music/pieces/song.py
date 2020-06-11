from music.custom_elements.riff import *
from music.pieces.phrase import *
from music.pieces.track import *
import pretty_midi
import pygame
import pyaudio
import wave
import os


class Song:
    def __init__(self, name):
        self.name = name
        self.tracks = []
        self.pm = None

        self.save_dir = '../../data/pieces/songs/'
        self.midi_path = self.save_dir + self.name + '.mid'
        self.wav_path = self.save_dir + self.name + '.wav'
        self.saved = False

    def add_track(self, track):
        assert isinstance(track, Track)
        self.tracks.append(track)

    def set_tracks(self, tracks):
        self.tracks = tracks

    def add_tracks_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()
        for track in self.tracks:
            instr = pretty_midi.Instrument(program=0, name=track.name, is_drum=track.is_drum)

            if track.is_drum:
                for phrase in track.phrases:
                    assert isinstance(phrase, DrumPhrase)

                    phrase_start = track.get_measure_start_time(phrase.start_measure)
                    riff_start = phrase_start
                    length_per_measure = get_measure_length(phrase.bpm)

                    for arrange in phrase.arrangement:
                        riff = phrase.riffs[arrange]
                        for part, pattern in riff.patterns.items():
                            if pattern is None:
                                continue
                            else:
                                assert isinstance(pattern, str)

                                total_num = len(pattern)
                                measure_length = get_measure_length(phrase.bpm) * riff.measure_length
                                unit_length = measure_length / total_num

                                for i in range(total_num):
                                    symbol = pattern[i]
                                    if symbol == '_':
                                        continue
                                    else:
                                        start_time, end_time = i * unit_length, (i + 1) * unit_length
                                        start_time += riff_start
                                        end_time += riff_start

                                        note = pretty_midi.Note(velocity=100, pitch=translate_symbol(part, symbol),
                                                                start=start_time, end=end_time)
                                        instr.notes.append(note)

                        riff_start += length_per_measure * riff.measure_length

            elif track.is_rhythm:
                for phrase in track.phrases:
                    assert isinstance(phrase, RhythmPhrase)
                    instr.program = phrase.instr

                    phrase_start = track.get_measure_start_time(phrase.start_measure)
                    riff_start = phrase_start
                    length_per_measure = get_measure_length(phrase.bpm)

                    for arrange in phrase.arrangement:
                        riff, riff_root_name = phrase.riffs[arrange[0]], arrange[1]
                        riff_root_dist = get_relative_distance(riff_root_name)

                        real_time_stamps = time_stamps_convert(riff.time_stamps, phrase.bpm)
                        for i in range(len(real_time_stamps)):
                            start_time, end_time = real_time_stamps[i]
                            start_time += riff_start
                            end_time += riff_start
                            if type(riff.velocity) == int:
                                velocity = riff.velocity
                            else:
                                assert type(riff.velocity) == list
                                velocity = riff.velocity[i]
                            chord = riff.chords[i]

                            for note_dist in chord:
                                note = pretty_midi.Note(velocity=velocity,
                                                        pitch=note_dist + phrase.root_note + riff_root_dist,
                                                        start=start_time, end=end_time)
                                instr.notes.append(note)

                        riff_start += length_per_measure * riff.measure_length

            self.pm.instruments.append(instr)

    def save(self):
        self.pm.write(self.midi_path)
        self.saved = True

    def play_it(self):

        assert self.saved

        freq = 44100
        bitsize = -16
        channels = 2
        buffer = 1024
        pygame.mixer.init(freq, bitsize, channels, buffer)
        pygame.mixer.music.set_volume(1)
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(self.midi_path)
        except:
            import traceback
            print(traceback.format_exc())
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30)

    def export_as_wav(self):

        do_ffmpeg_convert = False  # Uses FFmpeg to convert WAV files to MP3. Requires ffmpeg.exe in the script folder or PATH
        do_wav_cleanup = True  # Deletes WAV files after conversion to MP3
        sample_rate = 44100  # Sample rate used for WAV/MP3
        channels = 2  # Audio channels (1 = mono, 2 = stereo)
        buffer = 1024  # Audio buffer size
        mp3_bitrate = 128  # Bitrate to save MP3 with in kbps (CBR)
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

        play_music(self.midi_path)
        frames = []

        # Record frames while the song is playing
        while pygame.mixer.music.get_busy():
            frames.append(stream.read(buffer))

        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(self.wav_path, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(sample_rate)

        wave_file.writeframes(b''.join(frames))
        wave_file.close()


def play_music(music_file):

    try:
        pygame.mixer.music.load(music_file)

    except pygame.error:
        print("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
        return

    pygame.mixer.music.play()