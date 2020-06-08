from src.custom_elements.riff import *
from src.pieces.phrase import *
from src.pieces.track import *
import pretty_midi


class Song:
    def __init__(self, name):
        self.name = name
        self.tracks = []
        self.pm = pretty_midi.PrettyMIDI()

    def add_track(self, track):
        assert isinstance(track, Track)
        self.tracks.append(track)

    def set_tracks(self, tracks):
        self.tracks = tracks

    def add_tracks_to_pm(self):
        for track in self.tracks:
            instr = pretty_midi.Instrument(program=0, name=track.name)

            for phrase in track.phrases:
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

    def save(self, save_path):
        self.pm.write(save_path)
