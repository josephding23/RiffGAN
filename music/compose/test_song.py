from music.pieces.song.song import *


def test_griff():
    griff2 = GuitarRiff(measure_length=2,
                        degrees_and_types=[("I", "5"), ("I", "5"), ("II", "5"), ("V", "5"), ("III", "5"), ("I", "5"),
                                           ("III", "5"), ("VI", "5"),
                                           ("V", "M"), ("V", "M"), ("V", "M"), ("V", "M"), ("IV", "M"), ("IV", "M"),
                                           ("IV", "M")],
                        time_stamps=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                     0.5, 1,
                                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    griff2.add_notes_to_pm("G2", 120, 29)
    griff2.save("test3.mid")


def test_driff():
    drum_riff = DrumRiff(measure_length=2)
    drum_riff.set_pattern({"bass": "_________xxxxxxx", "tom": "_________1111___", "snare": "_____________xxx"})
    drum_riff.add_all_patterns_to_pm(120)
    drum_riff.save("drum1.mid")


def test_song():
    # Guitar track
    griff1 = GuitarRiff(measure_length=2,
                       degrees_and_types=[["I", "5"], ["I", "5"], ["II", "5"], ["V", "5"], ["III", "5"], ["I", "5"],
                                          ["III", "5"], ["VI", "5"], ["V", "5"], ["III", "5"], ["I", "5"],
                                          ["IV", ""], ["V", ""], ["IV", ""], ["III", ""], ["I", ""]],
                       time_stamps=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                    0.5, 1, 0.5, 0.5, 1,
                                    0.125, 0.125, 0.25, 0.5, 0.5])

    griff2 = GuitarRiff(measure_length=2,
                        degrees_and_types=[["I", "5"], ["I", "5"], ["II", "5"], ["V", "5"], ["III", "5"], ["I", "5"],
                                           ["III", "5"], ["VI", "5"],
                                           ["V", "M"], ["V", "M"], ["V", "M"], ["V", "M"], ["IV", "M"], ["IV", "M"],
                                           ["IV", "M"]],
                        time_stamps=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                     0.5, 1,
                                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    phrase11 = RhythmPhrase(8, ["G2", "major"], 120, 29, "guitar")
    phrase11.set_riffs([griff1, griff2])
    phrase11.set_arrangement([[0, "I"], [0, "V"], [0, "III"], [1, "I"]])

    track_guitar = Track(name="guitar",
                         bpm_list=[[0, 120]],
                         tonality_list=[[0, ['G', 'major']]],
                         is_drum=False,
                         instr_type='guitar')
    track_guitar.set_phrases([phrase11])
    track_guitar.set_arrangement([[0, 0], [0, 8]])
    track_guitar.add_phrases_to_pm()

    # Bass track
    briff1 = generate_briff_from_griff(griff1)
    briff2 = generate_briff_from_griff(griff2)

    phrase21 = RhythmPhrase(8, ["G1", "major"], 120, 33, "bass")
    phrase21.set_riffs([briff1, briff2])
    phrase21.set_arrangement([[0, "I"], [0, "V"], [0, "III"], [1, "I"]])

    track_bass = Track(name="bass",
                       bpm_list=[[0, 120]],
                       tonality_list=[[0, ['G', 'major']]],
                       is_drum=False,
                       instr_type='bass'
                       )
    track_bass.set_phrases([phrase21])
    track_bass.set_arrangement([[0, 0], [0, 8]])

    # Track Drum
    driff1 = DrumRiff(measure_length=1)
    driff1.set_pattern({"hi-hat": "ccccoccococco_cc", "snare": "____x__x_x__x___", "bass": "xxxxxxxx"})

    driff2 = DrumRiff(measure_length=2)
    driff2.set_pattern({"bass": "_________xxxxxxx", "tom": "_________1111___", "snare": "_____________xxx"})

    phrase31 = DrumPhrase(length=8, bpm=120)
    phrase31.set_riffs([driff1, driff2])
    phrase31.set_arrangement([0, 0, 0, 0, 0, 0, 1])

    track_drum = Track(name="drum",
                       bpm_list=[[0, 120]],
                       tonality_list=[],
                       is_drum=True,
                       instr_type=None)
    track_drum.set_phrases([phrase31])
    track_drum.set_arrangement([[0, 0], [0, 8]])

    song = Song("test_song")
    song.set_writer('troodeec')
    song.set_genre('Grunge')
    song.set_tracks([track_guitar, track_bass, track_drum])

    song.save_to_db()


def song_from_json():
    path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'
    song = create_song_from_json(path)
    song.add_tracks_to_pm()
    song.save_midi()
    song.play_it()


def get_composition():
    path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'
    song = create_song_from_json(path)
    info = song.get_all_phrases()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    test_song()
