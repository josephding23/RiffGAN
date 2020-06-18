from flask import Blueprint, render_template, redirect, url_for, request
from web.database.song import *
from music.custom_elements.riff.toolkit import *
from music.pieces.phrase.toolkit import *
from music.pieces.song.toolkit import *
from music.custom_elements.riff.drum_riff import examine_drum_patterns
import pygame
import time

riffs_bp = Blueprint('riffs', __name__, template_folder='templates', static_folder='static', url_prefix='/riffs')

freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(1)


@riffs_bp.route('/<riff_type>', methods=['GET'])
def get_riffs(riff_type):
    riffs = get_temp_riffs()
    print(riffs)
    return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type)


@riffs_bp.route('/delete/<riff_type>/<index>', methods=['POST'])
def delete_riff(riff_type, index):

    if request.method == 'POST':

        riffs = get_temp_riffs()
        phrases = get_temp_phrases()

        according_riffs_dict = {
            'griff': 'rhythm_guitar_phrase',
            'briff': 'rhythm_bass_phrase',
            'driff': 'drum_phrase'
        }
        riff_no_to_delete = riffs[riff_type][int(index)-1]['no']

        riff_no_in_use = get_all_used_riffs(phrases, according_riffs_dict[riff_type])

        if riff_no_to_delete in riff_no_in_use:
            error = 'Riff you tend to delete is in use.'
            return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                   error=error)

        riffs[riff_type].pop(int(index)-1)

        save_temp_riffs(riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/play/<riff_type>/<index>', methods=['POST'])
def play_riff(riff_type, index):
    if request.method == 'POST':
        riffs = get_temp_riffs()

        riff_info = riffs[riff_type][int(index)-1]

        if riff_type == 'griff':
            riff = parse_griff_json(riff_info)
            riff.add_notes_to_pm('E2', 120, 27)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        elif riff_type == 'briff':
            riff = parse_briff_json(riff_info)
            riff.add_notes_to_pm('E1', 120, 27)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        else:
            assert riff_type == 'driff'
            riff = parse_driff_json(riff_info)
            riff.add_all_patterns_to_pm(120)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/stop/<riff_type>', methods=['POST'])
def stop_riff(riff_type):
    if request.method == 'POST':

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/edit/<riff_type>/<index>', methods=['POST'])
def edit_riff(riff_type, index):
    if request.method == 'POST':

        riffs = get_temp_riffs()
        phrases = get_temp_phrases()
        tracks = get_temp_tracks()
        song = get_temp_song()

        if riff_type in ['griff', 'briff']:
            raw_length = request.form['edit_length_input']
            raw_degrees_and_types = request.form['edit_degrees_types_input']
            raw_timestamps = request.form['edit_timestamps_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            riffs[riff_type][int(index)-1] = {
                    'no': riffs[riff_type][int(index)-1]['no'],
                    'length': length,
                    'degrees_and_types': degrees_and_types,
                    'time_stamps': timestamps,
                    'raw_degrees_and_types': raw_degrees_and_types,
                    'raw_timestamps': raw_timestamps
                }

            refresh_all_phrases(phrases, riffs)
            refresh_all_tracks(tracks, phrases)
            refresh_all_tracks_in_song(song, tracks)

            save_temp_riffs(riffs)
            save_temp_phrases(phrases)
            save_temp_tracks(tracks)

            return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


        else:
            assert riff_type == 'driff'
            raw_length = request.form['edit_length_input']
            patterns_dict = {
                'hi-hat': '',
                'snare': '',
                'bass': '',
                'tom': '',
                'ride': '',
                'crash': '',
                'splash': ''
            }
            for part, _ in patterns_dict.items():
                patterns_dict[part] = request.form[f'edit_{part}_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            try:
                examine_drum_patterns(patterns_dict)
            except Exception:
                error = 'Invalid Drum Pattern'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                                       error=error)

            riffs[riff_type][int(index) - 1] = {
                'no': riffs[riff_type][int(index) - 1]['no'],
                'length': length,
                "patterns": patterns_dict
            }

            refresh_all_phrases(phrases, riffs)
            refresh_all_tracks(tracks, phrases)
            refresh_all_tracks_in_song(song, tracks)

            save_temp_riffs(riffs)
            save_temp_phrases(phrases)
            save_temp_tracks(tracks)

            return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/new/<riff_type>', methods=['POST'])
def new_riff(riff_type):

    riffs = get_temp_riffs()

    if request.method == 'POST':
        if riff_type in ['griff', 'briff']:
            raw_length = request.form['new_length_input']
            raw_degrees_and_types = request.form['new_degrees_types_input']
            raw_timestamps = request.form['new_timestamps_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type, error=error)

            riffs[riff_type].append(
                {
                    'no': get_largest_num_of_json(riffs[riff_type])+1,
                    'length': length,
                    'degrees_and_types': degrees_and_types,
                    'time_stamps': timestamps,
                    'raw_degrees_and_types': raw_degrees_and_types,
                    'raw_timestamps': raw_timestamps
                 }
            )
            save_temp_riffs(riffs)

        else:
            assert riff_type == 'driff'
            raw_length = request.form['edit_length_input']
            patterns_dict = {
                'hi-hat': '',
                'snare': '',
                'bass': '',
                'tom': '',
                'ride': '',
                'crash': '',
                'splash': ''
            }
            for part, _ in patterns_dict.items():
                patterns_dict[part] = request.form[f'edit_{part}_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            try:
                examine_drum_patterns(patterns_dict)
            except Exception:
                error = 'Invalid Drum Pattern'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            riffs[riff_type].append({
                'no': get_largest_num_of_json(riffs[riff_type]) + 1,
                'length': length,
                "patterns": patterns_dict
            })

            save_temp_riffs(riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))
