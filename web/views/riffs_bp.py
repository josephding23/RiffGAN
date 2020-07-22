from flask import Blueprint, render_template, redirect, url_for, request
from web.database.song import *
from web.database.riff import *
from music.custom_elements.rhythm_riff.toolkit import *
from music.custom_elements.modified_riff.modified_riff import *
from music.pieces.song.toolkit import *
from music.pieces.track.toolkit import *
from music.custom_elements.drum_riff.drum_riff import examine_drum_patterns
from util.riff_modification import modify_riff
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
    modified_riffs = get_temp_modified_riffs()
    existed_riffs = get_all_existed_riffs(riff_type)
    return render_template('riffs/' + riff_type + '.html',
                           riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                           riff_type=riff_type,
                           existed_riffs=existed_riffs,
                           t=time.time())


@riffs_bp.route('/delete/<riff_type>/<index>', methods=['POST'])
def delete_riff(riff_type, index):

    if request.method == 'POST':

        riffs = get_temp_riffs()
        modified_riffs = get_temp_modified_riffs()

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
            existed_riffs = get_all_existed_riffs(riff_type)
            return render_template('riffs/' + riff_type + '.html',
                                   riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                   riff_type=riff_type,
                                   error=error, existed_riffs=existed_riffs)

        riffs[riff_type].pop(int(index)-1)

        save_temp_riffs(riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/delete_modified/<riff_type>/<index>', methods=['POST'])
def delete_modified_riff(riff_type, index):

    if request.method == 'POST':

        modified_riffs = get_temp_modified_riffs()

        # phrases = get_temp_phrases()

        modified_riff_no_to_delete = modified_riffs[riff_type][int(index)-1]['no']

        '''
        riff_no_in_use = get_all_used_riffs(phrases, according_riffs_dict[riff_type])

        if riff_no_to_delete in riff_no_in_use:
            error = 'Riff you tend to delete is in use.'
            existed_riffs = get_all_existed_riffs(riff_type)
            return render_template('riffs/' + riff_type + '.html',
                                   riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                   riff_type=riff_type,
                                   error=error, existed_riffs=existed_riffs)
        '''
        modified_riffs[riff_type].pop(int(index)-1)

        save_temp_modified_riffs(modified_riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/play/<riff_type>/<index>', methods=['POST'])
def play_riff(riff_type, index):
    if request.method == 'POST':
        riffs = get_temp_riffs()

        riff_info = riffs[riff_type][int(index)-1]

        if riff_type == 'griff':
            riff = parse_griff_json(riff_info)
            riff.add_notes_to_pm('E2', 120, 29)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        elif riff_type == 'briff':
            riff = parse_briff_json(riff_info)
            riff.add_notes_to_pm('E1', 120, 33)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        else:
            assert riff_type == 'driff'
            riff = parse_driff_json(riff_info)
            riff.add_all_patterns_to_pm(120)
            riff.save_midi(f'temp_{riff_type}_{index}')
            riff.play_with_no_init()

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/play_modified/<riff_type>/<index>', methods=['POST'])
def play_modified_riff(riff_type, index):
    if request.method == 'POST':
        modified_riffs = get_temp_modified_riffs()

        modified_riff_info = modified_riffs[riff_type][int(index)-1]

        if riff_type == 'griff':
            modified_riff = parse_modified_griff_json(modified_riff_info)
            modified_riff.add_notes_to_pm(29)
            modified_riff.save_midi(f'temp_{riff_type}_{index}_{modified_riff.option}')
            modified_riff.play_with_no_init()

        else:
            assert riff_type == 'briff'
            modified_riff = parse_modified_briff_json(modified_riff_info)
            modified_riff.add_notes_to_pm(33)
            modified_riff.save_midi(f'temp_{riff_type}_{index}_{modified_riff.option}')
            modified_riff.play_with_no_init()

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/stop/<riff_type>', methods=['POST'])
def stop_riff(riff_type):
    if request.method == 'POST':

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/stop_modified/<riff_type>', methods=['POST'])
def stop_modified_riff(riff_type):
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
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

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
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                examine_drum_patterns(patterns_dict)
            except Exception:
                error = 'Invalid Drum Pattern'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

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
    modified_riffs = get_temp_modified_riffs()

    if request.method == 'POST':
        if riff_type in ['griff', 'briff']:
            raw_length = request.form['new_length_input']
            raw_degrees_and_types = request.form['new_degrees_types_input']
            raw_timestamps = request.form['new_timestamps_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html',
                                       riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                       riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                degrees_and_types = get_degrees_and_types_from_raw(raw_degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html',
                                       riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                       riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
            except Exception:
                error = 'Invalid Timestamps format.'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html',
                                       riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                       riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

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
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html',
                                       riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                       riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            try:
                examine_drum_patterns(patterns_dict)
            except Exception:
                error = 'Invalid Drum Pattern'
                existed_riffs = get_all_existed_riffs(riff_type)
                return render_template('riffs/' + riff_type + '.html',
                                       riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                       riff_type=riff_type,
                                       error=error, existed_riffs=existed_riffs)

            riffs[riff_type].append({
                'no': get_largest_num_of_json(riffs[riff_type]) + 1,
                'length': length,
                "patterns": patterns_dict
            })

            save_temp_riffs(riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/save/<riff_type>/<index>', methods=['POST'])
def save_riff(riff_type, index):
    if request.method == 'POST':
        riffs = get_temp_riffs()
        modified_riffs = get_temp_modified_riffs()
        name = request.form['save_name_input']

        if exists_riff(name, riff_type):
            error = f'Riff {name} already exists! Please choose another'
            existed_riffs = get_all_existed_riffs(riff_type)
            return render_template('riffs/' + riff_type + '.html',
                                   riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                                   riff_type=riff_type,
                                   error=error, existed_riffs=existed_riffs)

        else:
            riff_info = riffs[riff_type][int(index)-1]

            save_riff_as(name, riff_info, riff_type)

            return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/<riff_type>/<option>/<index>', methods=['POST'])
def alter_riff(riff_type, option, index):
    if request.method == 'POST':

        riffs = get_temp_riffs()
        riff_info = riffs[riff_type][int(index) - 1]

        riff = parse_griff_json(riff_info)
        riff.add_notes_to_pm('E2', 120, 27)
        riff.save_midi(f'temp_{riff_type}_{index}')

        modified_riffs = get_temp_modified_riffs()
        current_no = get_largest_num_of_json(modified_riffs[riff_type]) + 1

        modified_riff = modify_riff(riff, riff_type, option)
        modified_riff.add_notes_to_pm(instr=0)
        modified_riff.save_fig(f'fig_{riff_type}_{current_no}', riff_type)

        modified_riff_info = modified_riff.export_json_dict()
        modified_riff_info['original_no'] = riff_info['no']
        modified_riff_info['no'] = current_no

        modified_riffs[riff_type].append(modified_riff_info)

        save_temp_modified_riffs(modified_riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/load/<riff_type>/<index>', methods=['POST'])
def load_riff(riff_type, index):
    if request.method == 'POST':
        existed_riffs = get_all_existed_riffs(riff_type)
        riffs = get_temp_riffs()

        riff_info = existed_riffs[int(index)-1]

        if riff_type == 'griff':
            riff = parse_griff_json(riff_info)
        elif riff_type == 'briff':
            riff = parse_briff_json(riff_info)
        else:
            assert riff_type == 'driff'
            riff = parse_driff_json(riff_info)

        riff_info = riff.export_json_dict()
        riff_info['no'] = get_largest_num_of_json(riffs[riff_type]) + 1

        riffs[riff_type].append(riff_info)

        save_temp_riffs(riffs)

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/delete_stored_riff/<riff_type>/<index>', methods=['POST'])
def delete_stored_riff(riff_type, index):

    if request.method == 'POST':

        stored_riffs = get_all_existed_riffs(riff_type)
        riffs = get_temp_riffs()
        modified_riffs = get_temp_modified_riffs()

        riff_name_to_delete = stored_riffs[int(index)-1]['name']

        delete_stored_riff_in_db(riff_name_to_delete, riff_type)

        info = f'Delete {riff_type} in database'

        return render_template('riffs/' + riff_type + '.html',
                               riffs=riffs[riff_type], modified_riffs=modified_riffs[riff_type],
                               riff_type=riff_type,
                               info=info, existed_riffs=get_all_existed_riffs(riff_type))

