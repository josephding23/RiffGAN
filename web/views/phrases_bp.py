from flask import Blueprint, render_template, redirect, url_for, request
from web.database.song import *
from music.custom_elements.rhythm_riff.toolkit import *
from music.pieces.phrase.toolkit import *
from music.pieces.track.toolkit import *
from music.pieces.song.toolkit import *
import pygame


phrases_bp = Blueprint('phrases', __name__, template_folder='templates', static_folder='static', url_prefix='/phrases')

freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(1)


@phrases_bp.route('/<phrase_type>', methods=['GET'])
def get_phrases(phrase_type):
    return render_template('phrases/' + phrase_type + '.html',
                           phrases=get_temp_phrases()[phrase_type], phrase_type=phrase_type)


@phrases_bp.route('/delete/<phrase_type>/<index>', methods=['POST'])
def delete_phrase(phrase_type, index):
    phrases = get_temp_phrases()
    tracks = get_temp_tracks()

    according_phrases_dict = {
        'rhythm_guitar_phrase': 'guitar',
        'rhythm_bass_phrase': 'bass',
        'drum_phrase': 'drum'
    }
    phrase_no_to_delete = phrases[phrase_type][int(index) - 1]['no']

    phrase_no_in_use = get_all_used_phrases(tracks, according_phrases_dict[phrase_type])
    print(phrase_no_in_use)

    if phrase_no_to_delete in phrase_no_in_use:
        error = 'Phrase you tend to delete is in use.'
        return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type], phrase_type=phrase_type,
                               error=error)

    phrases[phrase_type].pop(int(index)-1)
    save_temp_phrases(phrases)

    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/play/<phrase_type>/<index>', methods=['POST'])
def play_phrase(phrase_type, index):
    if request.method == 'POST':

        phrases = get_temp_phrases()

        phrase_info = phrases[phrase_type][int(index)-1]
        # refresh_riff_info(phrase_info, phrase_type, riffs)

        if phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase']:
            phrase = parse_rhythm_phrase_json(phrase_info)
            phrase.add_riffs_to_pm()
            phrase.save_midi(f'temp_{phrase_type}_{index}')
            phrase.play_with_no_init()

        else:
            assert phrase_type == 'drum_phrase'
            phrase = parse_drum_phrase_json(phrase_info)
            phrase.add_riffs_to_pm()
            phrase.save_midi(f'temp_{phrase_type}_{index}')
            phrase.play_with_no_init()

        return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/stop/<phrase_type>', methods=['POST'])
def stop_phrase(phrase_type):
    if request.method == 'POST':

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()

        return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/edit/<phrase_type>/<index>', methods=['POST'])
def edit_phrase(phrase_type, index):
    according_riffs_dict = {
        'rhythm_guitar_phrase': 'griff',
        'rhythm_bass_phrase': 'briff',
        'drum_phrase': 'driff'
    }
    if request.method == 'POST':

        riffs = get_temp_riffs()
        phrases = get_temp_phrases()
        tracks = get_temp_tracks()
        song = get_temp_song()

        if phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase']:
            raw_length = request.form['edit_length_input']
            raw_bpm = request.form['edit_bpm_input']
            raw_tonality = request.form['edit_tonality_input']
            raw_instr = request.form['edit_instr_input']
            raw_used_riffs = request.form['edit_used_riffs_input']
            raw_arrangements = request.form['edit_arrangements_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                bpm = float(raw_bpm)
            except Exception:
                error = 'BPM must be float'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                tonic, mode = raw_tonality.split(' ')
                root_note = note_name_to_num(tonic)
                mode_index = ['major', 'minor'].index(mode)
                tonality = [tonic, mode]
            except Exception:
                error = 'Invalid tonality format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            instr = int(raw_instr)
            if phrase_type == 'rhythm_guitar_phrase':
                instr_name = get_guitar_str(instr)
                instr_type = 'guitar'
            else:
                assert phrase_type == 'rhythm_bass_phrase'
                instr_name = get_bass_str(instr)
                instr_type = 'bass'

            try:
                used_riffs = get_used_riffs_from_raw(raw_used_riffs)
                available_riffs = get_available_riff_no(riffs, according_riffs_dict[phrase_type])
                for riff_no in used_riffs:
                    if riff_no not in available_riffs:
                        error = f'Riff No.{riff_no} is not available.'
                        return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                               phrase_type=phrase_type, error=error)
                print(get_available_riff_no(riffs, 'griff'))
            except Exception:
                error = 'Invalid used riffs format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                arrangements = get_rhythm_arrangements_from_raw(raw_arrangements)
            except Exception:
                error = 'Invalid arrangements format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            phrases[phrase_type][int(index)-1] = {
                'no': phrases[phrase_type][int(index)-1]['no'],
                'length': length,
                'bpm': bpm,
                'tonality': tonality,
                'instr': instr,
                'instr_type': instr_type,
                'instr_str': instr_name,
                'riffs_no': used_riffs,
                'raw_riffs_no': raw_used_riffs,
                'arrangements': arrangements,
                'raw_arrangements': raw_arrangements
            }

            refresh_riff_info(phrases[phrase_type][int(index)-1], phrase_type, riffs)
            refresh_all_tracks(tracks, phrases)
            refresh_all_tracks_in_song(song, tracks)

            save_temp_phrases(phrases)
            save_temp_tracks(tracks)

            return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))

        else:
            assert phrase_type == 'drum_phrase'
            raw_length = request.form['edit_length_input']
            raw_bpm = request.form['edit_bpm_input']
            raw_used_riffs = request.form['edit_used_riffs_input']
            raw_arrangements = request.form['edit_arrangements_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                bpm = float(raw_bpm)
            except Exception:
                error = 'BPM must be float'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                used_riffs = get_used_riffs_from_raw(raw_used_riffs)
                available_riffs = get_available_riff_no(riffs, according_riffs_dict[phrase_type])
                for riff_no in used_riffs:
                    if riff_no not in available_riffs:
                        error = f'Riff No.{riff_no} is not available.'
                        return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                               phrase_type=phrase_type, error=error)

            except Exception:
                    error = 'Invalid used riffs format'
                    return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                           phrase_type=phrase_type, error=error)

            try:
                arrangements = get_drum_arrangements_from_raw(raw_arrangements)
            except Exception:
                error = 'Invalid arrangements format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            phrases[phrase_type][int(index) - 1] = {
                'no': phrases[phrase_type][int(index) - 1]['no'],
                'length': length,
                'bpm': bpm,
                'riffs_no': used_riffs,
                'raw_riffs_no': raw_used_riffs,
                'arrangements': arrangements,
                'raw_arrangements': raw_arrangements
            }

            refresh_riff_info(phrases[phrase_type][int(index) - 1], phrase_type, riffs)
            refresh_all_tracks(tracks, phrases)
            refresh_all_tracks_in_song(song, tracks)

            save_temp_phrases(phrases)
            save_temp_tracks(tracks)

            return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/new/<phrase_type>', methods=['POST'])
def new_phrase(phrase_type):
    phrases = get_temp_phrases()
    riffs = get_temp_riffs()

    according_riffs_dict = {
        'rhythm_guitar_phrase': 'griff',
        'rhythm_bass_phrase': 'briff',
        'drum_phrase': 'driff'
    }
    if request.method == 'POST':
        if phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase']:
            raw_length = request.form['new_length_input']
            raw_bpm = request.form['new_bpm_input']
            raw_tonality = request.form['new_tonality_input']
            raw_instr = request.form['new_instr_input']
            raw_used_riffs = request.form['new_used_riffs_input']
            raw_arrangements = request.form['new_arrangements_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                bpm = float(raw_bpm)
            except Exception:
                error = 'BPM must be float'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                tonic, mode = raw_tonality.split(' ')
                root_note = note_name_to_num(tonic)
                mode_index = ['major', 'minor'].index(mode)
                tonality = [tonic, mode]
            except Exception:
                error = 'Invalid tonality format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            instr = int(raw_instr)
            if phrase_type == 'rhythm_guitar_phrase':
                instr_name = get_guitar_str(instr)
                instr_type = 'guitar'
            else:
                assert phrase_type == 'rhythm_bass_phrase'
                instr_name = get_bass_str(instr)
                instr_type = 'bass'

            try:
                used_riffs = get_used_riffs_from_raw(raw_used_riffs)
                available_riffs = get_available_riff_no(riffs, according_riffs_dict[phrase_type])
                for riff_no in used_riffs:
                    if riff_no not in available_riffs:
                        error = f'Riff No.{riff_no} is not available.'
                        return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                               phrase_type=phrase_type, error=error)

                print(get_available_riff_no(riffs, 'griff'))
            except Exception:
                error = 'Invalid used riffs format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                arrangements = get_rhythm_arrangements_from_raw(raw_arrangements)
            except Exception:
                error = 'Invalid arrangements format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            phrase_info = {
                'no': get_largest_num_of_json(phrases[phrase_type]) + 1,
                'length': length,
                'bpm': bpm,
                'tonality': tonality,
                'instr': instr,
                'instr_type': instr_type,
                'instr_str': instr_name,
                'riffs_no': used_riffs,
                'raw_riffs_no': raw_used_riffs,
                'arrangements': arrangements,
                'raw_arrangements': raw_arrangements
            }

            refresh_riff_info(phrase_info, phrase_type, riffs)

            phrases[phrase_type].append(phrase_info)
            save_temp_phrases(phrases)

        else:
            assert phrase_type == 'drum_phrase'
            raw_length = request.form['new_length_input']
            raw_bpm = request.form['new_bpm_input']
            raw_used_riffs = request.form['new_used_riffs_input']
            raw_arrangements = request.form['new_arrangements_input']

            try:
                length = int(raw_length)
            except Exception:
                error = 'Length must be integer'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                bpm = float(raw_bpm)
            except Exception:
                error = 'BPM must be float'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                used_riffs = get_used_riffs_from_raw(raw_used_riffs)
                available_riffs = get_available_riff_no(riffs, according_riffs_dict[phrase_type])
                for riff_no in used_riffs:
                    if riff_no not in available_riffs:
                        error = f'Riff No.{riff_no} is not available.'
                        return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                               phrase_type=phrase_type, error=error)

            except Exception:
                error = 'Invalid used riffs format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            try:
                arrangements = get_drum_arrangements_from_raw(raw_arrangements)
            except Exception:
                error = 'Invalid arrangements format'
                return render_template('phrases/' + phrase_type + '.html', phrases=phrases[phrase_type],
                                       phrase_type=phrase_type, error=error)

            phrase_info = {
                'no': get_largest_num_of_json(phrases[phrase_type]) + 1,
                'length': length,
                'bpm': bpm,
                'riffs_no': used_riffs,
                'raw_riffs_no': raw_used_riffs,
                'arrangements': arrangements,
                'raw_arrangements': raw_arrangements
            }

            refresh_riff_info(phrase_info, phrase_type, riffs)

            phrases[phrase_type].append(phrase_info)
            save_temp_phrases(phrases)

        return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))
