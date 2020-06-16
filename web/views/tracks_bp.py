from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from web.data.song import riffs, phrases, tracks

tracks_bp = Blueprint('tracks', __name__, template_folder='templates', static_folder='static', url_prefix='/tracks')


@tracks_bp.route('/', methods=['GET'])
def get_tracks():
    return render_template('tracks.html', tracks=tracks)


@tracks_bp.route('/delete/<index>', methods=['POST'])
def delete_track(index):
    tracks.pop(int(index)-1)
    return redirect(url_for('tracks.get_tracks'))


@tracks_bp.route('/edit/<index>', methods=['POST'])
def edit_track(index):
    if request.method == 'POST':
        raw_is_drum = request.form['edit_is_drum_input']
        is_drum = bool(int(raw_is_drum))

        raw_bpm_info = request.form['edit_bpm_info_input']
        raw_used_phrases = request.form['edit_used_phrases_input']
        raw_arrangements = request.form['edit_arrangements_input']

        try:
            bpm_info = get_bpm_info_from_raw(raw_bpm_info)
        except:
            error = 'Invalid BPM info format'
            return render_template('tracks.html', tracks=tracks, error=error)

        try:
            used_phrases = get_used_phrases_from_raw(raw_used_phrases)
            if is_drum:
                phrase_type = 'drum_phrase'
            else:
                instr_type = request.form['edit_instr_type_input']
                if instr_type == 'guitar':
                    phrase_type = 'rhythm_guitar_phrase'
                else:
                    assert instr_type == 'bass'
                    phrase_type = 'rhythm_bass_phrase'

            available_phrases = get_available_phrase_no(phrases, phrase_type)
            for phrase_no in used_phrases:
                if phrase_no not in available_phrases:
                    error = f'Phrase No.{phrase_no} is not available.'
                    return render_template('tracks.html', tracks=tracks, error=error)

        except:
            error = 'Invalid used phrases format'
            return render_template('tracks.html', tracks=tracks, error=error)

        try:
            arrangements = get_phrase_arrangements_from_raw(raw_arrangements)
        except:
            error = 'Invalid arrangements format'
            return render_template('tracks.html', tracks=tracks, error=error)

        if not is_drum:
            instr_type = request.form['edit_instr_type_input']
            raw_tonality_info = request.form['edit_tonality_info_input']

            try:
                tonality_info = get_tonality_info_from_raw(raw_tonality_info)
            except:
                error = 'Invalid tonality info format'
                return render_template('tracks.html', tracks=tracks, error=error)

            tracks[int(index) - 1] = {
                'name': tracks[int(index) - 1]['name'],
                'is_drum': is_drum,
                'instr_type': instr_type,
                'bpm_list': bpm_info,
                'tonality_list': tonality_info,
                'phrases_no': used_phrases,
                'arrangements': arrangements,

                'raw_bpm_info': raw_bpm_info,
                'raw_tonality_info': raw_tonality_info,
                'raw_phrases_no': raw_used_phrases,
                'raw_arrangements': raw_arrangements
            }

        else:
            tracks[int(index) - 1] = {
                'name': tracks[int(index) - 1]['name'],
                'is_drum': is_drum,
                'bpm_list': bpm_info,
                'phrases_no': used_phrases,
                'arrangements': arrangements,

                'raw_bpm_info': raw_bpm_info,
                'raw_phrases_no': raw_used_phrases,
                'raw_arrangements': raw_arrangements
            }

        return redirect(url_for('tracks.get_tracks'))


@tracks_bp.route('/new', methods=['POST'])
def new_track():
    if request.method == 'POST':
        name = request.form['new_name_input']

        raw_is_drum = request.form['new_is_drum_input']
        is_drum = bool(raw_is_drum)

        raw_bpm_info = request.form['new_bpm_info_input']
        raw_used_phrases = request.form['new_used_phrases_input']
        raw_arrangements = request.form['new_arrangements_input']

        try:
            bpm_info = get_bpm_info_from_raw(raw_bpm_info)
        except:
            error = 'Invalid BPM info format'
            return render_template('tracks.html', tracks=tracks, error=error)

        try:
            used_phrases = get_used_phrases_from_raw(raw_used_phrases)
        except:
            error = 'Invalid used phrases format'
            return render_template('tracks.html', tracks=tracks, error=error)

        try:
            arrangements = get_phrase_arrangements_from_raw(raw_arrangements)
        except:
            error = 'Invalid arrangements format'
            return render_template('tracks.html', tracks=tracks, error=error)

        if not is_drum:
            instr_type = request.form['edit_instr_type_input']
            raw_tonality_info = request.form['edit_tonality_info_input']

            try:
                tonality_info = get_tonality_info_from_raw(raw_tonality_info)
            except:
                error = 'Invalid tonality info format'
                return render_template('tracks.html', tracks=tracks, error=error)

            tracks.append({
                'name': name,
                'is_drum': is_drum,
                'instr_type': instr_type,
                'bpm_list': bpm_info,
                'tonality_list': tonality_info,
                'phrases_no': used_phrases,
                'arrangements': arrangements,

                'raw_bpm_info': raw_bpm_info,
                'raw_tonality_info': raw_tonality_info,
                'raw_phrases_no': raw_used_phrases,
                'raw_arrangements': raw_arrangements
            })

        else:
            tracks.append({
                'name': name,
                'is_drum': is_drum,
                'bpm_list': bpm_info,
                'phrases_no': used_phrases,
                'arrangements': arrangements,

                'raw_bpm_info': raw_bpm_info,
                'raw_phrases_no': raw_used_phrases,
                'raw_arrangements': raw_arrangements
            })

        return redirect(url_for('tracks.get_tracks'))
