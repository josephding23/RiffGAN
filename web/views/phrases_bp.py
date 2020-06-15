from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from music.pieces.toolkit import set_used_riff_num_info, get_available_riff_no
from web.data.song import riffs, phrases

phrases_bp = Blueprint('phrases', __name__, template_folder='templates', static_folder='static', url_prefix='/phrases')


@phrases_bp.route('/<phrase_type>', methods=['GET'])
def get_phrases(phrase_type):
    if phrase_type == 'rhythm_guitar_phrase':
        return render_template('phrases/rhythm_guitar_phrase.html',
                               phrases=phrases['rhythm_guitar_phrase'], phrase_type='rhythm_guitar_phrase')
    elif phrase_type == 'rhythm_bass_phrase':
        return render_template('phrases/rhythm_bass_phrase.html',
                               phrases=phrases['rhythm_bass_phrase'], phrase_type='rhythm_bass_phrase')
    else:
        assert phrase_type == 'drum_phrase'
        return render_template('phrases/drum_phrase.html',
                               phrases=phrases['drum_phrase'], phrase_type='drum_phrase')


@phrases_bp.route('/delete/<phrase_type>/<index>', methods=['POST'])
def delete_phrase(phrase_type, index):
    phrases[phrase_type].pop(int(index)-1)
    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/edit/<phrase_type>/<index>', methods=['POST'])
def edit_phrase(phrase_type, index):
    according_riffs_dict = {
        'rhythm_guitar_phrase': 'griff',
        'rhythm_bass_phrase': 'briff',
        'drum_phrase': 'driff'
    }
    if request.method == 'POST':
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

        return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))


@phrases_bp.route('/new/<phrase_type>', methods=['POST'])
def new_phrase(phrase_type):
    return redirect(url_for('phrases.get_phrases', phrase_type=phrase_type))