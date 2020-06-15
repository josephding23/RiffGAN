from flask import Blueprint, render_template, redirect, url_for, request
from music.pieces.song import *
from music.custom_elements.toolkit import get_all_used_riffs
from web.data.song import phrases, riffs

riffs_bp = Blueprint('riffs', __name__, template_folder='templates', static_folder='static', url_prefix='/riffs')

@riffs_bp.route('/<riff_type>', methods=['GET'])
def get_riffs(riff_type):
    if riff_type == 'griff':
        return render_template('riffs/griff.html', riffs=riffs['griff'], riff_type='griff')

    elif riff_type == 'briff':
        return render_template('riffs/briff.html', riffs=riffs['briff'], riff_type='briff')

    else:
        assert riff_type == 'driff'
        return render_template('riffs/driff.html', riffs=riffs['driff'], riff_type='driff')


@riffs_bp.route('/delete/<riff_type>/<index>', methods=['POST'])
def delete_riff(riff_type, index):
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
    return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/edit/<riff_type>/<index>', methods=['POST'])
def edit_riff(riff_type, index):
    if request.method == 'POST':
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
                print(degrees_and_types)
            except Exception:
                error = 'Invalid Degrees & Types format.'
                return render_template('riffs/' + riff_type + '.html', riffs=riffs[riff_type], riff_type=riff_type,
                                       error=error)

            try:
                timestamps = get_timestamps_from_raw(raw_timestamps)
                print(timestamps)
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
            return redirect(url_for('riffs.get_riffs', riff_type=riff_type))


@riffs_bp.route('/new/<riff_type>', methods=['POST'])
def new_riff(riff_type):
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

        return redirect(url_for('riffs.get_riffs', riff_type=riff_type))
