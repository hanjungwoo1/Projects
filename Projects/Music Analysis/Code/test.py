import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from collections import defaultdict
from music21 import converter, corpus, instrument, midi, note, chord, pitch
from music21 import roman, stream
from multiprocessing import Pool


# Some helper methods.
def concat_path(path, child):
    return path + "/" + child


def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)


def get_composer_list(dir_list:list):

    dir_lists = []

    for era in dir_list:
        era_path = path+"/"+era
        composer_list = os.listdir(era_path)

        for composer in composer_list:
            temp = era_path + "/" + composer
            dir_lists.append(temp)

    return dir_lists


def get_midi_file(composer_list:list):

    midi_list = []

    for composer in composer_list:
        file_list = os.listdir(composer)

        for file in file_list:
            if "mid" in file:
                temp = composer + "/" + file
                midi_list.append(temp)

    return midi_list


def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)


def get_file_name(link):
    filename = link.split('/')[::-1][0]
    return filename


def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note)
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note

            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length

    return bass_note


def harmonic_reduction(midi_file):
    ret = []
    temp_midi = stream.Score()
    temp_midi_chords = midi_file.chordify()
    temp_midi.insert(0, temp_midi_chords)
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4
    for m in temp_midi_chords.measures(0, None):  # None = get all measures.
        if (type(m) != stream.Measure):
            continue

        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret.append("-")  # Empty measure
            continue

        sorted_items = sorted(count_dict.items(), key=lambda x: x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)

        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret.append(simplify_roman_name(roman_numeral))

    return ret


def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()

    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
             (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()

    if (inversion_name is not None):
        ret = ret + str(inversion_name)

    elif (roman_numeral.isDominantSeventh()):
        ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()):
        ret = ret + "o7"
    return ret


def process_single_file(midi_path):
    try:
        midi_name = get_file_name(midi_path)
        midi = open_midi(midi_path, True)
        return (
            midi.analyze('key'),
            midi_path,
            harmonic_reduction(midi),
            midi_name)
    except Exception as e:
        print("Error on {0}".format(midi_name))
        print(e)
        return None


def create_midi_dataframe(midi_files):
    key_signature_column = []
    game_name_column = []
    harmonic_reduction_column = []
    midi_name_column = []
    pool = Pool(8)


    results = pool.map(process_single_file, midi_files)

    for result in results:
        if (result is None):
            continue

        key_signature_column.append(result[0])
        game_name_column.append(result[1])
        harmonic_reduction_column.append(result[2])
        midi_name_column.append(result[3])

    d = {'midi_name': midi_name_column,
         'game_name': game_name_column,
         'key_signature': key_signature_column,
         'harmonic_reduction': harmonic_reduction_column}
    return pd.DataFrame(data=d)


def key_hist(df, game_name, ax):
    title = "All Games Key Signatures"
    filtered_df = df
    if (game_name is not None):
        title = game_name + " Key Signatures"
        filtered_df = df[df["game_name"] == game_name]

    filtered_df["key_signature"].value_counts().plot(ax=ax, kind='bar', title=title)




if __name__ == "__main__":

    path = "../Data"
    dir_lists = os.listdir(path)
    composer_lists = get_composer_list(dir_lists)
    midi_lists = get_midi_file(composer_lists)

    # for midi_file in midi_lists:
    #     print(midi_file, "~~~ now processing ----------------------")
    #
    #     data = process_single_file(midi_file)
    df = create_midi_dataframe(midi_lists)

    print(df)
    df.to_csv('../file_name.csv', sep=',', na_rep='NaN')