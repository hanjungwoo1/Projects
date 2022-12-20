import pandas as pd
import numpy as np
import seaborn as sns

import gensim, logging
import pprint


def get_related_chords(token, topn=3):
    print("Similar chords with " + token)
    for word, similarity in model.wv.most_similar(positive=[token], topn=topn):
        print (word, round(similarity, 3))


def get_chord_similarity(chordA, chordB):
    print("Similarity between {0} and {1}: {2}".format(
        chordA, chordB, model.wv.similarity(chordA, chordB)))


def vectorize_harmony(model, harmonic_reduction):
    # Gets the model vector values for each chord from the reduction.
    word_vecs = []
    for word in harmonic_reduction:
        try:
            vec = model.wv[word]
            word_vecs.append(vec)
        except KeyError:
            # Ignore, if the word doesn't exist in the vocabulary
            pass

    # Assuming that document vector is the mean of all the word vectors.
    return np.mean(word_vecs, axis=0)


def cosine_similarity(vecA, vecB):
    # Find the similarity between two vectors based on the dot product.
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0

    return csim


def calculate_similarity_aux(df, model, source_name, target_names=[], threshold=0):
    source_harmo = df[df["midi_name"] == source_name]["harmonic_reduction"].values[0]
    source_vec = vectorize_harmony(model, source_harmo)
    results = []
    for name in target_names:
        target_harmo = df[df["midi_name"] == name]["harmonic_reduction"].values[0]
        if (len(target_harmo) == 0):
            continue

        target_vec = vectorize_harmony(model, target_harmo)
        sim_score = cosine_similarity(source_vec, target_vec)
        if sim_score > threshold:
            results.append({
                'score': sim_score,
                'name': name
            })

    # Sort results by score in desc order
    results.sort(key=lambda k: k['score'], reverse=True)
    return results


def calculate_similarity_era(df, model, source_name, target_prefix, threshold=0):
    source_midi_names = df[df["midi_name"] == source_name]["midi_name"].values
    if (len(source_midi_names) == 0):
        print("Invalid source name")
        return

    source_midi_name = source_midi_names[0]

    target_midi_names = df[df["era_name"].str.startswith(target_prefix)]["midi_name"].values
    if (len(target_midi_names) == 0):
        print("Invalid target prefix")
        return

    return calculate_similarity_aux(df, model, source_midi_name, target_midi_names, threshold)


def calculate_similarity(df, model, source_name, threshold=0):
    source_midi_names = df[df["midi_name"] == source_name]["midi_name"].values
    if (len(source_midi_names) == 0):
        print("Invalid source name")
        return

    source_midi_name = source_midi_names[0]

    target_midi_names = df["midi_name"].values
    if (len(target_midi_names) == 0):
        print("Invalid target prefix")
        return

    return_list = []

    data = calculate_similarity_aux(df, model, source_midi_name, target_midi_names, threshold)

    for each_data in data:
        return_list.append(each_data["score"])


    return return_list


if __name__ == "__main__":

    df = pd.read_csv("../file_name.csv", sep=",")

    model = gensim.models.Word2Vec(df["harmonic_reduction"], min_count=2, window=4)

    print("List of chords found:")
    print(model.wv.index_to_key)
    print("Number of chords considered by model: {0}".format(len(model.wv.index_to_key)))

    # get_related_chords('I')
    # get_related_chords('#')
    # get_related_chords('V')

    # The first one should be smaller since "i" and "ii" chord doesn't share notes,
    # different from "IV" and "vi" which share 2 notes.
    # get_chord_similarity("i", "I")
    # get_chord_similarity("v", "5")

    # This one should be bigger because they are "enharmonic".

    total_list = []

    pp = pprint.PrettyPrinter(width=41, compact=True)
    for midi in df["midi_name"]:
        # pp.pprint(calculate_similarity(df, model, midi))
        total_list.append(calculate_similarity(df, model, midi))

    # pp = pprint.PrettyPrinter(width=41, compact=True)
    # pp.pprint(calculate_similarity(df, model, "lastravaganza.mid", "Romantic"))  #




