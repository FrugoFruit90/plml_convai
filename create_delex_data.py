# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib.request, urllib.parse, urllib.error
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np

from utils import dbPointer
from utils import delexicalize

from utils.nlp import normalize


np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 40


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domain = 'restaurant'

    # TODO TASK 1
    # Create one-hot encoding of the belief state.
    # The belief state should consists of 3 numbers per slot.
    # Each 0-1 number represents whether the slot was 'not mentioned'
    # or the user 'dont care' or the slot has specific value from the ontology:
    summary_bstate = []
    for slot in bstate[domain]['semi']:
        if slot != 'requested':
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care"]:
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1

            summary_bstate += slot_enc

    # TODO TASK 3
    # Create one-hot encoding of the belief state for requests.
    # The belief state should consists of 1 numbers per slot.
    # Each 0-1 number represents whether the slot was requested by the user.
    for slot in ['pricerange', 'area', 'food', 'phone', 'address', 'postcode', 'name']:
        if slot in bstate[domain]['semi']['requested']:
            slot_enc_req = [1.]
        else:
            slot_enc_req = [0.]
        summary_bstate += slot_enc_req

    return summary_bstate


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        #print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print('no db')
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            #d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print('not ascii')
                return None
            #d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    db = [t['db_pointer'] for t in d_orig['usr_log']]
    bs = [t['belief_summary'] for t in d_orig['sys_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    for u, d, s, b in zip(usr, db, sys, bs):
        dial.append((u, s, d, b))

    return dial


def createDict(word_freqs):
    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + len(extra_tokens)

    for key, idx in list(worddict.items()):
        if idx >= DICT_SIZE:
            del worddict[key]

    return worddict


def loadData():
    data_url = "data/multi-woz/data.json"
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/multi-woz")

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.request.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("data/multi-woz")
        zip_ref.close()
        shutil.copy('data/multi-woz/MULTIWOZ2 2/data.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/valListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/testListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/dialogue_acts.json', 'data/multi-woz/')


def createDelexData():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    
    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open('data/woz2/data.json')
    data = json.load(fin1)

    for dialogue_name in tqdm(data):
        if 'WOZ' not in dialogue_name:
            continue
        dialogue = data[dialogue_name]
        #print dialogue_name

        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])

            words = sent.split()
            sent = delexicalize.delexicalise(' '.join(words), dic)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)

                #print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

        delex_data[dialogue_name] = dialogue

    with open('data/delex.json', 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def divideData(data):
    """Given test and validation sets, divide
    the data for three different sets"""
    testListFile = []
    fin = open('data/testListFile')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open('data/valListFile')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    test_dials = {}
    val_dials = {}
    train_dials = {}
        
    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()
    
    for dialogue_name in tqdm(data):
        #print dialogue_name
        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['usr'] = []
            dialogue['sys'] = []
            dialogue['db'] = []
            dialogue['bs'] = []
            for turn in dial:
                dialogue['usr'].append(turn[0])
                dialogue['sys'].append(turn[1])
                dialogue['db'].append(turn[2])
                dialogue['bs'].append(turn[3])

            if dialogue_name in testListFile:
                test_dials[dialogue_name] = dialogue
            elif dialogue_name in valListFile:
                val_dials[dialogue_name] = dialogue
            else:
                train_dials[dialogue_name] = dialogue

            for turn in dial:
                line = turn[0]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_usr:
                        word_freqs_usr[w] = 0
                    word_freqs_usr[w] += 1

                line = turn[1]
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs_sys:
                        word_freqs_sys[w] = 0
                    word_freqs_sys[w] += 1

    # save all dialogues
    with open('data/val_dials.json', 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open('data/test_dials.json', 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open('data/train_dials.json', 'w') as f:
        json.dump(train_dials, f, indent=4)

    return word_freqs_usr, word_freqs_sys


def buildDictionaries(word_freqs_usr, word_freqs_sys):
    """Build dictionaries for both user and system sides.
    You can specify the size of the dictionary through DICT_SIZE variable."""
    dicts = []
    worddict_usr = createDict(word_freqs_usr)
    dicts.append(worddict_usr)
    worddict_sys = createDict(word_freqs_sys)
    dicts.append(worddict_sys)

    # reverse dictionaries
    idx2words = []
    for dictionary in dicts:
        dic = {}
        for k,v in list(dictionary.items()):
            dic[v] = k
        idx2words.append(dic)

    with open('data/input_lang.index2word.json', 'w') as f:
        json.dump(idx2words[0], f, indent=2)
    with open('data/input_lang.word2index.json', 'w') as f:
        json.dump(dicts[0], f,indent=2)
    with open('data/output_lang.index2word.json', 'w') as f:
        json.dump(idx2words[1], f, indent=2)
    with open('data/output_lang.word2index.json', 'w') as f:
        json.dump(dicts[1], f,indent=2)


def main():
    # # print('Creating delexicalized dialogues. Get yourself a coffee, this might take a while.')
    delex_data = createDelexData()

    with open('data/delex_data.json', 'w') as f:
        json.dump(delex_data, f, indent=4)

    with open('data/delex_data.json') as f:
        delex_data = json.load(f)

    print('Divide dialogues for separate bits - usr, sys, db, bs')
    word_freqs_usr, word_freqs_sys = divideData(delex_data)

    print('Building dictionaries')
    buildDictionaries(word_freqs_usr, word_freqs_sys)


if __name__ == "__main__":
    main()
