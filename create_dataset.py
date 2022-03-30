from checklist.editor import Editor
from checklist.perturb import Perturb
import json
import re


# Create dataset

data = []
challenge_set = {}


# Be_disambiguation

def change_be_frame(x, meta=True, *args, **kwargs):
    comments = ['teachers', 'leaders', 'safe', 'open', 'close', 'excited', 'concerned']
    verbs = ['married', 'opened', 'closed', 'done', 'hit', 'thinking', 'singing']

    ret = []
    ret_meta = []

    for c in comments:
        if re.search(r'\b%s\b' % c, x):
            for v in verbs:
                ret.append(re.sub(r'\b%s\b' % c, v, x))
                ret_meta.append((c, v))

    if meta:
        return ret, ret_meta
    else:
        return ret

be_01_sents = ['We are teachers.',
               'We are leaders.',
               'We are safe.',
               'We are open.',
               'We are close',
               'We are excited.',
               'We are concerned.']
t_p = Perturb.perturb(be_01_sents, change_be_frame,
                      meta=True, keep_original=True, n_samples=1)

sentences = []
for comment in t_p['data']:
    for verb in comment[1:]:
        sentences.append((comment[0], verb))

metas = []
for comment in t_p['meta']:
    metas.extend(comment[1:])

for i, sent in enumerate(sentences):
    example = {'capability': "be_disambiguation",
               'test_type': "DIR",
               'perturbed_sentences': sent,
               'targets': metas[i],
               'expected_label': ("ARG2", "O")}
    data.append(example)


# Location_recognition

def change_arg2_loc(x, meta=True, *args, **kwargs):
    field = ['fashion', 'sports', 'this industry', 'the media sector', 'the medical field']
    location = ['Amsterdam', 'this district', 'another country', 'New York', 'Narnia']

    ret = []
    ret_meta = []

    for f in field:
        if re.search(r'\b%s\b' % f, x):
            for l in location:
                ret.append(re.sub(r'\b%s\b' % f, l, x))
                ret_meta.append((f, l))

    if meta:
        return ret, ret_meta
    else:
        return ret

field_sents = ['I work in fashion.',
               'I work in sports.',
               'I work in this industry.',
               'I work in the media sector.',
               'I work in the medical field.']
t_p = Perturb.perturb(field_sents, change_arg2_loc,
                      meta=True, keep_original=True, n_samples=1)

sentences = []
for field in t_p['data']:
    for verb in field[1:]:
        sentences.append((field[0], verb))

metas = []
for field in t_p['meta']:
    metas.extend(field[1:])

for i, sent in enumerate(sentences):
    example = {'capability': "location_recognition",
               'test_type': "DIR",
               'perturbed_sentences': sent,
               'targets': metas[i],
               'expected_label': ("ARG2", "ARGM-LOC")}
    data.append(example)


# Negating ARG1
editor = Editor()
t = editor.template('I have no {mask}.', meta=True, nsamples=100)
for i, sent in enumerate(t['data']):
    example = {'capability': "negating_arg1",
               'test_type': "MFT",
               'sentence': sent,
               'target': t['meta'][i]['mask'][0],
               'expected_label': "ARG1"}
    data.append(example)


# Theme in causative alternation
verb = ["roll", "bounce", "swing", "break", "chip", "crack", "bend", "crease", "crinkle", "cheer",
        "delight", "thrill", "blunt", "clear", "clean", "blacken", "redden", "grey", "awaken",
        "brighten", "broaden", "solidify", "stratify", "emulsify", "democratize", "decentralize",
        "crystallize", "accelerate", "ameliorate", "operate"]
transitive = editor.template('They {verb} them.', verb = verb)
intransitive = editor.template('They {verb}.', verb = verb)

for i, sent in enumerate(transitive['data']):
    example = {'capability': "theme_in_causative_alternation",
               'test_type': "INV",
               'perturbed_sentences': (sent, intransitive['data'][i]),
               'targets': ("them", "They"),
               'expected_label': "ARG1"}
    data.append(example)


# ARG1 in passive
averb = ["sign", "send", "read", "tear", "burn"]
arg = ["letter", "card", "contract", "poster", "Harry Potter"]
pverb = ["signed", "sent", "read", "torn", "burned"]

active = [f"I {av} the {ar}." for av in averb for ar in arg]
passive = [f"The {ar} is {pv}." for pv in pverb for ar in arg]
meta = [(ar, ar) for av in averb for ar in arg]

for i, sent in enumerate(active):
    example = {'capability': "arg1_in_passive",
               'test_type': "INV",
               'perturbed_sentences': (sent, passive[i]),
               'targets': meta[i],
               'expected_label': "ARG1"}
    data.append(example)


# Writing to json
challenge_set['data'] = data
with open("srl_challenge_set.json", 'w', encoding='utf-8') as f:
    json.dump(challenge_set, f, ensure_ascii=False, indent=4)
