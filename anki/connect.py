import json
import urllib.request
import requests
import time
import JapaneseTokenizer
from functools import reduce, wraps
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from pydub import AudioSegment
import os
from pydub.playback import play
from typing import List, Tuple
import pickle


def cached(file_path=None):
    def fun(func):
        if file_path is not None and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                func.cache = pickle.load(f)
        else:
            func.cache = {}

        @wraps(func)
        def wrapper(*args):
            if args not in func.cache:
                func.cache[args] = func(*args)
                if file_path is not None:
                    with open(file_path, "wb") as f:
                        pickle.dump(func.cache, f)
            return func.cache[args]
        return wrapper
    return fun


def match_group(pattern, string, group, flags=0):
    m = re.match(pattern, string, flags=flags)
    if m is None:
        return None
    else:
        try:
            return m.group(group)
        except IndexError:
            return None

def request(action, **params):
    return {'action': action, 'params': params, 'version': 6}

def invoke(action, **params):
    request_json = json.dumps(request(action, **params)).encode('utf-8')
    response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8765', request_json)))
    if len(response) != 2:
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        raise Exception('response is missing required error field')
    if 'result' not in response:
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        raise Exception(response['error'])
    return response['result']


def clear_brackets(s):
    ret = []

    bracket_pairs = [('(', ')'), ('（', '）'), ('<', '>')]
    ob, cb = zip(*bracket_pairs)
    dept_map = {}

    for c in s:
        if c in ob:
            if ob.index(c) not in dept_map:
                dept_map[ob.index(c)] = 0
            dept_map[ob.index(c)] = dept_map[ob.index(c)] + 1
        elif c in cb:
            if cb.index(c) not in dept_map:
                dept_map[cb.index(c)] = 0
            dept_map[cb.index(c)] = dept_map[cb.index(c)] - 1
        elif all([d == 0 for d in dept_map.values()]):
            ret.append(c)

        if any([d < 0 for d in dept_map.values()]):
            print(dept_map)
            raise ValueError(f"Invalid bracketing: negative bracket depth in {s}")

    if any([d != 0 for d in dept_map.values()]):
        raise ValueError(f"Invalid bracketing: finished on unequal bracketing in {s}")

    return "".join(ret)


def load_lines(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    lines = [clear_brackets(line).strip() for line in lines]
    lines = [line for line in lines if line != ""]
    lines = ["".join([c for c in line if c != " "]) for line in lines]
    return lines


def split_list(lst, split_it, key=None):
    if key is None:
        key = lambda x: x

    res = []
    buf = []
    for it in lst:
        if key(it) == split_it:
            res.append(buf)
            buf = []
        else:
            buf.append(it)

    res.append(buf)
    return res


def tokenize(sentences):
    out = []
    buf = []
    junampp = JapaneseTokenizer.JumanppWrapper()
    for sentence in sentences:
        if sum([len(s) for s in buf]) + len(sentence) + len(buf) > 1000:
            payload = u"\u2424".join(buf)
            obj_list = junampp.tokenize(payload).tokenized_objects
            sentences = split_list(obj_list, u"\u2424", key=lambda x: x.word_stem)
            out.extend(sentences)
            buf = []

        buf.append(sentence)

    if len(buf) != 0:
        obj_list = junampp.tokenize(u"\u2424".join(buf)).tokenized_objects
        out.extend(split_list(obj_list, u"\u2424", key=lambda x: x.word_stem))

    return out


def index(lst, it):
    try:
        return lst.index(it)
    except ValueError:
        return None


def is_kanji(c):
    return u'\u9fff' > c > u'\u4e00' or ord(c) in [19968, 12293]


def contains_kanji(s):
    return any([is_kanji(c) for c in s])


def find_line():
    notes = invoke('findNotes', query="deck:yomichan")
    note_infos = invoke('notesInfo', notes=notes)
    note_sentences = [clear_brackets(note['fields']['Notes']['value']).strip() for note in note_infos]
    note_words = [clear_brackets(note['fields']['単語']['value']).strip() for note in note_infos]
    # t_note_sentences = tokenize(note_sentences)
    t_note_words = tokenize(note_words)

    script_lines = load_lines("shokugeki/s01e04")
    t_script_lines = tokenize(script_lines)
    t_script_lines = [[token for token in line if token.word_surface != u"\u3000"] for line in t_script_lines]
    t_script_line_stems = [[token.word_stem for token in line] for line in t_script_lines]
    t_script_line_surfaces = [[token.word_surface for token in line] for line in t_script_lines]

    for tokens in t_note_words:
        entry_stems = [token.word_stem for token in tokens]
        for i, stems in enumerate(t_script_line_stems):
            idx_list = [index(stems, entry_stem) for entry_stem in entry_stems]

            if all([idx is not None for idx in idx_list]) and all(b-a == 1 for a, b in zip(idx_list, idx_list[1:])):
                print("".join([token.word_surface for token in tokens]), " ".join(t_script_line_surfaces[i]), stems)
            else:
                pass


def add_audio():
    notes = invoke('findNotes', query="deck:yomichan")

    for note in invoke('notesInfo', notes=notes):
        if 'no-audio' in note['tags']:
            print("tagged as no-audio")
            continue

        if note['fields']['Audio']['value'] != '':
            print("audio exists")
            continue

        query = []
        if "読み" in note['fields']:
            query.append(('kana', note['fields']['読み']['value']))
            query.append(('kanji', note['fields']['単語']['value']))
        else:
            query.append(('kana', note['fields']['単語']['value']))

        file_id = "_".join([v for k, v in query])
        filename = f'yomichan_{file_id}.mp3'
        query = [f'{k}={v}' for k, v in query]
        audio_url = f"https://assets.languagepod101.com/dictionary/japanese/audiomp3.php?{'&'.join(query)}"
        req = requests.get(audio_url)

        if len(req.content) == 52288:
            invoke('addTags', notes=[note['noteId']], tags="no-audio")
            print('no audio')
        else:
            print("update note")
            invoke('updateNoteFields', note={
                'id': note['noteId'],
                'fields': {
                    "Audio": ''
                },
                'tags': [
                    'yomichan'
                ],
                'audio': [{
                    'url': audio_url,
                    'filename': filename,
                    "skipHash": "7e2c2f954ef6051373ba916f000168dc",
                    "fields": ["Audio"]
                }]
            })

            [note] = invoke('notesInfo', notes=[note['noteId']])
            while note['fields']['Audio']['value'] != f'[sound:{filename}]':
                print("update again")
                time.sleep(1.0)
                invoke('updateNoteFields', note={
                    'id': note['noteId'],
                    'fields': {
                        "Audio": f'[sound:{filename}]'
                    }
                })


def imis_info(token):
    info = token.misc_info['imis'].strip()

    if info == "NIL" or len(info) == 0:
        return None

    if info[-1] == "\"":
        info = info[:-1]

    items = info.split(" ")
    item_parts_list = [item.split(":") for item in items]
    d = {}
    for item_parts in item_parts_list:
        if len(item_parts) == 1:
            d[item_parts[0]] = True
        else:
            d[":".join(item_parts[:-1])] = item_parts[-1]

    return d


def stem_word(token):
    iinfo = imis_info(token)
    if iinfo is not None and '代表表記' in iinfo:
        return iinfo['代表表記'].split("/")[0]
    else:
        return token.word_stem


def stem_reading(token):
    iinfo = imis_info(token)
    if iinfo is not None and '代表表記' in iinfo:
        return iinfo['代表表記'].split("/")[-1]
    else:
        return token.word_stem


def query_jisho_by_token(token):
    if imis_info(token) is None:
        return None
    elif contains_kanji(token.word_stem):
        query = f"{stem_word(token)}+{stem_reading(token)}"
    else:
        query = stem_reading(token)

    return query_jisho(query)


@cached(file_path="data/jisho_cache.pkl")
def query_jisho(query):
    return requests.get(f"https://jisho.org/api/v1/search/words?keyword={query}").json()


def strip_na_ending(word):
    if word is None:
        return None
    word = word.strip()

    if word[-1] in ["だ", "な"]:
        return word[:-1]
    else:
        return word


def perfect_match_str(token_word, token_reading, jisho_word, jisho_reading):
    if (token_word is not None or token_word == jisho_word) and token_reading == jisho_reading:
        return token_word, token_reading

    if jisho_word is None and token_word == jisho_reading:
        return token_word, None

    return None


def perfect_match(token, jisho_item):
    if jisho_item is None:
        return None

    for japanese_item in jisho_item['japanese']:
        match = perfect_match_str(stem_word(token), stem_reading(token),
                                  japanese_item.get('word'), japanese_item.get('reading'))

        if match is not None:
            return match

        if token.misc_info.get('katuyou1') == 'ナ形容詞':
            match = perfect_match_str(strip_na_ending(stem_word(token)),
                                      strip_na_ending(stem_reading(token)),
                                      strip_na_ending(japanese_item.get('word')),
                                      strip_na_ending(japanese_item.get('reading')))
            if match is not None:
                return match

    return None


def format_common_annotation(jisho_res=None):
    if jisho_res is None:
        return ""

    common_annotations = []

    jlpt_levels = [match_group(r"jlpt-n(\d)", it, 1) for it in jisho_res.get("jlpt", [])]
    jlpt_levels = [lv for lv in jlpt_levels if lv is not None]
    jlpt_levels = [int(lv) for lv in jlpt_levels]
    if len(jlpt_levels) != 0:
        common_annotations.append(f"n{max(jlpt_levels)}")

    if jisho_res.get("is_common", False):
        common_annotations.append("c")

    if len(common_annotations) != 0:
        return f"({';'.join(common_annotations)})"
    else:
        return ""


def format_jisho_gloss_title(jisho_res=None):
    if jisho_res is None:
        return ""

    first_japanese_item = jisho_res['japanese'][0]
    word = first_japanese_item.get('word')
    reading = first_japanese_item.get('reading')

    if word is not None and word != reading:
        return f"{word} 【{reading}】{format_common_annotation(jisho_res)}"
    else:
        return reading


def format_token_gloss_title(token, jisho_res=None):
    # TODO implement that jisho_res can really be None => ignore perfect match criteria
    #  and just render token as it is
    common_annotation = format_common_annotation(jisho_res)

    match = perfect_match(token, jisho_res)
    if match is not None:
        word, reading = match
        if reading is not None and word != reading:
            if token.word_surface in [word, reading]:
                return f"{word} 【{reading}】{common_annotation}"
            else:
                return f"{token.word_surface} from {word} 【{reading}】{common_annotation}"
        else:
            if token.word_surface == word:
                return f"{word} {common_annotation}"
            else:
                return f"{token.word_surface} from {word} {common_annotation}"
    else:
        jp_list = jisho_res['japanese']

        l = []
        for jp in jp_list:
            if 'word' in jp and 'reading' in jp:
                return f"{jp['word']}【{jp['reading']}】{common_annotation}"
            elif 'word' in jp:
                return jp['word']
            else:
                return jp['reading']

        return f"{token.word_surface} PARTIAL MATCH {''.join(l)} {common_annotation}"


def format_gloss_title(jisho_res, token=None):
    if token is None:
        return format_jisho_gloss_title(jisho_res)
    else:
        return format_token_gloss_title(token, jisho_res)


def format_parts_of_speech(jisho_res):
    abbrev_map = {
        "Noun": "n",
        "Adverbial noun": "n-adv",
        "Temporal noun": "n-temp",
        "Noun - used as a suffix": "n",
        "Na-adjective": "adj-na",
        "No-adjective": "adj-no",
        "I-adjective": "adj-i",
        "Auxiliary adjective": "adj-aux",
        "Transitive verb": "v-trans",
        "intransitive verb": "v-intr",
        "Suru verb": "v-suru",
        "Suru verb - irregular": "v-suru",
        "Ichidan verb": "v1",
        "Godan verb with u ending": "v5u",
        "Godan verb with ku ending": "v5k",
        "Godan verb with gu ending": "v5g",
        "Godan verb with su ending": "v5s",
        "Godan verb with zu ending": "v5z",
        "Godan verb with tsu ending": "v5t",
        "Godan verb with nu ending": "v5n",
        "Godan verb with fu ending": "v5h",
        "Godan verb with bu ending": "v5b",
        "Godan verb with pu ending": "v5p",
        "Godan verb with mu ending": "v5m",
        "Godan verb with ru ending": "v5r",
        "Auxiliary verb": "v-aux",
        "Adverb": "adv",
        "Pronoun": "pron",
        "Conjunction": "conj",
        "Prefix": "pre",
        "Suffix": "suff",
        "Numeric": "num",
        "Counter": "count",
        "Place": "place",
        "Expression": "expr",
        "Wikipedia definition": None
    }
    if 'senses' in jisho_res:
        pos_list_list = [s['parts_of_speech'] for s in jisho_res['senses']]
    else:
        pos_list_list = [jisho_res['parts_of_speech']]

    pos_list = [set([abbrev_map.get(pos, pos) for pos in pos_list]) for pos_list in pos_list_list]

    pos = reduce(lambda a, b: a.union(b), pos_list)
    pos = [p for p in pos if p is not None]

    if len(pos) == 0:
        return None
    else:
        return f"({', '.join(sorted(pos))})"


def format_sense_item(sense_item):
    buf = []

    pos = format_parts_of_speech(sense_item)
    if pos is not None:
        buf.append(pos)

    if len(sense_item['restrictions']) != 0:
        buf.append(f"(only applies to {', '.join(sense_item['restrictions'])})")

    if len(sense_item['info']) != 0:
        buf.append(f"({', '.join(sense_item['info'])})")

    if "Usually written using kana alone" in sense_item['tags']:
        buf.append(f"(usually kana only)")

    buf.append("; ".join(sense_item['english_definitions']))

    return " ".join(buf)


def format_gloss_item(jisho_res, token=None, limit=1):
    if jisho_res is None or jisho_res['meta']['status'] != 200 or len(jisho_res['data']) == 0:
        return None

    gloss_buff = []
    jisho_res_list = jisho_res['data']
    for jisho_res in jisho_res_list[:limit]:
        buf = [format_gloss_title(jisho_res, token=token)]

        for i, sense_item in enumerate(jisho_res['senses']):
            buf.append(f" ({i+1}) {format_sense_item(sense_item)}")

        gloss_buff.append("".join(buf))

    return "\n\n".join(gloss_buff)


def format_furigana(token):
    if not contains_kanji(token.word_surface):
        return token.word_surface

    kanji_prefix = longest_common_prefix([stem_word(token), token.word_surface])
    if kanji_prefix == "" or not contains_kanji(token.word_surface):
        return token.word_surface

    if len(kanji_prefix) == 0:
        surface_postfix = ""
        base_postfix = stem_word(token)

    else:
        surface_postfix = token.word_surface[len(kanji_prefix):]
        base_postfix = stem_word(token)[len(kanji_prefix):]

    if len(base_postfix) == 0:
        stem_reading_prefix = stem_reading(token)
    else:
        stem_reading_prefix = stem_reading(token)[:-len(base_postfix)]

    buf = []
    kanji_blocks = []
    kanji_block = []
    char_seq = []

    last_kanji = False

    for c in kanji_prefix:
        if is_kanji(c) and not last_kanji:
            buf.append("(.+)")
            kanji_block.append(c)
            last_kanji = True
        elif is_kanji(c):
            kanji_block.append(c)
            pass
        else:
            if len(kanji_block) != 0:
                k = {"kanji": "".join(kanji_block), "furigana": None}
                kanji_blocks.append(k)
                char_seq.append(k)
                kanji_block = []

            buf.append(c)
            char_seq.append(c)
            last_kanji = False

    if len(kanji_block) != 0:
        k = {"kanji": "".join(kanji_block), "furigana": None}
        kanji_blocks.append(k)
        char_seq.append(k)

    char_seq.extend(surface_postfix)

    reg = "".join(buf)
    match = re.match(reg, stem_reading_prefix)

    if match is not None:
        for k, f in zip(kanji_blocks, match.groups()):
            k["furigana"] = f
    else:
        raise ValueError()

    buf = []
    for char in char_seq:
        if isinstance(char, dict) and char['furigana'] is not None and not contains_kanji(char['furigana']):
            buf.append(f" {char['kanji']}[{char['furigana']}]")
        elif isinstance(char, dict):
            buf.append(char['kanji'])
        else:
            buf.append(char)

    return "".join(buf)


def longest_common_prefix(strs):
    if len(strs) == 0:
        return ""
    longest_pre = ""

    shortest_str = min(strs, key=len)
    for i in range(len(shortest_str)):
        if all([x.startswith(shortest_str[:i+1]) for x in strs]):
            longest_pre = shortest_str[:i+1]
        else:
            break

    return longest_pre


def analyze_tokens(tokens):
    buf = []
    gloss_items = []
    for token in tokens:
        if token.word_surface.strip() == "EOS":
            continue

        if imis_info(token) is None:
            jisho_res = None
        else:
            jisho_res = query_jisho_by_token(token)

        buf.append(format_furigana(token))
        gi = format_gloss_item(jisho_res, token=token)
        if gi is not None:
            gloss_items.append(gi)

    return "".join(buf).strip(), gloss_items


def main():
    with open("shokugeki_no_soma_S01_E05.txt", "rb") as f:
        trans_lines = json.load(f)

    for line_nr, trans_line in enumerate(trans_lines):
        english_line = trans_line['en']
        japanese_line = trans_line['jp']

        tokens = tokenize([japanese_line])[0]
        reading, gloss_items = analyze_tokens(tokens)

        print(invoke("addNote", note={
            "deckName": "Series",
            "modelName": "Nayrs Japanese Core5000",
            "fields": {
                "Sound": f"[NO SOUND E01S05文{line_nr+1}]",
                "Expression": japanese_line,
                "Meaning": "",
                "Reading": reading,
                "English Translation": english_line,
                "Word": "",
                "Frequency Order": f"E01S05文{line_nr+1}",
                "Gloss": "<br><br>".join(gloss_items)
            },
            "tags": []
        }))

    # print(res)


def find_note_id(frequency_order):
    notes = invoke('findNotes', query=f"deck:Series \"Frequency Order:{frequency_order}\"")
    if len(notes) == 0:
        return None
    elif len(notes) > 1:
        raise ValueError("frequency_order in ambiguous")
    else:
        return notes[0]


def main2():
    note_id = find_note_id("E01S05文001")

    invoke('updateNoteFields', note={
        'id': note_id,
        'fields': {
            "Sound": ''
        },
        'tags': [
            'yomichan'
        ],
        'audio': [{
            'url': "/home/simon/Music/shokugeki_no_soma_S01E05_文001.mp3",
            'filename': "shokugeki_no_soma_S01E05_文001.mp3",
            "fields": ["Sound"]
        }]
    })

    [note] = invoke('notesInfo', notes=[note_id])
    while note['fields']['Sound']['value'] != f'[sound:shokugeki_no_soma_S01E05_文001]':
        print("update again")
        time.sleep(1.0)
        invoke('updateNoteFields', note={
            'id': note_id,
            'fields': {
                "Sound": f'[sound:shokugeki_no_soma_S01E05_文001]'
            }
        })


def main3():
    jisho_res = query_jisho("新聞")
    print(format_gloss_item(jisho_res, limit=10))

    # gi = gloss_item(token, jisho_res)


class Dialog:
    def __init__(self, number, time_from, time_to, lines, attributes=None):
        self.number = number
        self.time_from = time_from
        self.time_to = time_to
        self.lines = lines
        self.attributes = attributes


def to_dialogs(line_iter):
    dialogs = []
    buf_dialog = Dialog(None, None, None, [], attributes=None)
    for line in line_iter:
        if line.strip() == "":
            continue
        elif line.strip().split(" ")[0] in ["NOTE", "WEBVTT"]:
            continue
        else:
            match = re.match(r"^(\d+)$", line)
            if match is not None:
                if buf_dialog.number is not None:
                    dialogs.append(buf_dialog)
                    buf_dialog = Dialog(None, None, None, [], attributes=None)

                buf_dialog.number = int(match.group(1))
                continue

            match = re.match(r"^(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3}) (.*)$", line)
            if match is not None:
                time_from = total_milliseconds(datetime.strptime(match.group(1), "%H:%M:%S.%f").time())
                time_to = total_milliseconds(datetime.strptime(match.group(2), "%H:%M:%S.%f").time())
                attribute_str = match.group(3)
                attributes = attribute_str.split(" ")
                attributes = [a for a in attributes if len(a.strip()) != 0]
                attributes = {a[0]: a[1].split(",") for a in [att.split(":") for att in attributes]}

                buf_dialog.time_from = time_from
                buf_dialog.time_to = time_to
                buf_dialog.attributes = attributes
                continue

            buf_dialog.lines.append(line.strip())

    return dialogs


def preprocess_dialog(dialog, line_join_style='jp'):
    line = "".join(dialog.lines)
    tree = ET.fromstring(f"<root>{line}</root>")
    dialog_lines = list(tree.itertext())
    if line_join_style == 'jp':
        parsed_line = "".join([d.strip() for d in dialog_lines])
    elif line_join_style == 'en':
        parsed_line = " ".join([d.strip() for d in dialog_lines])
    else:
        raise ValueError(f"invalid line_join_style: {line_join_style}")

    while True:
        match = re.search(r"[(（][^(（）)]*[)）]", parsed_line)
        if match:
            parsed_line = "".join([parsed_line[:match.start()], parsed_line[match.end():]])
        else:
            break

    return parsed_line


def total_microseconds(t: time):
    return (t.hour * 60 * 60 * 1e6) + (t.minute * 60 * 1e6) + (t.second * 1e6) + t.microsecond


def total_milliseconds(t: time):
    return int(total_microseconds(t) / 1e3)


def time_mapping(t: int, dialogs_base: List[Dialog], dialogs_trans: List[Dialog], anchors: List[Tuple[int, int]]) -> int:
    if len(anchors) == 0:
        return t

    if t < find_dialog(anchors[0][0], dialogs_base).time_from:
        return t

    for base, next_base, trans in zip([a[0] for a in anchors], [a[0] for a in anchors[1:]], [a[1] for a in anchors]):
        if t < find_dialog(next_base, dialogs_base).time_from:
            diff = (find_dialog(trans, dialogs_trans).time_from - find_dialog(base, dialogs_base).time_from)
            return t + diff

    diff = (find_dialog(anchors[-1][1], dialogs_trans).time_from - find_dialog(anchors[-1][0], dialogs_base).time_from)
    return t + diff


def filter_dialogs_by_time_range(dialogs: List[Dialog], time_from: int, time_to: int, min_overlap=0.5) -> List[Dialog]:
    matches = []
    for dialog in dialogs:
        if time_to <= dialog.time_from or time_from >= dialog.time_to:
            continue

        dialog_len = dialog.time_to - dialog.time_from
        overlap_time = min(dialog.time_to, time_to) - max(dialog.time_from, time_from)
        if dialog_len > 0 and overlap_time / dialog_len > min_overlap:
            matches.append(dialog)

    return matches


def find_dialog(dialog_number: int, dialogs: List[Dialog]) -> Dialog:
    dialogs = [d for d in dialogs if d.number == dialog_number]

    if len(dialogs) == 0:
        raise IndexError(f"dialog number not found: {dialog_number}")
    elif len(dialogs) == 1:
        return dialogs[0]
    else:
        raise ValueError(f"dialog number is ambiguous: {dialog_number}")


def load_sections_file(path):
    with open(path, "r") as f:
        sections = json.load(f)

    with open(os.path.join(os.path.dirname(path), sections['base']['path'])) as f:
        dialogs_base = to_dialogs(f)

    dialogs_trans = {}
    anchors = {}
    for key, trans in sections['translations'].items():
        with open(os.path.join(os.path.dirname(path), trans['path'])) as f:
            dialogs_trans[key] = to_dialogs(f)
        anchors[key] = trans.get('anchors', [])

    return dialogs_base, dialogs_trans, anchors, sections['sections']


def main4():
    sections_path = "data/shokugeki_S01E05/sections.json"
    dialogs_base, dialogs_trans, anchors, sections = load_sections_file(sections_path)

    for seq_idx, section in enumerate(sections):
        sound_from = find_dialog(min(section), dialogs_base).time_from
        sound_to = find_dialog(max(section), dialogs_base).time_to

        match_from = time_mapping(find_dialog(min(section), dialogs_base).time_from, dialogs_base, dialogs_trans['en'], anchors['en'])
        match_to = time_mapping(find_dialog(max(section), dialogs_base).time_to, dialogs_base, dialogs_trans['en'], anchors['en'])

        f_trans_dialogs = filter_dialogs_by_time_range(dialogs_trans['en'], match_from, match_to)

        for section_item in section:
            print(find_dialog(section_item, dialogs_base).lines)

        for f_trans_dialog in f_trans_dialogs:
            print(f_trans_dialog.lines)

        sound = AudioSegment.from_mp3("/home/simon/Music/shokugeki_no_soma_S01E05_full.mp3")
        seg = sound[sound_from-100:sound_to+100]
        play(seg)
        # seg.export(f"seg_{seq_idx:03d}.mp3", format="mp3")


if __name__ == "__main__":
    main3()
