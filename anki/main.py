import tkinter as tk
import tkinter.font as font
from anki.connect import load_sections_file, preprocess_dialog, Dialog, tokenize, analyze_tokens, \
    time_mapping, filter_dialogs_by_time_range, query_jisho, format_gloss_item, invoke
from typing import List
from pydub import AudioSegment
from threading import Thread
import simpleaudio
import os


class CustomText(tk.Text):
    def __init__(self, *args, **kwargs):
        """A text widget that report on internal widget commands"""
        tk.Text.__init__(self, *args, **kwargs)

        self.modify_source = None
        # create a proxy for the underlying widget
        self._orig = self._w + "_orig"
        self.tk.call("rename", self._w, self._orig)
        self.tk.createcommand(self._w, self._proxy)

    def _proxy(self, command, *args):
        cmd = (self._orig, command) + args
        result = self.tk.call(cmd)

        if command in ("insert", "delete", "replace"):
            modify_source = self.modify_source if self.modify_source is not None else 'unknown'
            self.event_generate(f"<<TextModified-{modify_source}>>")
            self.event_generate("<<TextModified>>")

        return result

    def delete(self, *args, modify_source=None, **kwargs):
        self.modify_source = modify_source
        try:
            super().delete(*args, **kwargs)
        finally:
            self.modify_source = None

    def insert(self, *args, modify_source=None, **kwargs):
        self.modify_source = modify_source
        try:
            super().insert(*args, **kwargs)
        finally:
            self.modify_source = None


class PlayThread(Thread):
    def __init__(self, segment):
        super().__init__()
        self.segment = segment
        self.daemon = True
        self.playback = None

    def run(self) -> None:
        if self.segment is not None:
            self.playback = simpleaudio.play_buffer(
                self.segment.raw_data,
                num_channels=self.segment.channels,
                bytes_per_sample=self.segment.sample_width,
                sample_rate=self.segment.frame_rate
            )
            self.playback.wait_done()

    def stop_playback(self):
        if self.playback is not None and self.playback.is_playing():
            self.playback.stop()


class MainWindow(tk.Canvas):
    def __init__(self, master, dialogs: List[Dialog], trans_dialogs: List[Dialog], anchors, sound, tokens,
                 num_buttons=10):
        super().__init__(master=master)
        self.dialogs = dialogs
        self.trans_dialogs = trans_dialogs
        self.anchors = anchors
        self.sound = sound
        self.tokens = tokens
        self.num_buttons = num_buttons
        self.active_playback = None
        self.btns = []
        self.btn_vars = []
        self.binds = []
        self.toggled = [False for _ in range(len(dialogs))]
        self.pos = 0
        self.frm_buttons = self.frm_top = self.frm_bottom = self.frm_data = self.frm_query = self.menubar = \
            self.btn_text_linked = self.txt_normal_text = self.txt_furi_text = self.txt_trans_text = \
            self.txt_gloss_text = self.ent_seq_nr = self.ent_query = self.txt_query_res = self.btn_send_query = \
            self.btn_add_note = self.scl_offset_before = self.scl_offset_after = self.btn_replay = None
        self.var_ent_query = tk.StringVar()
        self.var_ent_seq_nr = tk.StringVar()
        self.var_scl_offset_before = tk.DoubleVar()
        self.var_scl_offset_after = tk.DoubleVar()

        self.config_layout()
        self.update_scroll_pos()
        self.text_linked = True

    def config_layout(self):
        self.frm_top = tk.Frame(master=self)
        self.frm_buttons = tk.Frame(master=self.frm_top, height=100, width=500)

        btn_font = font.Font(size=15)
        for i in range(10):
            btn_var = tk.StringVar()
            btn = tk.Button(self.frm_buttons, textvariable=btn_var, wraplength=500, width=40, anchor='w',
                            bg='lightgray', command=self.toggle_fun(i), font=btn_font, justify=tk.LEFT, relief='raised')
            btn.pack(fill=tk.X, expand=True)

            self.btns.append(btn)
            self.btn_vars.append(btn_var)
        self.frm_buttons.bind("<Enter>", self.bind_button_scroll)
        self.frm_buttons.bind("<Leave>", self.unbind_button_scroll)
        self.frm_buttons.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        self.btn_text_linked = tk.Button(master=self.frm_top, text="<->", bg='lightgray', relief='sunken',
                                         command=self.toggle_text_linked)
        self.btn_text_linked.pack(side=tk.LEFT)

        self.frm_data = tk.Frame(master=self.frm_top, width=500)

        self.ent_seq_nr = tk.Entry(master=self.frm_data, textvariable=self.var_ent_seq_nr)
        self.ent_seq_nr.pack(side=tk.TOP)

        self.txt_normal_text = CustomText(master=self.frm_data, height=3)
        self.txt_normal_text.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.txt_normal_text.bind("<<TextModified-unknown>>", self.unset_text_link)
        self.txt_normal_text.bind("<<TextModified>>", self.update_text_boxes)

        self.txt_furi_text = CustomText(master=self.frm_data, height=3)
        self.txt_furi_text.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.txt_furi_text.bind("<<TextModified-unknown>>", self.unset_text_link)

        self.txt_trans_text = CustomText(master=self.frm_data, height=3)
        self.txt_trans_text.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.txt_trans_text.bind("<<TextModified-unknown>>", self.unset_text_link)

        self.txt_gloss_text = CustomText(master=self.frm_data, height=10)
        self.txt_gloss_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.txt_gloss_text.bind("<<TextModified-unknown>>", self.unset_text_link)

        self.frm_data.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.frm_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frm_bottom = tk.Frame(master=self)
        self.frm_query = tk.Frame(master=self.frm_bottom, height=200)
        self.scl_offset_before = tk.Scale(master=self.frm_query, label="offset before", showvalue=1,
                                          variable=self.var_scl_offset_before, from_=-1000, to=1000,
                                          length=500, resolution=50, tickinterval=250, orient=tk.HORIZONTAL)
        self.scl_offset_before.grid(row=0, column=0)
        self.scl_offset_after = tk.Scale(master=self.frm_query, label="offset after", showvalue=1,
                                         variable=self.var_scl_offset_after, from_=-1000, to=1000,
                                         length=500, resolution=50, tickinterval=250, orient=tk.HORIZONTAL)
        self.scl_offset_after.grid(row=0, column=1)
        self.ent_query = tk.Entry(master=self.frm_query, textvariable=self.var_ent_query)
        self.ent_query.grid(row=1, column=0, sticky=tk.W)
        self.txt_query_res = tk.Text(master=self.frm_query, height=3)
        self.txt_query_res.grid(row=2, column=0, sticky=tk.W)
        self.btn_send_query = tk.Button(master=self.frm_query, text="search", command=self.send_query)
        self.btn_send_query.grid(row=3, column=0, sticky=tk.W, pady=4)
        self.btn_add_note = tk.Button(master=self.frm_query, text="add note", command=self.add_note)
        self.btn_add_note.grid(row=3, column=1, sticky=tk.W, pady=4)
        self.btn_replay = tk.Button(master=self.frm_query, text="replay", command=lambda: self.play_selection())
        self.btn_replay.grid(row=4, column=0, sticky=tk.W, pady=4)
        self.frm_query.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.frm_bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def bind_all(self, sequence=None, func=None, add=None):
        bind = super().bind_all(sequence, func, add)
        self.binds.append((bind, sequence))
        return bind

    def unbind_all_all(self):
        while len(self.binds) != 0:
            bind, sequence = self.binds.pop(0)
            self.unbind(sequence, bind)

    def bind_button_scroll(self, _=None):
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def unbind_button_scroll(self, _=None):
        self.unbind_all_all()

    def update_scroll_pos(self):
        for idx, var in enumerate(self.btn_vars):
            pos = int(self.pos) + idx
            var.set(preprocess_dialog(self.dialogs[pos]))

    def update_toggle(self):
        for idx, var in enumerate(self.btn_vars):
            pos = int(self.pos) + idx
            if self.toggled[pos]:
                self.btns[idx].config(relief="sunken", bg='white')
            else:
                self.btns[idx].config(relief="raised", bg='lightgray')

    def toggled_dialogs(self):
        dialogs = []
        for i, t in enumerate(self.toggled):
            if t:
                dialogs.append(self.dialogs[i])

        return dialogs

    def send_query(self):
        jisho_res = query_jisho(self.var_ent_query.get())
        gi = format_gloss_item(jisho_res)
        if gi is not None:
            self.txt_query_res.delete("1.0", tk.END)
            self.txt_query_res.insert(tk.END, gi)

    def add_note(self):
        try:
            seq_nr = int(self.var_ent_seq_nr.get())
        except ValueError:
            seq_nr = None

        anki_media_collection_folder = "/home/simon/.local/share/Anki2/User 1/collection.media"
        if seq_nr is not None:
            seg = self.selected_segment()

            sound_file = f"jane_the_virgin_S01E01_{seq_nr:03d}.mp3"
            seg.export(os.path.join(anki_media_collection_folder, sound_file), format="mp3")

            invoke("addNote", note={
                "deckName": "Jane the Virgin",
                "modelName": "Nayrs Japanese Core5000",
                "fields": {
                    "Sound": f"[sound:{sound_file}]",
                    "Expression": self.txt_normal_text.get(1.0, tk.END),
                    "Meaning": "",
                    # "Reading": self.txt_furi_text.get(1.0, tk.END),
                    "Reading": self.txt_normal_text.get(1.0, tk.END),
                    "English Translation": self.txt_trans_text.get(1.0, tk.END),
                    "Word": "",
                    "Frequency Order": f"E01S01-{seq_nr:03d}",
                    "Gloss": self.txt_gloss_text.get(1.0, tk.END).replace("\n", "<br>"),
                },
                "tags": []
            })

        if seq_nr is not None:
            self.var_ent_seq_nr.set(str(seq_nr+1))
        else:
            self.var_ent_seq_nr.set(0)

    def update_text(self):
        if self.text_linked:
            buf = []
            for d in self.toggled_dialogs():
                buf.append(preprocess_dialog(d, line_join_style='en'))
            self.txt_normal_text.delete("1.0", tk.END, modify_source='auto-update')
            # self.txt_normal_text.insert(tk.END, "".join(buf), modify_source='auto-update')
            self.txt_normal_text.insert(tk.END, " ".join(buf), modify_source='auto-update')

    def unset_text_link(self, _=None):
        self.text_linked = False

    def update_text_boxes(self, _=None):
        if self.text_linked:
            tokens = []
            for i, t in enumerate(self.toggled):
                if t:
                    tokens.extend(self.tokens[i])
        else:
            normal_text = self.txt_normal_text.get(1.0, tk.END)
            # tokens = tokenize([normal_text])[0]

        # reading, gloss_items = analyze_tokens(tokens)

        # self.txt_furi_text.delete("1.0", tk.END, modify_source='auto-update')
        # self.txt_furi_text.insert(tk.END, reading, modify_source='auto-update')

        # self.txt_gloss_text.delete("1.0", tk.END, modify_source='auto-update')
        # self.txt_gloss_text.insert(tk.END, "\n\n".join(gloss_items), modify_source='auto-update')

        if self.text_linked:
            self.txt_trans_text.delete("1.0", tk.END, modify_source='auto-update')
            toggled_dialogs = self.toggled_dialogs()
            if len(toggled_dialogs) != 0:
                match_from = time_mapping(toggled_dialogs[0].time_from, self.dialogs, self.trans_dialogs, self.anchors)
                match_to = time_mapping(toggled_dialogs[-1].time_to, self.dialogs, self.trans_dialogs, self.anchors)

                f_trans_dialogs = filter_dialogs_by_time_range(self.trans_dialogs, match_from, match_to)
                # translated_text = " ".join([preprocess_dialog(d, line_join_style='en') for d in f_trans_dialogs])
                translated_text = " ".join([preprocess_dialog(d, line_join_style='jp') for d in f_trans_dialogs])

                self.txt_trans_text.insert(tk.END, translated_text, modify_source='auto-update')

    def selected_segment(self):
        min_dialog = None
        max_dialog = None
        for i, t in enumerate(self.toggled):
            if t:
                max_dialog = i
                if min_dialog is None:
                    min_dialog = i

        if min_dialog is not None and max_dialog is not None:
            sound_from = self.dialogs[min_dialog].time_from
            sound_to = self.dialogs[max_dialog].time_to

            offset_before = self.var_scl_offset_before.get()
            offset_after = self.var_scl_offset_after.get()

            seg = self.sound[max(sound_from+offset_before, 0):min(sound_to+offset_after, len(self.sound-1))]
            return seg
        else:
            return None

    def play_selection(self):
        seg = self.selected_segment()

        if self.active_playback is not None:
            self.active_playback.stop_playback()

        self.active_playback = PlayThread(seg)
        self.active_playback.start()

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.pos = max(self.pos - 1.0, 0)
        elif event.num == 5:
            self.pos = min(self.pos + 1.0, len(self.dialogs) - self.num_buttons)

        self.update_scroll_pos()
        self.update_toggle()

    @property
    def text_linked(self):
        return self.__text_linked

    @text_linked.setter
    def text_linked(self, text_linked):
        self.__text_linked = text_linked

        if self.text_linked:
            self.btn_text_linked.config(relief="sunken")
        else:
            self.btn_text_linked.config(relief="raised")

    def toggle_text_linked(self):
        self.text_linked = True if not self.text_linked else False

        if self.text_linked:
            self.update_text()

    def toggle_fun(self, idx):
        def fun():
            self.toggled[int(self.pos) + idx] = True if not self.toggled[int(self.pos) + idx] else False
            self.update_toggle()
            self.update_text()
            self.unbind_all_all()
            self.bind_all("<MouseWheel>", self._on_mousewheel)
            self.bind_all("<Button-4>", self._on_mousewheel)
            self.bind_all("<Button-5>", self._on_mousewheel)
            self.after(200, self.play_selection)

        return fun


def main():
    sections_path = "data/jane_the_virgin_S01E01/sections.json"
    dialogs_base, dialogs_trans, anchors, sections = load_sections_file(sections_path)

    # preprocessed_lines = [preprocess_dialog(d) for d in dialogs_base]
    preprocessed_lines = [preprocess_dialog(d, line_join_style='en') for d in dialogs_base]

    tokens = tokenize(preprocessed_lines)

    sound = AudioSegment.from_mp3("/home/simon/repos/anki/data/jane_the_virgin_S01E01/jane_the_virgin_S01E01.mp3")

    window = tk.Tk()
    menubar = tk.Menu(master=window)
    filemenu = tk.Menu(master=menubar, tearoff=0)
    filemenu.add_command(label="Open")
    menubar.add_cascade(label="File", menu=filemenu)
    window.config(menu=menubar)

    main_window = MainWindow(master=window, dialogs=dialogs_base, trans_dialogs=dialogs_trans['ja'],
                             anchors=anchors['ja'], sound=sound, tokens=tokens)
    main_window.pack()

    window.mainloop()


if __name__ == "__main__":
    main()
