import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import time
from pydub import AudioSegment
import simpleaudio as sa
import svara_characterisation_context  # Asegúrate de tener este módulo disponible

# Modificación para usar la función real
def feature_extraction(audio_file, text_file=None):
    return svara_characterisation_context.characterisation_context(audio_file, text_file)

def divide_audio_tracks(audio_file):
    # Dummy implementation for demonstration
    time.sleep(2)  # Simulate processing time
    return ["track1.wav", "track2.wav"]

def isolate_voice(audio_file):
    # Dummy implementation for demonstration
    time.sleep(2)  # Simulate processing time
    return "isolated_voice.wav"

def detect_svara(audio_file):
    # Dummy implementation for demonstration
    time.sleep(2)  # Simulate processing time
    return ["svara1", "svara2"]

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Processing Interface")
        self.geometry("800x600")

        self.tab_control = ctk.CTkTabview(self, corner_radius=32)
        self.tab_control.pack(expand=1, fill="both")

        self.tab1 = self.tab_control.add("File Selection")
        self.tab2 = self.tab_control.add("Feature Extraction")
        self.tab3 = self.tab_control.add("Isolated Voice")
        self.tab4 = self.tab_control.add("Separated Svaras")
        self.tab5 = self.tab_control.add("Segmentation")

        self.create_tab1()
        self.create_tab2()
        self.create_tab3()
        self.create_tab4()
        self.create_tab5()

        self.loading_label = ctk.CTkLabel(self, text="", fg_color="#3a7ebf", text_color="black")
        self.loading_label.pack(side="bottom", fill="x")

    def create_tab1(self):
        frame1 = ctk.CTkFrame(self.tab1, fg_color= "#92A0AD", corner_radius=32)
        frame1.pack(side="left", expand=True, fill="both", padx=50, pady=125)

        frame2 = ctk.CTkFrame(self.tab1, fg_color= "#92A0AD", corner_radius=32)
        frame2.pack(side="right", expand=True, fill="both", padx=50, pady=125)

        frame3 = ctk.CTkFrame(self.tab1, corner_radius=32)
        frame3.pack(side="right", expand=True, fill="both", pady=10)

        self.audio_file_path = ctk.StringVar()
        self.text_file_path = ctk.StringVar()
        self.audio_file_path2 = ctk.StringVar()

        # Section 1
        section1_label = ctk.CTkLabel(frame1, text="ANALYSIS", font=("Montserrat", 15, "bold"), text_color=self._fg_color)
        section1_label.pack()
        audio_button = ctk.CTkButton(frame1, text="Select Audio File", command=self.select_audio_file, corner_radius=32)
        audio_button.pack(pady=15)

        text_button = ctk.CTkButton(frame1, text="Select Text File", command=self.select_text_file, corner_radius=32)
        text_button.pack(pady=15)

        run_button = ctk.CTkButton(frame1, text="Run", command=self.run_functions_section1, corner_radius=32)
        run_button.pack(pady=15)

        clear_button = ctk.CTkButton(frame3, text="Clear", command=self.clear_all, corner_radius=32)
        clear_button.pack(side="bottom", pady=10)

        # Section 2
        section2_label = ctk.CTkLabel(frame2, text="SEGMENTATION", font=("Montserrat", 15, "bold"), text_color=self._fg_color)
        section2_label.pack(pady=10)

        audio_button2 = ctk.CTkButton(frame2, text="Select Audio File", command=self.select_audio_file2, corner_radius=32)
        audio_button2.pack(pady=15)

        run_button2 = ctk.CTkButton(frame2, text="Run All", command=self.run_functions_section2, corner_radius=32)
        run_button2.pack(pady=15)

    def create_tab2(self):
        self.csv_file_path = ctk.StringVar()
        download_csv_button = ctk.CTkButton(self.tab2, text="Download CSV", command=self.download_csv, corner_radius=32)
        download_csv_button.pack(pady=10)

    def create_tab3(self):
        self.audio_frame = ctk.CTkFrame(self.tab3)
        self.audio_frame.pack(expand=True, fill="both", padx=10, pady=10)

        play_original_button = ctk.CTkButton(self.tab3, text="Play Original Audio", command=self.play_original_audio, corner_radius=32)
        play_original_button.pack(pady=10)

        play_isolated_button = ctk.CTkButton(self.tab3, text="Play Isolated Voice", command=self.play_isolated_audio, corner_radius=32)
        play_isolated_button.pack(pady=10)

        download_isolated_button = ctk.CTkButton(self.tab3, text="Download Isolated Voice", command=self.download_isolated_voice, corner_radius=32)
        download_isolated_button.pack(pady=10)

    def create_tab4(self):
        download_tracks_button = ctk.CTkButton(self.tab4, text="Download All Tracks", command=self.download_all_tracks, corner_radius=32)
        download_tracks_button.pack(pady=10)

    def create_tab5(self):
        self.svara_text = ctk.CTkTextbox(self.tab5)
        self.svara_text.pack(expand=True, fill="both", padx=10, pady=10)

        download_svara_button = ctk.CTkButton(self.tab5, text="Download Svara Text", command=self.download_svara_text, corner_radius=32)
        download_svara_button.pack(pady=10)

    def select_audio_file(self):
        self.audio_file_path.set(filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")]))

    def select_text_file(self):
        self.text_file_path.set(filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")]))

    def select_audio_file2(self):
        self.audio_file_path2.set(filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")]))

    def run_functions_section1(self):
        threading.Thread(target=self._run_functions_section1).start()

    def _run_functions_section1(self):
        self.set_loading(True)
        try:
            audio_file = self.audio_file_path.get()
            text_file = self.text_file_path.get()
            csv_file = feature_extraction(audio_file, text_file)
            isolated_voice = isolate_voice(audio_file)
            divided_tracks = divide_audio_tracks(audio_file)

            self.csv_file_path.set(csv_file)
            self.isolated_voice_path = isolated_voice
            self.divided_tracks = divided_tracks
        finally:
            self.set_loading(False)

    def run_functions_section2(self):
        threading.Thread(target=self._run_functions_section2).start()

    def _run_functions_section2(self):
        self.set_loading(True)
        try:
            audio_file = self.audio_file_path2.get()
            csv_file = feature_extraction(audio_file)
            isolated_voice = isolate_voice(audio_file)
            divided_tracks = divide_audio_tracks(audio_file)
            svara_list = detect_svara(audio_file)  # Run detect_svara here

            self.csv_file_path.set(csv_file)
            self.isolated_voice_path = isolated_voice
            self.divided_tracks = divided_tracks
            self.display_svara_list(svara_list)  # Display svara list
        finally:
            self.set_loading(False)

    def display_svara_list(self, svara_list):
        self.svara_text.delete("1.0", ctk.END)
        for svara in svara_list:
            self.svara_text.insert(ctk.END, f"{svara}\n")

    def download_csv(self):
        csv_file = self.csv_file_path.get()
        if csv_file:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if save_path:
                # For demonstration, just copy the file
                with open(csv_file, "r") as src, open(save_path, "w") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Download", "CSV file downloaded successfully!")
        else:
            messagebox.showwarning("Download", "No CSV file to download!")

    def play_audio(self, audio_file):
        if not audio_file:
            messagebox.showwarning("Play Audio", "No audio file selected!")
            return

        try:
            audio = AudioSegment.from_file(audio_file)
            play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
            play_obj.wait_done()
        except Exception as e:
            messagebox.showerror("Play Audio", f"Could not play audio file: {str(e)}")

    def play_original_audio(self):
        audio_file = self.audio_file_path.get()
        self.play_audio(audio_file)
        
    def play_isolated_audio(self):
        if hasattr(self, 'isolated_voice_path'):
            self.play_audio(self.isolated_voice_path)
        else:
            messagebox.showwarning("Play Audio", "No isolated voice file available!")

    def download_isolated_voice(self):
        if hasattr(self, 'isolated_voice_path'):
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Files", "*.wav")])
            if save_path:
                # For demonstration, just copy the file
                with open(self.isolated_voice_path, "rb") as src, open(save_path, "wb") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Download", "Isolated voice downloaded successfully!")
        else:
            messagebox.showwarning("Download", "No isolated voice file to download!")

    def download_all_tracks(self):
        if hasattr(self, 'divided_tracks'):
            save_folder = filedialog.askdirectory()
            if save_folder:
                for track in self.divided_tracks:
                    base_name = os.path.basename(track)
                    save_path = os.path.join(save_folder, base_name)
                    with open(track, "rb") as src, open(save_path, "wb") as dst:
                        dst.write(src.read())
                messagebox.showinfo("Download", "All tracks downloaded successfully!")
        else:
            messagebox.showwarning("Download", "No tracks to download!")

    def download_svara_text(self):
        svara_text = self.svara_text.get("1.0", ctk.END).strip()
        if svara_text:
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if save_path:
                with open(save_path, "w") as file:
                    file.write(svara_text)
                messagebox.showinfo("Download", "Svara text downloaded successfully!")
        else:
            messagebox.showwarning("Download", "No Svara text to download!")

    def clear_all(self):
        self.audio_file_path.set("")
        self.text_file_path.set("")
        self.audio_file_path2.set("")
        self.csv_file_path.set("")
        self.isolated_voice_path = ""
        self.divided_tracks = []
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.svara_text.delete("1.0", ctk.END)
        messagebox.showinfo("Clear", "All fields have been cleared!")

    def set_loading(self, is_loading):
        self.loading_label.configure(text="Loading..." if is_loading else "")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = App()
    app.mainloop()