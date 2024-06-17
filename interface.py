import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import svara_characterisation_2_context
import svara_characterisation_3_labels


# ModificaciÃ³n para usar las dos funciones
def feature_extraction(audio_file, text_file=None):
    context_csv = svara_characterisation_2_context.characterisation_context(audio_file, text_file)
    labels_csv = svara_characterisation_3_labels.characterisation_labels(audio_file, text_file)
    return context_csv, labels_csv

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Processing Interface")
        self.geometry("800x600")

        self.tab_control = ctk.CTkTabview(self, corner_radius=32)
        self.tab_control.pack(expand=1, fill="both")

        self.tab1 = self.tab_control.add("File Selection")
        self.tab2 = self.tab_control.add("Feature Extraction")

        self.create_tab1()
        self.create_tab2()

        self.loading_label = ctk.CTkLabel(self, text="", fg_color="#3a7ebf", text_color="black")
        self.loading_label.pack(side="bottom", fill="x")

    def create_tab1(self):
        frame1 = ctk.CTkFrame(self.tab1, fg_color= "#92A0AD", corner_radius=32)
        frame1.pack(expand=True, fill="both", padx=100, pady=150)


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

        clear_button = ctk.CTkButton(frame1, text="Clear", command=self.clear_all, corner_radius=32)
        clear_button.pack(side="bottom")

    def create_tab2(self):
        self.context_csv_file_path = ctk.StringVar()
        self.labels_csv_file_path = ctk.StringVar()

        # Dropdown for feature extraction options
        options = ["Features with context", "Features with labels"]
        self.feature_extraction_option = ctk.CTkComboBox(self.tab2, values=options, corner_radius=32)
        self.feature_extraction_option.pack(pady=10)
        self.feature_extraction_option.set("Select")  # Set default selection

        # Button to download CSV
        download_csv_button = ctk.CTkButton(self.tab2, text="Download CSV", command=self.download_csv, corner_radius=32)
        download_csv_button.pack(pady=10)


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
            context_csv, labels_csv = feature_extraction(audio_file, text_file)

            self.context_csv_file_path.set(context_csv)
            self.labels_csv_file_path.set(labels_csv)
        finally:
            self.set_loading(False)


    def download_csv(self):
        selected_option = self.feature_extraction_option.get()
        if selected_option == "Features with context":
            csv_file = self.context_csv_file_path.get()
        elif selected_option == "Features with labels":
            csv_file = self.labels_csv_file_path.get()
        else:
            csv_file = None

        if csv_file:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if save_path:
                # For demonstration, just copy the file
                with open(csv_file, "r") as src, open(save_path, "w") as dst:
                    dst.write(src.read())
                messagebox.showinfo("Download", "CSV file downloaded successfully!")
        else:
            messagebox.showwarning("Download", "No CSV file to download!")


    def clear_all(self):
        self.audio_file_path.set("")
        self.text_file_path.set("")
        self.audio_file_path2.set("")
        self.context_csv_file_path.set("")
        self.labels_csv_file_path.set("")
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