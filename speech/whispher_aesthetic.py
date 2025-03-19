import os
import tempfile
import threading
import wave
from datetime import datetime

import pyaudio
import whisper
import customtkinter as ctk
import numpy as np

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")


class VoiceWaveform(ctk.CTkCanvas):
    """Custom canvas for animated voice waveform visualization"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg='black', highlightthickness=0)

        # Waveform parameters
        self.active = False
        self.bars = 35  # More bars for smoother appearance
        self.bar_width = 3
        self.bar_spacing = 2
        self.max_height = 40
        self.animation_speed = 60

        # Initialize bars with random heights
        self.heights = [0] * self.bars
        self.animation_id = None

    def start_animation(self):
        """Start waveform animation"""
        self.active = True
        self.animate()

    def stop_animation(self):
        """Stop waveform animation"""
        self.active = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None

        # Reset all bars to zero height with a fade-out effect
        self.fadeout()

    def fadeout(self):
        """Fade out the waveform animation"""

        def fade_step():
            still_visible = False
            for i in range(self.bars):
                if self.heights[i] > 0:
                    self.heights[i] = max(0, self.heights[i] - 2)
                    still_visible = True

            self.redraw()
            if still_visible:
                self.animation_id = self.after(30, fade_step)
            else:
                self.animation_id = None

        if self.animation_id:
            self.after_cancel(self.animation_id)
        self.animation_id = self.after(30, fade_step)

    def animate(self):
        """Animate the waveform bars"""
        if not self.active:
            return

        # Update heights with smooth transitions
        for i in range(self.bars):
            if np.random.random() > 0.5:
                # Move toward a new random target
                target = np.random.randint(0, self.max_height)
                self.heights[i] += (target - self.heights[i]) * 0.15

        self.redraw()
        self.animation_id = self.after(self.animation_speed, self.animate)

    def redraw(self):
        """Redraw the waveform visualization"""
        self.delete("all")

        # Get dimensions
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1:  # Not yet properly sized
            return

        # Calculate bar positions
        total_width = self.bars * (self.bar_width + self.bar_spacing) - self.bar_spacing
        start_x = (width - total_width) // 2

        # Draw bars
        for i in range(self.bars):
            x = start_x + i * (self.bar_width + self.bar_spacing)
            bar_height = max(1, int(self.heights[i]))

            # Create gradient effect (lighter at top)
            gradient_steps = 10
            for step in range(gradient_steps):
                step_height = bar_height * (step + 1) / gradient_steps
                y1 = height / 2 - step_height / 2
                y2 = height / 2 + step_height / 2

                # Calculate color (blue to cyan gradient)
                opacity = 0.3 + 0.7 * (step / gradient_steps)

                # Only draw if the step is visible
                if step_height >= 1:
                    color = self.calculate_color(opacity)
                    self.create_rectangle(
                        x, y1, x + self.bar_width, y2,
                        fill=color, outline="", width=0
                    )

    def calculate_color(self, opacity):
        """Calculate color hex value with opacity"""
        # Blue to cyan gradient
        rgb = (0, int(100 + opacity * 155), 255)
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


class SleekWhisperRecorder(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Whisper Voice Recorder")
        self.geometry("520x400")  # Smaller, more compact size
        self.minsize(500, 350)

        # Variables
        self.is_recording = False
        self.audio_thread = None
        self.frames = []
        self.model = None
        self.transcript_history = []

        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

        # Create UI
        self.create_ui()

        # Load Whisper model
        self.update_status("Loading Whisper model...")
        self.load_model_thread = threading.Thread(target=self.load_whisper_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def create_ui(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header frame
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        header_frame.grid_columnconfigure(0, weight=1)

        # App title - minimal but elegant
        title_label = ctk.CTkLabel(
            header_frame,
            text="Whisper Voice Recorder",
            font=("Helvetica", 18, "bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        # Small status indicator
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="Ready",
            font=("Helvetica", 12),
            text_color="#888888"
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        # Main content frame
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)

        # Top controls area
        control_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        control_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10))
        control_frame.grid_columnconfigure(1, weight=1)

        # Voice waveform - compact but visible
        self.waveform = VoiceWaveform(
            control_frame,
            height=60,
            width=300
        )
        self.waveform.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        # Button frame - centered
        button_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)

        # Record button
        self.record_button = ctk.CTkButton(
            button_frame,
            text="Record",
            font=("Helvetica", 14),
            width=120,
            height=36,
            corner_radius=18,
            command=self.toggle_recording
        )
        self.record_button.grid(row=0, column=1, padx=5)

        # Button row for Clear and Save
        action_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        action_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

        # Clear button - smaller and subtle
        self.clear_button = ctk.CTkButton(
            action_frame,
            text="Clear",
            font=("Helvetica", 12),
            width=90,
            height=30,
            corner_radius=15,
            fg_color="#555555",
            hover_color="#666666",
            command=self.clear_transcript
        )
        self.clear_button.grid(row=0, column=0, sticky="e", padx=5, pady=5)

        # Save button - smaller and subtle
        self.save_button = ctk.CTkButton(
            action_frame,
            text="Save",
            font=("Helvetica", 12),
            width=90,
            height=30,
            corner_radius=15,
            fg_color="#226b80",
            hover_color="#2d8aa0",
            command=self.save_transcript
        )
        self.save_button.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Transcript area
        transcript_frame = ctk.CTkFrame(content_frame)
        transcript_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        transcript_frame.grid_columnconfigure(0, weight=1)
        transcript_frame.grid_rowconfigure(1, weight=1)

        # Transcript header
        transcript_header = ctk.CTkFrame(transcript_frame, fg_color="transparent")
        transcript_header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(
            transcript_header,
            text="Transcript",
            font=("Helvetica", 14, "bold")
        ).grid(row=0, column=0, sticky="w")

        # Transcript text area
        self.transcript_text = ctk.CTkTextbox(
            transcript_frame,
            font=("Helvetica", 13),
            wrap="word",
            corner_radius=5
        )
        self.transcript_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def load_whisper_model(self):
        """Load the Whisper model in a background thread"""
        try:
            self.model = whisper.load_model("base")
            self.after(100, lambda: self.update_status("Ready"))
        except Exception as e:
            self.after(100, lambda: self.update_status(f"Error: {str(e)}"))

    def toggle_recording(self):
        """Start or stop recording"""
        if self.model is None:
            self.update_status("Please wait for model to load...")
            return

        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.record_button.configure(
                text="Record",
                fg_color="#1f6aa5"
            )
            self.waveform.stop_animation()
            self.update_status("Processing speech...")
        else:
            # Start recording
            self.is_recording = True
            self.record_button.configure(
                text="Stop",
                fg_color="#b3001b"
            )
            self.waveform.start_animation()
            self.update_status("Recording...")

            # Start recording in a new thread
            self.audio_thread = threading.Thread(target=self.record_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()

    def record_audio(self):
        """Record audio from microphone"""
        self.frames = []
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        try:
            # Collect audio frames while recording flag is True
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            self.after(100, lambda: self.update_status(f"Error: {str(e)}"))
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()

            # Process if we have recorded something
            if self.frames:
                self.process_audio()

    def process_audio(self):
        """Process the recorded audio with Whisper"""
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        try:
            # Transcribe with Whisper
            result = self.model.transcribe(temp_filename)
            text = result["text"].strip()

            if text:
                # Add to transcript history
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.transcript_history.append({"time": timestamp, "text": text})

                # Add to transcript text
                self.add_transcript(timestamp, text)
                self.update_status("Transcription complete")
            else:
                self.update_status("No speech detected")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

    def add_transcript(self, timestamp, text):
        """Add new transcript entry to the text area"""
        self.transcript_text.configure(state="normal")
        current_text = self.transcript_text.get("1.0", "end-1c")

        # Add formatting
        if current_text:
            self.transcript_text.insert("end", "\n\n")

        # Clean up time formatting for a more elegant look
        time_str = f"[{timestamp}]"

        self.transcript_text.insert("end", time_str, "timestamp")
        self.transcript_text.insert("end", "\n" + text, "transcript")

        # Configure tags for styling
        self.transcript_text.tag_configure(
            "timestamp",
            font=("Helvetica", 11, "bold"),
            foreground="#1f6aa5"
        )
        self.transcript_text.tag_configure(
            "transcript",
            font=("Helvetica", 13)
        )

        self.transcript_text.configure(state="disabled")
        self.transcript_text.see("end")

    def update_status(self, message):
        """Update status message"""
        self.status_label.configure(text=message)

    def clear_transcript(self):
        """Clear the transcript"""
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.configure(state="disabled")
        self.transcript_history = []
        self.update_status("Transcript cleared")

    def save_transcript(self):
        """Save the transcript to a file"""
        if not self.transcript_history:
            self.update_status("Nothing to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("WHISPER SPEECH TRANSCRIPT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for entry in self.transcript_history:
                    f.write(f"[{entry['time']}]\n")
                    f.write(f"{entry['text']}\n\n")

            self.update_status(f"Saved to {filename}")
        except Exception as e:
            self.update_status(f"Error saving: {str(e)}")

    def on_closing(self):
        """Clean up resources on closing"""
        self.is_recording = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        self.audio.terminate()
        self.destroy()


def main():
    # Try to ensure required packages are installed
    try:
        import whisper
        import customtkinter
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "openai-whisper", "pyaudio", "customtkinter", "numpy"])
        print("Packages installed. Starting application...")

    app = SleekWhisperRecorder()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()