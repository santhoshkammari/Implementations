import threading
import tempfile
import os
import wave
import pyaudio
import whisper
import customtkinter as ctk
import requests
import json

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")


class SimpleWhisperRecorder:
    def __init__(self):
        # Initialize the main window
        self.root = ctk.CTk()
        self.root.title("Simple Whisper Recorder")
        self.root.geometry("500x800")  # Increased height for response display
        self.root.resizable(False, False)

        # Variables
        self.is_recording = False
        self.frames = []
        self.model = None
        self.api_url = "http://localhost:8000/process_text"  # FastAPI endpoint

        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

        # Create UI
        self.setup_ui()

        # Load Whisper model in background
        self.update_status("Loading Whisper model...")
        self.load_model_thread = threading.Thread(target=self.load_whisper_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def setup_ui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # App title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Whisper + LLM Voice Assistant",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.title_label.pack(pady=(0, 15))

        # Record button
        self.record_button = ctk.CTkButton(
            self.main_frame,
            text="Record",
            font=ctk.CTkFont(size=14),
            width=120,
            height=40,
            corner_radius=8,
            command=self.toggle_recording,
            fg_color="#1f6aa5",
            hover_color="#2a7bbf"
        )
        self.record_button.pack(pady=10)

        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        self.status_label.pack(pady=5)

        # Frame for transcription and response
        self.text_frame = ctk.CTkFrame(self.main_frame)
        self.text_frame.pack(fill="both", expand=True, pady=10)

        # Transcription label
        self.transcription_label = ctk.CTkLabel(
            self.text_frame,
            text="Your speech:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        self.transcription_label.pack(anchor="w", padx=10, pady=(10, 0))

        # Transcription text display
        self.transcription_text = ctk.CTkTextbox(
            self.text_frame,
            height=80,
            wrap="word"
        )
        self.transcription_text.pack(fill="x", padx=10, pady=5)

        # LLM Response label
        self.response_label = ctk.CTkLabel(
            self.text_frame,
            text="LLM Response:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        self.response_label.pack(anchor="w", padx=10, pady=(10, 0))

        # LLM Response text display
        self.response_text = ctk.CTkTextbox(
            self.text_frame,
            height=80,
            wrap="word"
        )
        self.response_text.pack(fill="x", padx=10, pady=5)

    def load_whisper_model(self):
        """Load the Whisper model in background"""
        try:
            self.model = whisper.load_model("small")
            self.update_status("Ready")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def update_status(self, message):
        """Update status message"""
        self.status_label.configure(text=message)

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
                fg_color="#1f6aa5",
                hover_color="#2a7bbf"
            )
            self.update_status("Processing speech...")
        else:
            # Start recording
            self.is_recording = True
            self.record_button.configure(
                text="Stop",
                fg_color="#b3001b",
                hover_color="#cc0020"
            )
            self.update_status("Recording...")

            # Clear previous text
            self.transcription_text.delete("1.0", "end")
            self.response_text.delete("1.0", "end")

            # Start recording in a thread
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
            # Collect audio frames while recording
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"Error recording: {str(e)}")
            self.update_status(f"Error: {str(e)}")
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()

            # Process the audio if we have frames
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
            self.update_status("Transcribing...")
            result = self.model.transcribe(temp_filename)
            text = result["text"].strip()

            if text:
                # Display the transcription
                print("\nTranscription:")
                print(text)
                print("-" * 50)

                # Update UI with transcription
                self.transcription_text.delete("1.0", "end")
                self.transcription_text.insert("1.0", text)

                # Send to LLM API
                self.update_status("Getting LLM response...")
                self.get_llm_response(text)
            else:
                print("No speech detected")
                self.update_status("No speech detected")
        except Exception as e:
            print(f"Error: {str(e)}")
            self.update_status(f"Error: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

    def get_llm_response(self, transcribed_text):
        """Send transcribed text to LLM API and handle response"""
        try:
            # Prepare request data
            payload = {"text": transcribed_text}

            # Send request to FastAPI server
            response = requests.post(self.api_url, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                llm_response = response_data.get("response", "No response from LLM")

                # Display LLM response
                print("LLM Response:")
                print(llm_response)
                print("-" * 50)

                # Update UI with LLM response
                self.response_text.delete("1.0", "end")
                self.response_text.insert("1.0", llm_response)
                self.update_status("Ready")
            else:
                error_msg = f"API error: {response.status_code}"
                print(error_msg)
                self.response_text.delete("1.0", "end")
                self.response_text.insert("1.0", f"Error: {error_msg}")
                self.update_status(error_msg)

        except Exception as e:
            error_msg = f"Failed to get LLM response: {str(e)}"
            print(error_msg)
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", f"Error: {error_msg}")
            self.update_status(error_msg)

    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Clean up on window closing"""
        self.is_recording = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        self.audio.terminate()
        self.root.destroy()


def main():
    # Try to ensure required packages are installed
    try:
        import whisper
        import customtkinter
        import pyaudio
        import requests
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "openai-whisper", "pyaudio", "customtkinter", "requests"])
        print("Packages installed. Starting application...")

    app = SimpleWhisperRecorder()
    app.run()


if __name__ == "__main__":
    main()