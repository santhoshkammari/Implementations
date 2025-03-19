import speech_recognition as sr


def audio_to_text():
    # Create a recognizer instance
    recognizer = sr.Recognizer()

    # Use the microphone as source
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Listen for audio input
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")


if __name__ == "__main__":
    audio_to_text()