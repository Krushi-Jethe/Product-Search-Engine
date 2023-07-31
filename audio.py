import speech_recognition as sr

def audio_to_text():
    
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")

        try:
            # Adjust for ambient noise for better recognition
            recognizer.adjust_for_ambient_noise(source)

            # Capture audio from the microphone
            audio = recognizer.listen(source)

            # Use Google Web Speech API to recognize the speech
            text = recognizer.recognize_google(audio)

            # Print the recognized text
            # print("You said:", text)
            
            return text

        except sr.UnknownValueError:
           
            return "Google Web Speech API could not understand the audio."
        
        except sr.RequestError as e:

            return f"Could not request results from Google Web Speech API; {e}"

# if __name__ == "__main__":
#     audio_to_text()
