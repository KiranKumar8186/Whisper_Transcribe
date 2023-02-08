import streamlit as st
import whisper
import time
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

st.set_page_config(
    page_title="Elite_Notes",
    page_icon=":clipboard:",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: red; background: lightgrey;'>Elite Notes</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='padding: 10px; text-align: center; background: lightgrey; color: lightblack; font-size: 15px; margin : 15px auto;'>Upload Your Audio Files, Get The Transcription And Save Your Time</h1>", unsafe_allow_html=True)

st.title("Audio Transcribing")
st.markdown("---")


#------------------------------------------------
## Downloading Models

model = whisper.load_model('tiny')

#----------------------------------------------------------------------------------------------

## Upload audio file with streamlit

audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'm4a'])
st.audio(audio_file)

  #Saving the Audio file into the Temp_dir then we can use that audio file to transcribe
if audio_file is not None:
    upload_path = "data/"
    with open(os.path.join(upload_path, audio_file.name),"wb") as f:
        f.write(audio_file.getbuffer())
    st.success('AudioFile Saved')

if st.button('Transcribe Audio'):
    with st.spinner('whisperModel Loading'):
        time.sleep(3)
    st.success('WhisperModel Loaded')

    if audio_file is not None:
        options = whisper.DecodingOptions(language='en', fp16=False)
        transcription = model.transcribe(os.path.join("data", audio_file.name), **options.__dict__)
        final_transcript = transcription['text']
        st.markdown(f'<p style="padding: 10px; text-align: left; color:lightblack; background:#FFFACD ; font-size:15px; margin : 15px auto;">{final_transcript}</p>', unsafe_allow_html=True)
        st.success('Transcription Completed')
        st.download_button('Download', final_transcript, 'Transcript.txt')

    else:
        st.error('Please Upload An Audio File')
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------

# st.markdown("---")

# import streamlit as st
# import openai
# import time

# # Replace YOUR_API_KEY with your actual GPT-3 API key
# openai.api_key = "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"


# # Set the page title and add a banner image
# st.title("Sentiment Analysis")
# st.markdown("---")
# # Add a text input field and a button to submit the text
# text = st.text_area("Enter some text to analyze:")
# if st.button("Analyze"):

#     with st.spinner('Analyzing Sentiment.....'):
#         time.sleep(3)    
#     # Use GPT-3 to analyze the sentiment of the text
#     model_engine = "text-davinci-002"
#     prompt = (
#         f"Analyze the sentiment of the following text:\n{text}\n"
#         "Indicate whether the sentiment is positive, neutral, or negative."
#     )
#     completions = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1,stop=None,temperature=0.5)
#     sentiment_text = completions.choices[0].text

#     # Parse the sentiment from the response text
#     if "positive" in sentiment_text.lower():
#         sentiment = "positive 👍"
        
#     elif "neutral" in sentiment_text.lower():
#         sentiment = "neutral"
        
#     else:
#         sentiment = "negative 👎"

#     # Display the results in a visually appealing way
    
#     st.markdown(f'<p style="text-align: left; color: black; background: lightyellow; font-size: 20px; margin : 15px auto;">The sentiment of the text is: {sentiment}</p>', unsafe_allow_html=True)



