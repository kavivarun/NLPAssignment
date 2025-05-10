import streamlit as st

st.set_page_config(page_title="SpeakSmart", page_icon="🧠", layout="wide")

# Create tabs for navigation instead of sidebar
tab1, tab2, tab3 = st.tabs(["Upload Audio", "Record Audio", "Direct Input"])

with tab1:
    st.markdown("""<h1 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #e0d6e9;
                text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">SpeakSmart</h1>""",unsafe_allow_html=True)

    from upload import UploadRecord
    upr = UploadRecord()
    upr.transcribe()
    
with tab2:
    st.markdown("""<h1 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #e0d6e9;
                text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">SpeakSmart</h1>""",unsafe_allow_html=True)

    from record import AudioRecorder
    rec = AudioRecorder()
    rec.record_audio()

with tab3:
    st.markdown("""<h1 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #e0d6e9;
                text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">SpeakSmart</h1>""",
                unsafe_allow_html=True)

    from direct_input import DirectText
    DirectText().enter_text()
