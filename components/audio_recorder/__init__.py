import streamlit as st
import streamlit.components.v1 as components
import base64
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os

# Create a _component_func which will call the frontend component
_component_func = components.declare_component(
    "audio_recorder",
    path=os.path.join(os.path.dirname(__file__)),
)

def audio_recorder(key=None, height=300):
    """
    Create an audio recorder component.
    
    Parameters:
    -----------
    key : str or None
        An optional key that uniquely identifies this component.
    height : int
        The height of the component in pixels.
        
    Returns:
    --------
    dict or None
        A dictionary containing audio data if recording is complete, None otherwise.
        Dictionary contains:
        - 'audioData': base64 encoded audio
        - 'mimeType': audio format
        - 'duration': recording duration in seconds
    """
    
    # Call the component function
    component_value = _component_func(
        key=key,
        default=None,
        height=height,
    )
    
    return component_value

def process_audio_data(audio_data_dict):
    """
    Process the audio data received from the JavaScript component.
    
    Parameters:
    -----------
    audio_data_dict : dict
        Dictionary containing audio data from the component
        
    Returns:
    --------
    numpy.ndarray
        Audio data as numpy array ready for analysis
    """
    
    if not audio_data_dict:
        return None
        
    try:
        # Decode base64 audio data
        audio_base64 = audio_data_dict['audioData']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create temporary file to process the audio
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Convert webm to wav using pydub
            audio_segment = AudioSegment.from_file(temp_file_path, format="webm")
            
            # Convert to mono and set sample rate to 44100 Hz
            audio_segment = audio_segment.set_channels(1).set_frame_rate(44100)
            
            # Export to wav format in memory
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Read with soundfile
            audio_data, sample_rate = sf.read(wav_io)
            
            # Ensure float32 format
            audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def create_audio_recorder_section():
    """
    Create a complete audio recorder section with processing.
    
    Returns:
    --------
    numpy.ndarray or None
        Processed audio data ready for analysis, or None if no recording
    """
    
    st.markdown("### 🎙️ Professional Audio Recorder")
    st.markdown("*High-quality browser-based recording for accurate speech analysis*")
    
    # Create the audio recorder component
    audio_result = audio_recorder(key="speech_recorder", height=300)
    
    # Process audio if received
    if audio_result:
        with st.spinner("🔄 Processing high-quality audio..."):
            audio_data = process_audio_data(audio_result)
            
            if audio_data is not None:
                # Display audio info
                duration = len(audio_data) / 44100  # Assuming 44100 Hz sample rate
                st.success(f"✅ Audio processed successfully!")
                st.info(f"📊 Duration: {duration:.2f} seconds | Sample Rate: 44,100 Hz | Quality: Professional")
                
                # Optional: Display audio player for verification
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    sf.write(temp_wav.name, audio_data, 44100)
                    with open(temp_wav.name, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                    os.unlink(temp_wav.name)
                
                return audio_data
            else:
                st.error("❌ Failed to process audio. Please try recording again.")
                return None
    
    return None
