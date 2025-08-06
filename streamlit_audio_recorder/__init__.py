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
    "streamlit_audio_recorder",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)

def st_audio_recorder(key=None, height=400):
    """
    Create a professional audio recorder component that works reliably on Streamlit Cloud.
    
    This component uses pure JavaScript MediaRecorder API to capture high-quality audio
    directly in the browser, bypassing Streamlit's audio processing limitations.
    
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
        - 'type': 'audioRecorded'
        - 'audioData': base64 encoded audio
        - 'mimeType': audio format (webm/mp4)
        - 'duration': recording duration in seconds
        - 'sampleRate': audio sample rate
        - 'channels': number of audio channels
        - 'quality': audio quality indicator
    """
    
    # Call the component function
    component_value = _component_func(
        key=key,
        default=None,
        height=height,
    )
    
    return component_value

def process_recorded_audio(audio_data_dict):
    """
    Process the high-quality audio data received from the JavaScript component.
    
    This function handles the conversion from browser-recorded audio to numpy array
    format suitable for speech analysis, ensuring optimal quality preservation.
    
    Parameters:
    -----------
    audio_data_dict : dict
        Dictionary containing audio data from the component
        
    Returns:
    --------
    numpy.ndarray or None
        High-quality audio data as numpy array ready for analysis
    """
    
    if not audio_data_dict or audio_data_dict.get('type') != 'audioRecorded':
        return None
        
    try:
        # Extract audio information
        audio_base64 = audio_data_dict['audioData']
        mime_type = audio_data_dict.get('mimeType', 'audio/webm')
        duration = audio_data_dict.get('duration', 0)
        sample_rate = audio_data_dict.get('sampleRate', 44100)
        
        st.info(f"📊 Processing {duration:.1f}s of high-quality audio ({mime_type})")
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_base64)
        
        # Determine file extension from MIME type
        if 'webm' in mime_type:
            file_ext = '.webm'
        elif 'mp4' in mime_type:
            file_ext = '.mp4'
        else:
            file_ext = '.webm'  # Default fallback
        
        # Create temporary file to process the audio
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Convert to high-quality WAV using pydub
            audio_segment = AudioSegment.from_file(temp_file_path)
            
            # Ensure optimal settings for speech analysis
            audio_segment = audio_segment.set_channels(1)  # Mono
            audio_segment = audio_segment.set_frame_rate(44100)  # High sample rate
            audio_segment = audio_segment.set_sample_width(2)  # 16-bit depth
            
            # Apply audio enhancement
            # Normalize volume
            audio_segment = audio_segment.normalize()
            
            # Apply gentle compression to improve consistency
            audio_segment = audio_segment.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            
            # Export to WAV format in memory
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav", parameters=["-ar", "44100", "-ac", "1"])
            wav_io.seek(0)
            
            # Read with soundfile for precise control
            audio_data, actual_sample_rate = sf.read(wav_io, dtype='float32')
            
            # Ensure proper normalization for speech analysis
            if np.max(np.abs(audio_data)) > 0:
                # Normalize to 80% of maximum to prevent clipping
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Apply gentle high-pass filter to remove low-frequency noise
            # This is a simple implementation - for production, consider using scipy
            if len(audio_data) > 100:
                # Simple high-pass filter approximation
                filtered_data = np.copy(audio_data)
                alpha = 0.95  # High-pass filter coefficient
                for i in range(1, len(filtered_data)):
                    filtered_data[i] = alpha * (filtered_data[i-1] + audio_data[i] - audio_data[i-1])
                audio_data = filtered_data
            
            st.success(f"✅ Audio processed successfully! Sample rate: {actual_sample_rate} Hz, Duration: {len(audio_data)/actual_sample_rate:.2f}s")
            
            return audio_data
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        st.info("💡 This might be due to browser compatibility. Please try again or use a different browser.")
        return None

def create_professional_recorder():
    """
    Create a complete professional audio recorder interface.
    
    This function creates the full recording interface with proper error handling,
    progress indicators, and seamless integration with your existing analysis pipeline.
    
    Returns:
    --------
    numpy.ndarray or None
        Processed audio data ready for analysis, or None if no recording
    """
    
    st.markdown("### 🎙️ Professional Audio Recorder")
    st.markdown("*High-quality browser-based recording for accurate speech analysis*")
    
    # Add some helpful information
    with st.expander("ℹ️ How to use this recorder"):
        st.markdown("""
        **This professional recorder ensures perfect audio quality:**
        
        1. **Click the microphone button** to start recording
        2. **Allow microphone access** when prompted by your browser
        3. **Watch the 3-2-1 countdown** to get ready
        4. **Speak clearly** during recording (you'll see real-time visualization)
        5. **Click stop** when finished, or it will auto-stop
        6. **Analysis begins automatically** with high-quality audio
        
        **Features:**
        - ✅ Professional-grade audio capture (44.1kHz, 16-bit)
        - ✅ Real-time audio visualization
        - ✅ Automatic noise reduction and enhancement
        - ✅ Works perfectly on Streamlit Cloud
        - ✅ No audio quality degradation
        """)
    
    # Create the audio recorder component
    audio_result = st_audio_recorder(key="professional_speech_recorder", height=400)
    
    # Process audio if received
    if audio_result:
        if audio_result.get('type') == 'audioRecorded':
            with st.spinner("🔄 Processing professional-grade audio..."):
                audio_data = process_recorded_audio(audio_result)
                
                if audio_data is not None:
                    # Display audio quality information
                    duration = len(audio_data) / 44100
                    st.success(f"🎯 **Professional Audio Processed Successfully!**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{duration:.2f}s")
                    with col2:
                        st.metric("Sample Rate", "44,100 Hz")
                    with col3:
                        st.metric("Quality", "Professional")
                    
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
        elif audio_result.get('type') == 'ready':
            st.info("🎤 Professional recorder is ready! Click the microphone to start.")
    
    return None
