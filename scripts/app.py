"""
app.py — SpectroGen Streamlit GUI
Upload an audio file → generate album art via DDIM inference
Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder   import AudioEncoder
from unet      import UNet
from diffusion import DDPM

import gdown
import os

def download_checkpoint():
    ckpt_path = "checkpoints/spectrogen_final_best.pt.pt"
    if not os.path.exists(ckpt_path):
        os.makedirs("checkpoints", exist_ok=True)
        # Replace with your Google Drive file ID
        gdown.download(
            "https://drive.google.com/uc?id=YOUR_FILE_ID",
            ckpt_path, quiet=False
        )
    return ckpt_path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SpectroGen",
    page_icon   = "🎵",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎵 SpectroGen</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Music Album Art Generation via Cross-Modal Spectrogram Diffusion</p>',
            unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")

    ckpt_path = st.text_input(
        "Checkpoint Path",
        value="checkpoints/spectrogen_baseline_best.pt",
        help="Path to trained model checkpoint"
    )

    ddim_steps = st.slider(
        "DDIM Sampling Steps",
        min_value=20, max_value=200, value=50, step=10,
        help="More steps = better quality but slower"
    )

    n_mels = st.select_slider(
        "Mel Bins",
        options=[64, 128],
        value=64
    )

    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    **SpectroGen** generates album cover art from audio using a Denoising Diffusion Probabilistic Model (DDPM).

    **Architecture:**
    - 🎵 CNN Audio Encoder (mel → 256-dim)
    - 🏗️ U-Net with cross-attention
    - 🔊 DDIM fast sampler

    **Dataset:** FMA Small + MusicBrainz Cover Art Archive
    """)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt_path):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = AudioEncoder(embed_dim=256).to(device)
    unet    = UNet(in_channels=3, base_channels=64, audio_dim=256).to(device)
    ddpm    = DDPM(unet, encoder, T=500, schedule='cosine', device=device).to(device)

    if not os.path.exists(ckpt_path):
        return None, None, f"Checkpoint not found: {ckpt_path}"

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        ddpm.load_state_dict(ckpt['model'])
        ddpm.eval()
        epoch = ckpt.get('epoch', 0) + 1
        return ddpm, device, f"✅ Model loaded (epoch {epoch})"
    except Exception as e:
        return None, None, f"❌ Error loading model: {e}"


# ── Extract mel-spectrogram ───────────────────────────────────────────────────
def extract_melspec(audio_path, n_mels=64, sr=22050, duration=30):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    target = sr * duration
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y       = y[:target]
    mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32), y, sr


# ── Plot spectrogram ──────────────────────────────────────────────────────────
def plot_spectrogram(mel, sr=22050, hop_length=512):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    img = librosa.display.specshow(
        mel, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        ax=ax, cmap='magma'
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-Spectrogram', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    return fig


# ── Generate album art ────────────────────────────────────────────────────────
def generate_art(ddpm, mel_array, device, steps=50):
    mel_tensor = torch.tensor(mel_array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        sample = ddpm.sample_ddim(mel_tensor, steps=steps)
        sample = sample[0].cpu()
        # Proper normalization
        sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
        sample = (sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(sample)


# ── Main UI ───────────────────────────────────────────────────────────────────

# Load model status
with st.spinner("Loading model..."):
    model, device, status = load_model(ckpt_path)

if model is None:
    st.error(status)
    st.info("Make sure your checkpoint path is correct in the sidebar.")
    st.stop()
else:
    st.success(f"{status} | Device: **{device}**")

st.markdown("### 🎵 Upload Audio")
uploaded = st.file_uploader(
    "Upload an MP3 or WAV file (30 seconds will be used)",
    type=["mp3", "wav", "flac", "ogg"],
    help="Any audio format works — the model uses the first 30 seconds"
)

if uploaded:
    # Save to temp file
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    col1, col2, col3 = st.columns([1, 1, 1])

    # ── Extract spectrogram ───────────────────────────────────────────────────
    with st.spinner("Extracting mel-spectrogram..."):
        try:
            mel, y, sr = extract_melspec(tmp_path, n_mels=n_mels)
            spec_ok = True
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            spec_ok = False

    if spec_ok:
        with col1:
            st.markdown("#### 📊 Mel-Spectrogram")
            fig = plot_spectrogram(mel, sr=sr)
            st.pyplot(fig)
            plt.close()

            # Stats
            st.markdown(f"""
            **Audio Stats:**
            - Duration: `{len(y)/sr:.1f}s`
            - Sample rate: `{sr} Hz`
            - Mel bins: `{n_mels}`
            - Time frames: `{mel.shape[1]}`
            """)

        # ── Generate ──────────────────────────────────────────────────────────
        with col2:
            st.markdown("#### 🎨 Generated Album Art")
            generate_btn = st.button(f"🚀 Generate ({ddim_steps} DDIM steps)")

            if generate_btn:
                with st.spinner(f"Running DDIM inference ({ddim_steps} steps)..."):
                    try:
                        img = generate_art(model, mel, device, steps=ddim_steps)
                        img_large = img.resize((512, 512), Image.LANCZOS)

                        st.image(img_large, caption="Generated Album Art", use_column_width=True)

                        # Download button
                        buf = BytesIO()
                        img_large.save(buf, format="PNG")
                        st.download_button(
                            label     = "⬇️ Download Album Art",
                            data      = buf.getvalue(),
                            file_name = f"spectrogen_{uploaded.name.split('.')[0]}.png",
                            mime      = "image/png"
                        )

                        # Store in session
                        st.session_state['generated'] = img_large
                        st.session_state['mel']        = mel

                    except Exception as e:
                        st.error(f"Generation error: {e}")

        # ── Comparison ────────────────────────────────────────────────────────
        with col3:
            st.markdown("#### 🔍 Model Info")
            total_params = sum(p.numel() for p in model.parameters())
            st.markdown(f"""
            **Architecture:**
            - Model: DDPM U-Net
            - Parameters: `{total_params:,}`
            - Resolution: `128 × 128`
            - Timesteps T: `500`
            - Schedule: Cosine
            - Embedding dim: `256`

            **Inference:**
            - Sampler: DDIM
            - Steps: `{ddim_steps}`
            - Conditioning: Cross-attention

            **Dataset:**
            - Source: FMA Small + MusicBrainz
            - Pairs: `5,461`
            - Genres: 8
            """)

    # Cleanup
    os.unlink(tmp_path)

else:
    # Show example when no file uploaded
    st.info("👆 Upload an audio file to generate album art!")

    st.markdown("### 🏗️ Architecture Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🎵 Audio Encoder**
        ```
        Input: Mel-spectrogram
        (64 mel × 1292 frames)
              ↓
        5× Conv2D + BatchNorm
              ↓
        Global Average Pool
              ↓
        Output: 256-dim embedding
        ```
        """)

    with col2:
        st.markdown("""
        **🏗️ Diffusion U-Net**
        ```
        Input: Noisy image + t
        (3 × 128 × 128)
              ↓
        Encoder [64→128→256]
              ↓
        Bottleneck + Self-Attn
              ↓
        Decoder + Cross-Attn ←── audio
              ↓
        Output: Predicted noise
        ```
        """)

    with col3:
        st.markdown("""
        **🔊 DDIM Sampler**
        ```
        Start: Pure noise
        (3 × 128 × 128)
              ↓
        50 denoising steps
        (vs 500 for DDPM)
              ↓
        Audio conditioning
        at each step
              ↓
        Output: Album art
        ```
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align: center; color: #555; font-size: 0.85rem;'>
SpectroGen — Music Album Art Generation via Cross-Modal Spectrogram Diffusion<br>
Trained from scratch on FMA Small + MusicBrainz Cover Art Archive
</p>
""", unsafe_allow_html=True)
