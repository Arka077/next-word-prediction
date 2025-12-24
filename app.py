import streamlit as st
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os
import urllib.request

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Notepad - Sherlock Holmes",
    page_icon="📝",
    layout="wide"
)

# ============================================
# LOAD KERAS MODEL & VOCAB
# ============================================
@st.cache_resource
def load_model_and_vocab():
    """
    Load Keras model and vocabulary, downloading if not present (for Streamlit Cloud).
    """
    import pickle

    # URLs for model and vocab files (replace with your own if needed)
    MODEL_URL = "https://huggingface.co/datasets/Arka077/next-word-prediction/resolve/main/next_word_model.keras"
    WORD_TO_IDX_URL = "https://huggingface.co/datasets/Arka077/next-word-prediction/resolve/main/word_to_idx.pkl"
    IDX_TO_WORD_URL = "https://huggingface.co/datasets/Arka077/next-word-prediction/resolve/main/idx_to_word.pkl"

    def download_if_missing(filename, url):
        if not os.path.exists(filename):
            try:
                st.info(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                raise

    # Download model and vocab if not present
    download_if_missing('next_word_model.keras', MODEL_URL)
    download_if_missing('word_to_idx.pkl', WORD_TO_IDX_URL)
    download_if_missing('idx_to_word.pkl', IDX_TO_WORD_URL)

    # Load Keras model
    try:
        model = tf.keras.models.load_model('next_word_model.keras')
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        raise

    # Load vocab dicts
    try:
        with open('word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        with open('idx_to_word.pkl', 'rb') as f:
            idx_to_word = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️  Vocabulary files not found or failed to load: {e}")
        word_to_idx = {"PAD": 0, "UNK": 1}
        idx_to_word = {0: "PAD", 1: "UNK"}

    seq_length = 30
    unk_id = word_to_idx.get("UNK", 1)
    return model, word_to_idx, idx_to_word, seq_length, unk_id

# Load model and vocab (only logs once due to cache)
try:
    model, word_to_idx, idx_to_word, seq_length, unk_id = load_model_and_vocab()
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"⚠️ Model not loaded: {e}")
    model_loaded = False
    model = None
    word_to_idx = {"PAD": 0, "UNK": 1}
    idx_to_word = {0: "PAD", 1: "UNK"}
    seq_length = 30
    unk_id = 1

# ============================================
# HELPER FUNCTIONS
# ============================================
def suggest_next_words(model, current_text, word_to_idx, idx_to_word, 
                       seq_length, unk_id, top_k=5, temperature=0.7):
    """Get word suggestions"""
    if not current_text.strip():
        return []
    
    tokens = current_text.lower().split()
    seed = [word_to_idx.get(w, unk_id) for w in tokens]
    if len(seed) < seq_length:
        seed = [word_to_idx.get("PAD", 0)] * (seq_length - len(seed)) + seed
    else:
        seed = seed[-seq_length:]
    seed_array = np.array(seed, dtype=np.int32).reshape(1, -1)
    preds = model.predict(seed_array, verbose=0)[0]
    preds = np.log(preds + 1e-9) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    top_indices = np.argsort(preds)[-top_k:][::-1]
    suggestions = [(idx_to_word.get(idx, "UNK"), preds[idx]) for idx in top_indices]
    return suggestions

def generate_continuation(model, start_text, word_to_idx, idx_to_word,
                         seq_length, unk_id, num_words=30, temperature=0.7):
    """Generate text continuation"""
    if not start_text.strip():
        return ""
    
    tokens = start_text.lower().split()
    seed = [word_to_idx.get(w, unk_id) for w in tokens]
    if len(seed) < seq_length:
        seed = [word_to_idx.get("PAD", 0)] * (seq_length - len(seed)) + seed
    else:
        seed = seed[-seq_length:]
    generated = []
    for i in range(num_words):
        seed_array = np.array(seed, dtype=np.int32).reshape(1, -1)
        preds = model.predict(seed_array, verbose=0)[0]
        preds = np.log(preds + 1e-9) / temperature
        preds = np.exp(preds)
        preds = preds / np.sum(preds)
        next_id = np.random.choice(len(preds), p=preds)
        next_word = idx_to_word.get(next_id, "UNK")
        generated.append(next_word)
        seed = seed[1:] + [next_id]
    result = " ".join(generated)
    return result

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

if 'show_suggestions' not in st.session_state:
    st.session_state.show_suggestions = False

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if 'preview_text' not in st.session_state:
    st.session_state.preview_text = ""

if 'typing_active' not in st.session_state:
    st.session_state.typing_active = False

# Callback for text updates
def update_text_callback():
    st.session_state.last_update_time = time.time()
    st.session_state.typing_active = True

# ============================================
# CUSTOM CSS & JAVASCRIPT FOR AUTO-DETECTION
# ============================================
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Georgia', serif;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .suggestion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .suggestion-item {
        background: rgba(255,255,255,0.2);
        border-radius: 5px;
        padding: 10px 15px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .suggestion-item:hover {
        background: rgba(255,255,255,0.3);
        transform: translateX(5px);
    }
    
    .preview-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        font-style: italic;
        color: #555;
    }
    
    .stats-box {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>

<script>
// Monitor textarea for typing activity
let typingTimer;
const typingDelay = 2000; // 2 seconds

function setupTypingDetector() {
    const textarea = document.querySelector('.stTextArea textarea');
    if (textarea) {
        textarea.addEventListener('input', function() {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(function() {
                // Force Streamlit to detect the change
                const event = new Event('change', { bubbles: true });
                textarea.dispatchEvent(event);
            }, typingDelay);
        });
    }
}

// Run setup when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupTypingDetector);
} else {
    setupTypingDetector();
}

// Also run on Streamlit reruns
setTimeout(setupTypingDetector, 100);
</script>
""", unsafe_allow_html=True)

# ============================================
# MAIN UI
# ============================================
st.title("📝 AI-Powered Notepad")
st.markdown("*Sherlock Holmes Edition - Start typing and pause for suggestions*")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    temperature = st.slider(
        "Creativity Level",
        min_value=0.3,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower = more predictable, Higher = more creative"
    )
    
    suggestion_count = st.slider(
        "Number of Suggestions",
        min_value=3,
        max_value=10,
        value=5
    )
    
    preview_length = st.slider(
        "Preview Length (words)",
        min_value=10,
        max_value=50,
        value=20
    )
    
    auto_suggest_delay = st.slider(
        "Auto-suggest Delay (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All"):
        st.session_state.text_content = ""
        st.session_state.suggestions = []
        st.session_state.preview_text = ""
        st.rerun()
    
    if st.button("💾 Export Text"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download",
            data=st.session_state.text_content,
            file_name=f"notepad_{timestamp}.txt",
            mime="text/plain"
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Your Text")
    
    # Text area with on_change callback
    text_input = st.text_area(
        label="Write here...",
        value=st.session_state.text_content,
        height=400,
        key="text_area",
        label_visibility="collapsed",
        placeholder="Start typing your Sherlock Holmes story...",
        on_change=lambda: update_text_callback()
    )
    
    # Update session state
    if text_input != st.session_state.text_content:
        st.session_state.text_content = text_input
        st.session_state.last_update_time = time.time()
        # Force a rerun after delay to show suggestions
        time.sleep(0.1)
    
    # Word count
    word_count = len(text_input.split()) if text_input else 0
    char_count = len(text_input)
    
    st.markdown(f"""
    <div class="stats-box">
        <strong>📊 Stats:</strong> {word_count} words | {char_count} characters
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("💡 AI Suggestions")
    
    if not model_loaded:
        st.warning("⚠️ Model not loaded. Please ensure 'next_word_model.keras' and vocab files are in the same directory.")
        st.info("The app will still work but suggestions will be placeholder text.")
    
    # Check if we should show suggestions (after delay of no typing)
    time_since_update = time.time() - st.session_state.last_update_time
    
    if text_input.strip() and time_since_update >= auto_suggest_delay:
        if model_loaded:
            # Generate real suggestions
            try:
                print("\n" + "="*60)
                print(f"⏰ User stopped typing ({auto_suggest_delay}s pause detected)")
                print("="*60)
                
                suggestions = suggest_next_words(
                    model,
                    text_input, word_to_idx, idx_to_word, 
                    seq_length, unk_id, 
                    top_k=suggestion_count, 
                    temperature=temperature
                )
                
                st.markdown("""
                <div class="suggestion-box">
                    <h4>🎯 Next Word Suggestions:</h4>
                """, unsafe_allow_html=True)
                
                for i, (word, prob) in enumerate(suggestions, 1):
                    if st.button(f"{i}. {word} ({prob*100:.0f}%)", key=f"sugg_{i}"):
                        print(f"\n👆 User selected suggestion: '{word}'")
                        st.session_state.text_content += f" {word}"
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Preview continuation
                preview = generate_continuation(
                    model,
                    text_input, word_to_idx, idx_to_word,
                    seq_length, unk_id,
                    num_words=preview_length,
                    temperature=temperature
                )
                
                st.markdown(f"""
                <div class="preview-box">
                    <strong>✨ Preview:</strong><br>
                    {preview}
                </div>
                """, unsafe_allow_html=True)
                
                # Full continuation button
                if st.button("🚀 Generate Full Continuation"):
                    print("\n🚀 User requested full continuation (50 words)")
                    continuation = generate_continuation(
                        model,
                        text_input, word_to_idx, idx_to_word,
                        seq_length, unk_id,
                        num_words=50,
                        temperature=temperature
                    )
                    print(f"✅ Full continuation inserted into text\n")
                    st.session_state.text_content += " " + continuation
                    st.rerun()
                    
            except Exception as e:
                print(f"\n❌ ERROR: {e}\n")
                st.error(f"Error generating suggestions: {e}")
                
        else:
            # Placeholder suggestions when model not loaded
            st.markdown("""
            <div class="suggestion-box">
                <h4>🎯 Next Word Suggestions:</h4>
            """, unsafe_allow_html=True)
            
            placeholder_suggestions = [
                ("detective", 0.25),
                ("investigation", 0.20),
                ("mystery", 0.18),
                ("holmes", 0.15),
                ("evidence", 0.12)
            ]
            
            for i, (word, prob) in enumerate(placeholder_suggestions[:suggestion_count], 1):
                if st.button(f"{i}. {word} ({prob*100:.0f}%)", key=f"sugg_{i}"):
                    st.session_state.text_content += f" {word}"
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="preview-box">
                <strong>✨ Preview:</strong><br>
                (Placeholder) the detective carefully examined the evidence...
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info(f"⏳ Keep typing or pause for {auto_suggest_delay:.1f}s for suggestions...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>💡 Tip: Stop typing for suggestions to appear automatically | Click suggestions to insert | 
    Adjust creativity in sidebar</small>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism - check every 0.5 seconds
if text_input.strip():
    time_since_update = time.time() - st.session_state.last_update_time
    if time_since_update < auto_suggest_delay:
        # Still typing or just stopped, wait a bit and check again
        time.sleep(0.5)
        st.rerun()