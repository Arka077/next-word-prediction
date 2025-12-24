import streamlit as st
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import urllib.request

# ============================================
# PAGE CONFIG
# ============================================
st. set_page_config(
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
    Load Keras model and vocabulary from GitHub repository. 
    """
    import pickle

    # GitHub raw content URLs for model and vocab files
    REPO_BASE_URL = "https://github.com/Arka077/next-word-prediction/raw/main/"
    MODEL_URL = REPO_BASE_URL + "next_word_model.keras"
    WORD_TO_IDX_URL = REPO_BASE_URL + "word_to_idx.pkl"
    IDX_TO_WORD_URL = REPO_BASE_URL + "idx_to_word.pkl"

    def download_if_missing(filename, url):
        if not os.path.exists(filename):
            try:
                st.info(f"Downloading {filename} from GitHub repository...")
                urllib.request.urlretrieve(url, filename)
                st.success(f"✅ {filename} downloaded successfully!")
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
        st. warning(f"⚠️  Vocabulary files not found or failed to load: {e}")
        word_to_idx = {"PAD": 0, "UNK": 1}
        idx_to_word = {0: "PAD", 1: "UNK"}

    seq_length = 30
    unk_id = word_to_idx. get("UNK", 1)
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
    if not current_text. strip():
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
    suggestions = [(idx_to_word. get(idx, "UNK"), preds[idx]) for idx in top_indices]
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

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if 'preview_text' not in st.session_state:
    st.session_state.preview_text = ""

if 'show_suggestions' not in st.session_state:
    st.session_state.show_suggestions = False

# ============================================
# CUSTOM CSS
# ============================================
st. markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Georgia', serif;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .suggestion-box {
        background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .suggestion-item {
        background:  rgba(255,255,255,0.2);
        border-radius: 5px;
        padding: 10px 15px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    . suggestion-item:hover {
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
    
    .info-banner {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        border-radius:  5px;
        margin:  10px 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN UI
# ============================================
st.title("📝 AI-Powered Notepad")
st.markdown("*Sherlock Holmes Edition - Type your text and press **Ctrl+Enter** to predict*")

# Info banner
st.markdown("""
<div class="info-banner">
    <strong>💡 How to use:</strong> Type your text in the left panel, then press <strong>Ctrl+Enter</strong> (or Cmd+Enter on Mac) to generate next word predictions! 
</div>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st. header("⚙️ Settings")
    
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
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All"):
        st.session_state.text_content = ""
        st.session_state.suggestions = []
        st.session_state.preview_text = ""
        st.session_state.show_suggestions = False
        st.rerun()
    
    if st.button("💾 Export Text"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download",
            data=st.session_state.text_content,
            file_name=f"notepad_{timestamp}.txt",
            mime="text/plain"
        )
    
    st.markdown("---")
    
    if not model_loaded:
        st. warning("⚠️ Model not loaded")
        st.info("Suggestions will be placeholder text.")
    else:
        st. success("✅ Model loaded successfully")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Your Text")
    
    # Use a form to capture Ctrl+Enter
    with st. form(key='text_form', clear_on_submit=False):
        text_input = st.text_area(
            label="Write here...",
            value=st.session_state.text_content,
            height=400,
            key="text_area",
            label_visibility="collapsed",
            placeholder="Start typing your Sherlock Holmes story...  (Press Ctrl+Enter to predict next words)",
        )
        
        # This button is triggered by Ctrl+Enter
        predict_button = st.form_submit_button("🔮 Predict Next Words (Ctrl+Enter)", use_container_width=True)
    
    # Handle form submission
    if predict_button: 
        st.session_state.text_content = text_input
        st. session_state.show_suggestions = True
    
    # Word count
    word_count = len(st.session_state.text_content.split()) if st.session_state.text_content else 0
    char_count = len(st.session_state.text_content)
    
    st.markdown(f"""
    <div class="stats-box">
        <strong>📊 Stats:</strong> {word_count} words | {char_count} characters
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("💡 AI Suggestions")
    
    # Check if we should show suggestions
    if st.session_state.show_suggestions and st.session_state.text_content. strip():
        if model_loaded:
            # Generate real suggestions
            try:
                print("\n" + "="*60)
                print(f"🔮 User pressed Ctrl+Enter - Generating predictions")
                print("="*60)
                
                suggestions = suggest_next_words(
                    model,
                    st.session_state.text_content,
                    word_to_idx,
                    idx_to_word,
                    seq_length,
                    unk_id,
                    top_k=suggestion_count,
                    temperature=temperature
                )
                
                st.markdown("""
                <div class="suggestion-box">
                    <h4>🎯 Next Word Suggestions:</h4>
                """, unsafe_allow_html=True)
                
                for i, (word, prob) in enumerate(suggestions, 1):
                    if st.button(f"{i}. {word} ({prob*100:.0f}%)", key=f"sugg_{i}"):
                        print(f"\n👆 User selected suggestion:  '{word}'")
                        st.session_state.text_content += f" {word}"
                        st. session_state.show_suggestions = False
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Preview continuation
                preview = generate_continuation(
                    model,
                    st.session_state.text_content,
                    word_to_idx,
                    idx_to_word,
                    seq_length,
                    unk_id,
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
                if st.button("🚀 Generate Full Continuation (50 words)"):
                    print("\n🚀 User requested full continuation (50 words)")
                    continuation = generate_continuation(
                        model,
                        st.session_state.text_content,
                        word_to_idx,
                        idx_to_word,
                        seq_length,
                        unk_id,
                        num_words=50,
                        temperature=temperature
                    )
                    print(f"✅ Full continuation inserted into text\n")
                    st.session_state.text_content += " " + continuation
                    st. session_state.show_suggestions = False
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
            
            for i, (word, prob) in enumerate(placeholder_suggestions[: suggestion_count], 1):
                if st.button(f"{i}. {word} ({prob*100:.0f}%)", key=f"sugg_{i}"):
                    st.session_state.text_content += f" {word}"
                    st.session_state.show_suggestions = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="preview-box">
                <strong>✨ Preview: </strong><br>
                (Placeholder) the detective carefully examined the evidence... 
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("⏳ Type your text in the left panel and press **Ctrl+Enter** to get predictions!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color:  #666;'>
    <small>💡 Tip: Press <strong>Ctrl+Enter</strong> (Cmd+Enter on Mac) in the text area to generate predictions | 
    Click suggestions to insert | Adjust creativity in sidebar</small>
</div>
""", unsafe_allow_html=True)
