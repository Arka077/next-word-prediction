# 📝 Next Word Predictor

A neural network-powered next word prediction application inspired by Sherlock Holmes. This project features an interactive Streamlit web interface and includes tools for training custom language models.

## 🚀 Live Demo

**Try it now:** [AI Notepad - Sherlock Holmes](https://ai-notepad-sherlock-holmes.streamlit.app/)

## ✨ Features

- **Real-time Prediction**: Get instant next word suggestions as you type
- **Neural Network Model**: Powered by a trained deep learning model
- **Interactive UI**: Clean, user-friendly Streamlit interface
- **Custom Training**: Train the model on your own text data
- **Sherlock Holmes Theme**: Model trained on classic literature

## 📁 Project Structure

```
next-word-predictor/
│
├── app.py                    # Streamlit web application
├── train_model.ipynb         # Model training notebook
├── next_word_model.keras     # Saved Keras model
├── model/                    # Model artifacts directory
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## 🛠️ Local Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd next-word-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 💡 Usage

### Web Application

1. Open the app in your browser (locally or via the live demo)
2. Enter a text prompt in the input field
3. View real-time next word predictions
4. Continue typing to get contextual suggestions

### Model Training

To train the model with your own data:

1. Open `train_model.ipynb` in Jupyter Notebook or JupyterLab
2. Replace the training data with your custom text corpus
3. Run all cells to train the model
4. The trained model will be saved as `next_word_model.keras`

## 🧠 Model Architecture

The model uses a neural network architecture designed for sequential text prediction, leveraging:
- Embedding layers for word representation
- LSTM/GRU layers for capturing temporal dependencies
- Dense layers for prediction output

## 📦 Dependencies

Key libraries used in this project:
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

See `requirements.txt` for the complete list.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Training data inspired by Arthur Conan Doyle's Sherlock Holmes stories
- Built with Streamlit and TensorFlow

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ and AI
