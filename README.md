# Next Word Predictor

This project is a next word prediction app using a trained neural network model. It includes a Streamlit web interface and a Jupyter notebook for model training.

## Project Structure
- `app.py`: Streamlit app for next word prediction
- `train_model.ipynb`: Jupyter notebook for training the model
- `next_word_model.keras`, `model/`: Saved model files
- `requirements.txt`: Python dependencies
- `.gitignore`: Files and folders to ignore in Git

## Setup
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the Streamlit app in your browser and enter a text prompt to get next word predictions.

## Training
- Use `train_model.ipynb` to retrain or fine-tune the model with your own data.

## License
MIT License
