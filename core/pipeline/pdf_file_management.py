from config import PDF_DIR, CSV_PATH, OUTPUT_PATH, UPLOAD_PDF_DIR, CHECKPOINTS_DIR, OUTPUT_PATH_LEGACY
import config
import os
import fitz
import streamlit as st
import pickle

def save_pdf(file, filename):
    """Save uploaded PDF to the designated upload directory."""
    # Check if UPLOAD_PDF_DIR exists, if not create it
    if not os.path.exists(UPLOAD_PDF_DIR):
        os.makedirs(UPLOAD_PDF_DIR)
    filepath = os.path.join(UPLOAD_PDF_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(file.getbuffer())  # Save the uploaded file
    return filepath


def process_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        # Extract text from the PDF
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text() for page in doc])
        return full_text
    except Exception as e:
        st.error(f"❌ Error processing the PDF: {e}")
        return None


def update_document_info(pdf_file):
    """Add a new uploaded PDF to the extracted_texts.pkl file."""
    # Save the uploaded PDF
    filename = pdf_file.name
    file_path = save_pdf(pdf_file, filename)

    # Process the PDF (extract text)
    full_text = process_pdf(file_path)

    if full_text is None:
        return

    # Load existing data
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'rb') as f:
            link_df = pickle.load(f)
    else:
        link_df = pd.DataFrame(columns=['document_id', 'document_name', 'download_link', 'title', 'text', 'status'])

    # Check if the filename already exists in the DataFrame
    if filename in link_df['document_name'].values:
        # If it exists, show a warning and skip the addition
        st.warning(f"⚠️ The file '{filename}' already exists in the database. It will not be added again.")
        return

    # Create a new DataFrame for the new document info
    new_row = pd.DataFrame({
        'document_id': [f"doc_{len(link_df)}"],
        'document_name': [filename],
        'download_link': [file_path],
        'title': [''],  # Extract title if needed
        'text': [full_text],
        'status': ['ok']
    })

    # Use pd.concat to append the new row
    link_df = pd.concat([link_df, new_row], ignore_index=True)

    # Save the updated dataframe
    Path("data").mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(link_df, f)

    st.success(f"File uploaded and processed: {filename}")


def save_as_new_file():
    """Save current extracted_texts.pkl as a new checkpoint."""
    # Load the current DataFrame
    df = pd.read_pickle(OUTPUT_PATH)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)  # Ensure directory exists

    # Get the next available checkpoint number
    existing_checkpoints = [f.stem for f in CHECKPOINTS_DIR.glob("checkpoint_*.pkl")]
    checkpoint_numbers = [int(f.split("_")[1]) for f in existing_checkpoints if f.startswith("checkpoint_")]
    next_checkpoint_number = max(checkpoint_numbers, default=0) + 1

    # Define the new file path
    new_file_path = CHECKPOINTS_DIR / f"checkpoint_{next_checkpoint_number}.pkl"

    # Save the current DataFrame to the new file
    df.to_pickle(new_file_path)
    st.success(f"File saved as {new_file_path.name}")

    return next_checkpoint_number


def load_and_overwrite(file_name):
    """Overwrite extracted_texts.pkl with a selected checkpoint or legacy file."""
    # Check if the file is in the checkpoints folder or in the data folder
    if file_name == "extracted_texts_legacy":
        file_path = OUTPUT_PATH_LEGACY
    else:
        file_path = CHECKPOINTS_DIR / f"{file_name}.pkl"

    # Load the selected file
    df = pd.read_pickle(file_path)

    # Save it back to extracted_texts.pkl (overwrite)
    df.to_pickle(OUTPUT_PATH)
    st.success(f"extracted_texts.pkl has been overwritten with {file_name}.pkl")