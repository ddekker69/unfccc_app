# utils/azure_blob_utils.py

import os
import streamlit as st
import zipfile
from azure.storage.blob import BlobServiceClient
import logging

logger = logging.getLogger(__name__)

# --- Load connection string from config module ---
try:
    from config import AZURE_CONNECTION_STRING, AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY
    logger.info("Loaded Azure configuration from config.py")
except ImportError:
    # Fallback to direct environment loading if config module not available
    AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    logger.info("Loaded Azure configuration from environment variables")

# Build connection string from components if not provided directly
if not AZURE_CONNECTION_STRING and AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
    AZURE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT_NAME};AccountKey={AZURE_STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    logger.info("Built Azure connection string from account credentials")

# Make Azure connection optional - don't raise error for local usage
if AZURE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        logger.info("✅ Azure Blob Service Client initialized successfully")
    except Exception as e:
        blob_service_client = None
        logger.error(f"❌ Failed to initialize Azure Blob Service Client: {e}")
else:
    blob_service_client = None
    logger.warning("⚠️ Azure connection string not found - running in local mode")

def download_blob(container_name, blob_name, download_path):
    """
    Download a blob from Azure Storage to a local file.
    
    Args:
        container_name (str): Name of the Azure container
        blob_name (str): Name of the blob to download
        download_path (str): Local path where the file should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if not blob_service_client:
        logger.error(f"❌ Azure not configured - cannot download blob: {blob_name}")
        return False
    
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        with open(download_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        logger.info(f"✅ Downloaded: {blob_name} → {download_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download blob: {blob_name} → {e}")
        return False

def upload_blob(container_name, blob_name, local_file_path):
    """
    Upload a local file to Azure Storage as a blob.
    
    Args:
        container_name (str): Name of the Azure container
        blob_name (str): Name for the blob in Azure
        local_file_path (str): Path to the local file to upload
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    if not blob_service_client:
        logger.error(f"❌ Azure not configured - cannot upload blob: {blob_name}")
        return False
        
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(local_file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
        logger.info(f"✅ Uploaded: {local_file_path} → {blob_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload blob: {local_file_path} → {e}")
        return False

def download_and_extract_model_from_azure(container_name, blob_name, extract_to="models/"):
    """
    Download a model zip file from Azure and extract it locally.
    
    Args:
        container_name (str): Name of the Azure container
        blob_name (str): Name of the model zip blob
        extract_to (str): Local directory to extract the model to
        
    Returns:
        bool: True if download and extraction successful, False otherwise
    """
    if not blob_service_client:
        logger.error(f"❌ Azure not configured - cannot download model: {blob_name}")
        return False
        
    os.makedirs(extract_to, exist_ok=True)
    zip_filename = os.path.basename(blob_name)
    zip_path = os.path.join(extract_to, zip_filename)

    try:
        logger.info("📦 Downloading model zip from Azure Blob...")
        success = download_blob(container_name, blob_name, zip_path)
        if not success:
            raise RuntimeError(f"Download failed for blob: {blob_name}")

        logger.info("📂 Extracting model zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # Clean up zip file
        os.remove(zip_path)
        logger.info(f"✅ Model extracted to: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Error during download/extract: {e}")
        # Clean up partial zip file if it exists
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass
        return False

def list_blobs(container_name, prefix=""):
    """
    List all blobs in a container with an optional prefix filter.
    
    Args:
        container_name (str): Name of the Azure container
        prefix (str): Optional prefix to filter blobs
        
    Returns:
        list: List of blob names, or empty list if failed
    """
    if not blob_service_client:
        logger.error("❌ Azure not configured - cannot list blobs")
        return []
        
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        blob_names = [blob.name for blob in blobs]
        logger.info(f"✅ Listed {len(blob_names)} blobs in container {container_name}")
        return blob_names
    except Exception as e:
        logger.error(f"❌ Failed to list blobs: {e}")
        return []

def upload_file(container_name, blob_name, local_path):
    """
    Alias for upload_blob for backwards compatibility.
    """
    return upload_blob(container_name, blob_name, local_path)

def check_azure_connection():
    """
    Check if Azure connection is properly configured and working.
    
    Returns:
        dict: Status information about Azure connection
    """
    status = {
        'configured': False,
        'connection_string': bool(AZURE_CONNECTION_STRING),
        'account_name': bool(AZURE_STORAGE_ACCOUNT_NAME),
        'account_key': bool(AZURE_STORAGE_ACCOUNT_KEY),
        'client_initialized': bool(blob_service_client),
        'test_connection': False
    }
    
    if blob_service_client:
        status['configured'] = True
        # Try a simple operation to test the connection
        try:
            # List containers (lightweight operation)
            list(blob_service_client.list_containers(max_results=1))
            status['test_connection'] = True
            logger.info("✅ Azure connection test successful")
        except Exception as e:
            status['test_connection'] = False
            logger.warning(f"⚠️ Azure connection test failed: {e}")
    
    return status
