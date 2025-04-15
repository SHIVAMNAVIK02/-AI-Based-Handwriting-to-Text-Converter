## Handwritten Text Recognition with Text Improvement


## 📝 Overview
This Streamlit application performs Optical Character Recognition (OCR) on handwritten text images, with optional text improvement features. It supports both user-uploaded images and samples from the IAM Handwriting Database.

## ✨ Features
Handwritten Text Recognition: Uses EasyOCR to extract text from images

Text Improvement: Offers spelling correction and basic grammar suggestions

Image Preprocessing: Includes adaptive thresholding and denoising for better OCR results

IAM Dataset Integration: Access to pre-loaded handwriting samples for testing

Visual Comparison: Shows original vs. processed images

Accuracy Metrics: Compares results with ground truth when available

## 🛠️ Installation
Clone this repository

Create and activate a virtual environment (recommended)

Install dependencies:

bash
Copy
pip install -r requirements.txt


## 🚀 Usage
Run the application with:

bash
Copy
streamlit run app4.py
The app will open in your default browser at http://localhost:8501

Interface Options:
Upload Mode: Upload your own handwritten text images (JPG, PNG, JPEG)

Sample Mode: Select from pre-loaded IAM dataset samples

Settings (Sidebar):

Toggle image preprocessing visualization

Enable/disable text improvement

Switch between uploaded images and IAM samples

## 📊 Output
The application provides:

Improved text output (with spelling corrections)

List of suggested improvements

Raw OCR output for comparison

Image preprocessing visualization

Accuracy metrics when ground truth is available

## 📚 Dependencies
Streamlit (frontend)

EasyOCR (OCR engine)

OpenCV (image processing)

TextBlob (text correction)

Hugging Face Datasets (IAM dataset access)

Matplotlib (visualization)

## ⚖️ License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

Note: For best results, use clear images of handwritten text with good contrast between the text and background.