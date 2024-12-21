## Waste Sorting Using Image Processing

---

## Project Description
This project uses image processing and deep learning techniques to classify waste into predefined categories: **Green Leaves**, **Bottle**, **Clothes**, and **Vegetables**. The aim is to automate the waste sorting process to facilitate better recycling and disposal practices.

---

## Features
1. **Image Classification**: 
   - Classifies images of waste into categories using a Convolutional Neural Network (CNN) model.
   - Ensures accurate sorting with a confidence threshold.

2. **Real-Time Detection**:
   - Integrates with a live video feed for real-time waste classification.
   - Captures and processes frames using OpenCV and TensorFlow.

3. **Model Persistence**:
   - Saves the trained model for future use.
   - Supports model loading for predictions without retraining.

---

## Technology Stack
- **Programming Language**: Python
- **Libraries**:
  - TensorFlow/Keras: For building and training the CNN model.
  - NumPy: For numerical computations.
  - OpenCV: For video feed and frame capturing.
  - PIL: For image preprocessing.

---

## Dataset
The dataset consists of images categorized into:
- Green Leaves
- Bottles
- Clothes
- Vegetables

Each image is resized to `224x224` pixels and normalized for training.

---

## Model Architecture
- **Input Layer**: Accepts `224x224x3` RGB images.
- **Conv2D Layers**: Feature extraction with 3 layers of increasing filters (32, 64, 128).
- **MaxPooling2D Layers**: Reduces dimensionality while retaining important features.
- **Flatten Layer**: Converts 2D feature maps to 1D vectors.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Softmax activation with 4 neurons (one for each category).

---

## Installation Instructions
1. Clone the repository or download the project files.
2. Install the required libraries:
   ```bash
   pip install tensorflow opencv-python numpy pillow
   ```
3. Place the dataset in the specified directory structure:
   ```
   dataset/
   ├── Greenleaves/
   ├── bottle/
   ├── clothes/
   ├── vegetables/
   ```
4. Run the script to train the model:
   ```bash
   python main.py
   ```

---

## Usage
### Training the Model
1. The script will train a CNN model on the provided dataset.
2. The trained model is saved as `image_classifier_model.h5`.

### Real-Time Classification
1. Connect a mobile camera or webcam using the specified URL (replace `http://192.168.29.190:4747/video` with your camera feed URL).
2. Run the script:
   ```bash
   python main.py
   ```
3. Press **SPACE** to capture a frame for classification.
4. View the prediction in the console.

---

## Sample Output
- **Prediction**:
  ```plaintext
  The image is classified as: Bottle
  ```

- **Test Metrics**:
  ```plaintext
  Test loss: 0.1
  Test accuracy: 98.5%
  ```

---

## Limitations
- Accuracy depends on the quality and diversity of the dataset.
- Requires a stable video feed for real-time classification.

---

## Future Enhancements
- Expand the dataset with more categories.
- Implement hardware integration for sorting mechanisms.
- Improve accuracy with advanced deep learning architectures like ResNet or MobileNet.

---
