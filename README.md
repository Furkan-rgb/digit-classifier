# CNN Digit Classifier

A web-based MNIST digit classifier using a Convolutional Neural Network (CNN) built with TensorFlow and deployed with Vue.js + Vite.

## Features

- Real-time digit classification using TensorFlow.js
- Interactive drawing canvas for digit input
- Visual representation of CNN architecture
- Live probability visualization for all digits (0-9)
- Responsive design with modern UI
- Client-side inference (no server required)

## Tech Stack

- **Frontend**: Vue 3 + Vite
- **ML Model**: TensorFlow/Keras (Python)
- **Model Serving**: TensorFlow.js
- **Deployment**: GitHub Pages

## Project Structure

- `frontend/` - Vue.js application
- `model/` - Python code for CNN model training
- `data/` - MNIST dataset
- `public/tfjs_model/` - Converted TensorFlow.js model

## Getting Started

1. Install dependencies:

```bash
cd frontend
npm install
```
