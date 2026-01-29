# ❤️ SmartHeart - Multi-Modal Cardiovascular Health Analysis System

A comprehensive machine learning application for analyzing cardiovascular health using ECG (Electrocardiogram) and PCG (Phonocardiogram) signals. This system combines deep learning models for ECG analysis with advanced audio processing for heart sound classification, providing healthcare professionals with powerful diagnostic tools.

## 🎯 System Overview

**SmartHeart** consists of two main components:

### Backend (Python + FastAPI)
- **ECG Analysis**: Deep learning model for arrhythmia and abnormality detection
- **PCG Analysis**: Gradient-boosted classifier for heart murmur detection
- **Multi-Modal Fusion**: Combined risk assessment from both modalities
- **RESTful API**: FastAPI server with file upload and prediction endpoints

### Frontend (Next.js + React)
- **Modern Web Interface**: Built with Next.js 16 and React 19
- **Interactive Visualizations**: Real-time charts, spectrograms, and waveforms
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Risk Assessment UI**: Color-coded indicators and confidence scores

## 🚀 Complete Setup Guide

Follow these instructions to set up and run the entire project from scratch after cloning from GitHub.

### Prerequisites

Before starting, ensure you have the following installed:

#### Required Software
- **Python 3.12+**: [Download Python](https://www.python.org/downloads/)
- **Node.js 20+**: [Download Node.js](https://nodejs.org/) (LTS version recommended)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **uv** (Python package manager): Install with `pip install uv`

#### Verify Installation
```bash
python --version    # Should be 3.12 or higher
node --version      # Should be 20.x or higher
npm --version       # Comes with Node.js
git --version       # Any recent version
uv --version        # Python package manager
```

---

## 📥 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodel-heart.git

# Navigate to the project directory
cd multimodel-heart
```

---

## 🐍 Step 2: Backend Setup

### 2.1 Navigate to Backend Directory
```bash
cd backend
```

### 2.2 Install Python Dependencies

Using `uv` (recommended - fastest):
```bash
uv sync
```

Or using traditional pip:
```bash
pip install -r requirements.txt
```

This will install all required packages:
- FastAPI & Uvicorn (API server)
- TensorFlow (ECG deep learning model)
- Librosa & SciPy (audio processing)
- Scikit-learn (PCG machine learning)
- NumPy & Pandas (data processing)

### 2.3 Verify Model Files

Ensure the following trained models are present:
```
backend/
├── heart_ecg_model/
│   └── ecg_model_final.keras       # ECG deep learning model
└── heart_sound_models/
    └── pcg_crnn_model.keras         # PCG classification model
```

If models are missing, you'll need to train them (see Training section below).

### 2.4 Start the Backend Server

```bash
# Run the FastAPI server
uv run main.py

# Or using uvicorn directly
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Keep this terminal running!** The backend server must be running for the frontend to work.

### 2.5 Test Backend API

Open a new terminal and test:
```bash
# Health check
curl http://localhost:8000/

# Should return: {"message": "Heart Analysis API is running"}
```

Or visit in your browser: [http://localhost:8000](http://localhost:8000)

---

## 🎨 Step 3: Frontend Setup

### 3.1 Open a New Terminal

Keep the backend terminal running. Open a **new terminal window** for frontend setup.

### 3.2 Navigate to Frontend Directory
```bash
cd frontend
```

### 3.3 Install Node Dependencies

```bash
npm install
```

This will install:
- Next.js 16 & React 19 (framework)
- Tailwind CSS v4 (styling)
- Recharts (data visualization)
- Axios (HTTP client)
- Framer Motion (animations)
- And other UI libraries

Installation may take 2-5 minutes depending on your internet speed.

### 3.4 Configure Environment Variables

Create a `.env` file in the frontend directory:

```bash
# Create .env file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env
```

Or manually create `frontend/.env` with:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3.5 Start the Frontend Development Server

```bash
npm run dev
```

Expected output:
```
  ▲ Next.js 16.1.4
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

### 3.6 Open the Application

Open your browser and navigate to:
```
http://localhost:3000
```

You should see the SmartHeart interface with file upload options for ECG and PCG files.

---

## 🎬 Step 4: Using the Application

### 4.1 Test with Demo Files

The project includes demo files for testing:

#### ECG Demo Files
```
backend/Demo_Files/ECG/
├── Normal/          # Normal ECG signals
│   ├── ecg_normal_01.csv
│   ├── ecg_normal_02.csv
│   └── ... (10 files)
└── Abnormal/        # Abnormal ECG signals
    ├── ecg_abnormal_01.csv
    ├── ecg_abnormal_02.csv
    └── ... (10 files)
```

#### PCG Demo Files
```
backend/Demo_Files/PCG/
├── Normal/          # Normal heart sounds
├── Abnormal/        # Abnormal heart sounds
└── unknown/         # Unclassified samples
```

### 4.2 Upload and Analyze

1. **Click "Upload ECG File"** or drag an ECG CSV file
2. **Click "Upload PCG File"** or drag a PCG WAV file
3. **Click "Analyze Signals"** button
4. **View Results** in the tabbed interface:
   - ECG tab: Waveform plot and risk score
   - PCG tab: Waveform, spectrogram, and risk score
   - Combined tab: Fused risk assessment

### 4.3 Understanding Results

**Risk Scores**:
- **0.0 - 0.5**: Low risk (Normal) - Green indicator
- **0.5 - 1.0**: High risk (Abnormal) - Red indicator

**Labels**:
- **Normal**: No significant abnormalities detected
- **Abnormal**: Potential cardiac issues detected

---

## 🔧 Step 5: Development Workflow

### Running Both Services Simultaneously

**Terminal 1 - Backend:**
```bash
cd backend
uv run main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### File Watching

Both servers support hot-reloading:
- **Backend**: FastAPI auto-reloads on Python file changes (with `--reload` flag)
- **Frontend**: Next.js hot-reloads on React/TypeScript file changes

### Making Changes

1. **Edit backend code** in `backend/` - Server auto-restarts
2. **Edit frontend code** in `frontend/` - Page hot-reloads
3. **Add new dependencies**:
   - Backend: Add to `pyproject.toml`, run `uv sync`
   - Frontend: `npm install package-name`

---

## 🧪 Step 6: Training Models (Optional)

If you need to retrain the models with your own data:

### ECG Model Training

```bash
cd backend
jupyter notebook output_colab_files/ecg_model_training_final.ipynb
```

Or use the training script:
```bash
uv run model_training.py
```

### PCG Model Training

```bash
cd backend
jupyter notebook output_colab_files/pcg_model_training_final.ipynb
```

Training notebooks include:
- Data preprocessing pipelines
- Model architecture definitions
- Training loops with validation
- Performance evaluation metrics

---

## 📁 Project Structure Overview

```
multimodel-heart/
│
├── backend/                          # Python FastAPI Backend
│   ├── main.py                      # FastAPI application entry point
│   ├── model_training.py            # Training scripts
│   ├── explainability.py            # Model interpretation tools
│   ├── pyproject.toml               # Python dependencies (uv)
│   ├── README.md                    # Backend documentation
│   │
│   ├── heart_ecg_model/             # ECG model files
│   │   └── ecg_model_final.keras
│   │
│   ├── heart_sound_models/          # PCG model files
│   │   └── pcg_crnn_model.keras
│   │
│   ├── Demo_Files/                  # Test data
│   │   ├── ECG/
│   │   │   ├── Normal/
│   │   │   └── Abnormal/
│   │   └── PCG/
│   │       ├── Normal/
│   │       └── Abnormal/
│   │
│   ├── ecg_data/                    # ECG datasets
│   │   ├── mitbih_train.csv
│   │   ├── mitbih_test.csv
│   │   ├── ptbdb_normal.csv
│   │   └── ptbdb_abnormal.csv
│   │
│   └── pcg_data1/                   # PCG datasets
│       └── pcg_data2/
│
└── frontend/                         # Next.js React Frontend
    ├── app/                         # Next.js App Router
    │   ├── page.tsx                # Main application page
    │   ├── layout.tsx              # Root layout
    │   └── globals.css             # Global styles
    │
    ├── components/                  # React components
    │   ├── charts/                 # Visualization components
    │   │   ├── SignalChart.tsx
    │   │   ├── SpectrogramViewer.tsx
    │   │   └── RiskBarChart.tsx
    │   └── ui/                     # UI components
    │       ├── Card.tsx
    │       └── Tabs.tsx
    │
    ├── services/                    # API integration
    │   └── api.ts
    │
    ├── package.json                 # Node dependencies
    ├── next.config.ts              # Next.js configuration
    ├── tailwind.config.ts          # Tailwind CSS config
    └── README.md                    # Frontend documentation
```

---

## 🛠️ Troubleshooting

### Backend Issues

#### Port 8000 Already in Use
```bash
# Find and kill the process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

#### Module Import Errors
```bash
# Reinstall dependencies
cd backend
uv sync --refresh
```

#### Model Files Missing
Download trained models from the releases page or retrain using the notebooks.

### Frontend Issues

#### Port 3000 Already in Use
```bash
# Use a different port
npm run dev -- -p 3001
```

#### Cannot Connect to Backend
- Verify backend is running on `http://localhost:8000`
- Check `.env` file has correct `NEXT_PUBLIC_API_URL`
- Check browser console for CORS errors

#### Module Not Found Errors
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### CSS Not Loading
```bash
# Rebuild Tailwind CSS
npm run build
npm run dev
```

### CORS Issues

If you see CORS errors, ensure the backend `main.py` has proper CORS configuration:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🚀 Production Deployment

### Backend Deployment

#### Using Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t smartheart-backend .
docker run -p 8000:8000 smartheart-backend
```

#### Using Gunicorn (Production WSGI)
```bash
uv run gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend Deployment

#### Vercel (Recommended)
1. Push code to GitHub
2. Import repository in [Vercel](https://vercel.com)
3. Add environment variable: `NEXT_PUBLIC_API_URL=<your-backend-url>`
4. Deploy automatically

#### Build Static Export
```bash
npm run build
npm run start
```

#### Using Docker
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

---

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (2.0 GHz)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended Requirements
- **CPU**: 8 cores (3.0 GHz)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (for faster ECG inference)
- **Storage**: 10 GB SSD

---

## 🧪 Running Tests

### Backend Tests
```bash
cd backend
uv run pytest
```

### Frontend Tests
```bash
cd frontend
npm run test
```

---

## 📚 API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Main Endpoints

#### `POST /predict`
Upload ECG and/or PCG files for analysis.

**Request**: multipart/form-data
- `ecg_file`: CSV file (optional)
- `pcg_file`: WAV file (optional)

**Response**: JSON
```json
{
  "ecg_risk": 0.85,
  "pcg_risk": 0.42,
  "combined_risk": 0.635,
  "ecg_label": "Abnormal",
  "pcg_label": "Normal",
  "combined_label": "Abnormal",
  "ecg_plot_data": [[0, 0.5], [1, 0.7], ...],
  "pcg_waveform_data": [0.1, 0.2, ...],
  "pcg_spectrogram_data": [[...], [...], ...]
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/multimodel-heart/issues)
- **Documentation**: Check [Backend README](backend/README.md) and [Frontend README](frontend/README.md)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🙏 Acknowledgments

- ECG Dataset: MIT-BIH Arrhythmia Database, PTB Diagnostic ECG Database
- PCG Dataset: PhysioNet Challenge 2016, Pascal Challenge
- Deep Learning Frameworks: TensorFlow, Keras
- Web Framework: FastAPI, Next.js

---

## 📈 Quick Start Summary

```bash
# 1. Clone repository
git clone https://github.com/yourusername/multimodel-heart.git
cd multimodel-heart

# 2. Setup backend (Terminal 1)
cd backend
uv sync
uv run main.py

# 3. Setup frontend (Terminal 2)
cd ../frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env
npm run dev

# 4. Open browser
# Visit: http://localhost:3000
```

**That's it! You're ready to analyze cardiovascular signals.** 🎉
