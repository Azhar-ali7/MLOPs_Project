# 🫀 Heart Disease Prediction - Streamlit UI

## Unified Interface for Complete MLOps Workflow

This Streamlit application provides an integrated interface for testing the entire MLOps pipeline.

### Features

- **🏠 Home**: Overview and quick start guide
- **🔮 Prediction**: Interactive prediction interface
- **📈 Metrics**: View API metrics and statistics
- **🧪 Testing**: Run integration and load tests
- **📚 Documentation**: Complete documentation

### Quick Start

#### Option 1: One Command (Recommended)
```bash
./start_app.sh
```

This will:
1. Start the API server on port 8000
2. Start the Streamlit UI on port 8501
3. Open the UI in your browser

#### Option 2: Manual Start
```bash
# Terminal 1 - Start API
python -m uvicorn src.api:app --port 8000

# Terminal 2 - Start UI
streamlit run ui/streamlit_app.py
```

### Access

- **UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Benefits of Unified Interface

Consolidated access instead of managing multiple localhost ports:

✅ **Unified Interface** - Everything accessible from one dashboard  
✅ **Streamlined Setup** - Two services (API + UI)  
✅ **Integrated Testing** - Test workflows directly in browser  
✅ **Professional UX** - Clean, intuitive interface  
✅ **Efficient** - Optimized for development and demonstration  

### For Production Monitoring

For production, you can still use the full monitoring stack:
```bash
./scripts/deploy-docker.sh
```

The Streamlit UI provides an optimized development and demonstration experience!

### Screenshots

The UI includes:
- Real-time API health monitoring
- Interactive prediction form
- Metrics visualization
- Load testing tools
- Complete documentation

### Technologies

- **Streamlit** - Web UI framework
- **FastAPI** - Backend API
- **Scikit-learn** - ML models
- **Requests** - HTTP client

### Assignment Benefits

This unified approach effectively demonstrates:
- ✅ Working ML pipeline
- ✅ API integration
- ✅ Testing capabilities
- ✅ User-friendly interface
- ✅ Complete workflow

Provides a cohesive demonstration experience compared to navigating multiple localhost ports!

