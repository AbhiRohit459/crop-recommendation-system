# 🌱 Crop Recommendation System

An intelligent web application that uses Machine Learning to recommend the best crops for cultivation based on soil and climate conditions.

## ✨ Features

### 🎯 Core Features
- **AI-Powered Recommendations**: Uses Random Forest Classifier for accurate crop predictions
- **Top 3 Recommendations**: Get multiple crop options ranked by confidence scores
- **Confidence Scores**: See how confident the model is about each recommendation
- **Detailed Crop Information**: View season, duration, and water requirements for each crop

### 🎨 Enhanced UI/UX
- **Modern Design**: Beautiful gradient-based interface with smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Input Validation**: Real-time validation with helpful range indicators
- **Form Persistence**: Your input values are preserved after submission
- **Loading States**: Visual feedback during prediction processing

### 📊 Input Features
- **Nitrogen (N)**: Soil nitrogen content in kg/ha
- **Phosphorus (P)**: Soil phosphorus content in kg/ha
- **Potassium (K)**: Soil potassium content in kg/ha
- **Temperature**: Average temperature in °C
- **Humidity**: Relative humidity percentage
- **pH Level**: Soil pH value (0-14 scale)
- **Rainfall**: Annual rainfall in mm

### 📈 Output Information
For each recommended crop, you get:
- **Crop Name**: The recommended crop
- **Confidence Score**: Prediction confidence percentage
- **Season**: Best season for cultivation (Kharif/Rabi/Summer/Year-round)
- **Duration**: Crop growth duration
- **Water Requirements**: Water needs (Low/Moderate/High)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbhiRohit459/crop-recommendation-system.git
   cd crop-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

### 🌐 Deploy on Render

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push
   ```

2. **Go to Render Dashboard**
   - Visit [render.com](https://render.com)
   - Sign up/Login with your GitHub account

3. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `AbhiRohit459/crop-recommendation-system`
   - Use these settings:
     - **Name**: `crop-recommendation-system`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Plan**: Free tier is fine for testing

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be live at: `https://crop-recommendation-system.onrender.com` (or your custom domain)

**Note**: The first deployment may take a few minutes. Subsequent deployments are faster.

## 📦 Dependencies

- **Flask** (>=3.0.0): Web framework
- **NumPy** (>=1.26.0): Numerical computing
- **scikit-learn** (>=1.3.0): Machine learning library

## 🎯 Supported Crops

The system can recommend 22 different crops:
1. Rice
2. Maize
3. Jute
4. Cotton
5. Coconut
6. Papaya
7. Orange
8. Apple
9. Muskmelon
10. Watermelon
11. Grapes
12. Mango
13. Banana
14. Pomegranate
15. Lentil
16. Blackgram
17. Mungbean
18. Mothbeans
19. Pigeonpeas
20. Kidneybeans
21. Chickpea
22. Coffee

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: MinMaxScaler + StandardScaler
- **Input Features**: 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Output Classes**: 22 crops

### File Structure
```
Crop_Recommendation-main/
├── app.py                 # Main Flask application
├── model.pkl             # Trained ML model
├── minmaxscaler.pkl      # MinMax scaler
├── standscaler.pkl       # Standard scaler
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend template
├── static/
│   └── crop.png          # Crop icon
└── README.md             # This file
```

## 🎨 UI Enhancements

- **Gradient Backgrounds**: Modern purple gradient theme
- **Card-based Design**: Beautiful recommendation cards with hover effects
- **Icon Integration**: Bootstrap Icons for better visual communication
- **Responsive Grid**: Adaptive layout for all screen sizes
- **Smooth Animations**: Transitions and hover effects

## 🛠️ Recent Enhancements

### Version 2.0 Updates
- ✅ Added top 3 crop recommendations
- ✅ Implemented confidence scores
- ✅ Enhanced UI with modern design
- ✅ Added crop information (season, duration, water)
- ✅ Input validation with range checking
- ✅ Form data persistence
- ✅ Better error handling
- ✅ Loading states and animations
- ✅ Responsive design improvements

## 📝 Usage Example

1. Enter your soil and climate data:
   - Nitrogen: 90 kg/ha
   - Phosphorus: 42 kg/ha
   - Potassium: 43 kg/ha
   - Temperature: 20.9°C
   - Humidity: 82%
   - pH: 6.5
   - Rainfall: 202.9 mm

2. Click "Get Recommendations"

3. View the top 3 recommended crops with:
   - Confidence percentages
   - Crop details (season, duration, water needs)
   - Ranked recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- UI designed with Bootstrap 5
- Icons by Bootstrap Icons

---

**Made with ❤️ for better agriculture**
