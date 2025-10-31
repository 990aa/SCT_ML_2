# Customer Segmentation Dashboard

A comprehensive web-based dashboard for customer segmentation analysis using K-Means clustering, built with FastAPI, Plotly, and Scikit-learn.

## 🌟 Features

- **Interactive Web Dashboard**: Beautiful, responsive UI with multiple pages for different analyses
- **Animated K-Means Visualization**: Watch the clustering algorithm in action with step-by-step animations
- **Comprehensive Analysis**: 
  - Data exploration and distribution analysis
  - Elbow method with multiple metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - 2D and 3D cluster visualizations
  - PCA dimensionality reduction
  - Quality metrics and statistical validation
  - Cluster stability analysis
  - Computational efficiency testing
  - Business insights and segment profiles
- **All Plotly Visualizations**: Every chart is interactive with zoom, pan, and hover capabilities
- **Complete Test Results**: Comprehensive test suite results displayed on the dashboard

## 📋 Requirements

- Python 3.10+
- UV package manager

## 🚀 Installation

1. Clone the repository and navigate to the project directory:
```bash
cd customer_segmentation
```

2. Install dependencies using UV:
```bash
uv sync
```

This will automatically:
- Create a virtual environment
- Install all required packages (FastAPI, Uvicorn, Plotly, Pandas, NumPy, Scikit-learn, etc.)

## 🎯 Running the Application

Start the server using UV:
```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The server will:
1. Load the customer data from `Mall_Customers.csv`
2. Run comprehensive clustering analysis
3. Generate all visualizations and animations
4. Start the web server on http://127.0.0.1:8000

Open your browser and navigate to http://127.0.0.1:8000 to access the dashboard.

## 📊 Dashboard Pages

### 1. Home (`/`)
Welcome page with feature overview and quick navigation

### 2. Data Overview (`/overview`)
- Dataset statistics and information
- Distribution plots for Age, Income, Spending Score, and Gender

### 3. Optimal Cluster Selection (`/elbow-method`)
- Elbow method visualization with WCSS
- Silhouette score analysis
- Calinski-Harabasz index
- Davies-Bouldin index
- Optimal K recommendations

### 4. K-Means Animation (`/kmeans-animation`)
**⭐ Main Feature**: Interactive animations showing:
- Step-by-step clustering process
- Centroid movements at each iteration
- Cluster assignments evolution
- Real-time metrics (Inertia, Silhouette score)
- Multiple feature pair visualizations:
  - Income vs Spending Score
  - Income vs Age
  - Spending Score vs Age
- Metrics evolution chart showing convergence

### 5. Clustering Results (`/clustering-results`)
- 2D scatter plots with centroids
- 3D interactive cluster visualization
- PCA visualization
- Color-coded clusters

### 6. Quality Metrics (`/quality-metrics`)
- Silhouette analysis plot
- Cluster characteristics (sizes, distances, feature importance)
- Statistical validation (ANOVA tests)
- Cluster stability analysis

### 7. Computational Efficiency (`/efficiency`)
- Computation time vs number of clusters
- Iterations to convergence
- Performance analysis

### 8. Business Insights (`/business-insights`)
- Business segment overview
- Cluster profiles table
- Segment interpretations:
  - High Value Customers
  - Budget Enthusiasts
  - Wealthy but Conservative
  - Low Value Customers
  - Average Customers

### 9. Test Results (`/test-results`)
**Complete test.py output** including:
- ✅ Quality metrics validation
- 🔄 Stability analysis results
- ⚡ Efficiency measurements
- 📊 Statistical significance tests
- 🎯 Cluster characteristics
- 💼 Business validation
- Overall test summary with pass/fail indicators

## 🛠️ Project Structure

```
customer_segmentation/
├── app.py                  # FastAPI application with all routes
├── analysis.py             # Data loading and clustering analysis
├── kmeans_animation.py     # K-Means animation generator
├── visualizations.py       # Plotly visualization functions
├── Mall_Customers.csv      # Dataset
├── pyproject.toml          # UV package configuration
├── templates/              # HTML templates
│   ├── home.html
│   ├── overview.html
│   ├── elbow.html
│   ├── animation.html
│   ├── clustering.html
│   ├── quality.html
│   ├── efficiency.html
│   ├── business.html
│   └── test_results.html
├── main.py                 # (Old file - not used)
└── test.py                 # (Old file - not used)
```

## 🎨 Technology Stack

- **Backend**: FastAPI (async web framework)
- **Server**: Uvicorn (ASGI server)
- **Visualizations**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Package Management**: UV (fast Python package installer)
- **Templates**: Jinja2

## 📈 Key Features Explained

### Animated K-Means Clustering
The animation feature provides unique insights into how the algorithm works:
1. **Initialization**: Shows random/k-means++ centroid initialization
2. **Assignment**: Points are colored by their nearest centroid
3. **Update**: Centroids move to the mean of their cluster
4. **Convergence**: Process repeats until centroids stabilize
5. **Metrics**: Real-time display of clustering quality at each step

### Comprehensive Testing
All test results from the original `test.py` are integrated into the dashboard:
- Clustering quality metrics with interpretation
- Stability analysis across multiple runs
- Computational efficiency measurements
- Statistical validation (ANOVA tests for feature significance)
- Business validation with segment interpretations

## 🔧 API Endpoints

- `GET /`: Home page
- `GET /overview`: Data overview
- `GET /elbow-method`: Optimal cluster selection
- `GET /kmeans-animation`: K-Means animation
- `GET /clustering-results`: Clustering visualizations
- `GET /quality-metrics`: Quality analysis
- `GET /efficiency`: Efficiency analysis
- `GET /business-insights`: Business insights
- `GET /test-results`: Test results
- `GET /api/analysis`: JSON API for analysis data
- `GET /api/animation`: JSON API for animation metadata

## 🎓 Learning Resources

This project demonstrates:
- Modern Python web development with FastAPI
- Interactive data visualization with Plotly
- Machine learning with Scikit-learn
- K-Means clustering algorithm
- Data analysis and business intelligence
- Package management with UV

## 📝 Notes

- The analysis runs automatically on server startup (takes ~10-15 seconds)
- All visualizations are interactive - you can zoom, pan, and hover
- The animations use Play/Pause controls and a slider for step-by-step navigation
- The server supports hot-reload (code changes automatically restart the server)

## 🙏 Credits

- Dataset: Mall Customers Dataset [https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data]
- Libraries: FastAPI, Plotly, Scikit-learn, Pandas, NumPy
- Package Manager: UV (Astral)


