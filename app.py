"""
FastAPI application for Customer Segmentation Dashboard
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from analysis import load_data, get_comprehensive_analysis
from kmeans_animation import create_kmeans_animation
from visualizations import (
    create_distribution_plots,
    create_elbow_plots,
    create_2d_cluster_plot,
    create_3d_cluster_plot,
    create_pca_plot,
    create_silhouette_plot,
    create_cluster_characteristics_plot,
    create_efficiency_plot,
    create_business_segments_plot,
    create_cluster_profiles_table,
    create_statistical_validation_plot,
    create_stability_plot,
)
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Customer Segmentation Dashboard")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables to store analysis results
analysis_results = None
animation_results = None


def initialize_analysis():
    """Run comprehensive analysis on startup"""
    global analysis_results, animation_results

    print("Loading data...")
    df = load_data("Mall_Customers.csv")

    print("Running comprehensive analysis...")
    analysis_results = get_comprehensive_analysis(df, n_clusters=5)

    print("Creating K-means animations...")
    from analysis import prepare_features

    X_std, scaler = prepare_features(df)

    animation_results = create_kmeans_animation(
        X_std,
        n_clusters=5,
        feature_pairs=[(0, 1), (0, 2), (1, 2)],
        feature_names=["Annual Income (k$)", "Spending Score (1-100)", "Age"],
    )

    print("Analysis complete!")


@app.on_event("startup")
async def startup_event():
    """Run analysis on startup"""
    initialize_analysis()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Single page dashboard with all content"""
    if analysis_results is None or animation_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    # Prepare all data
    df = pd.DataFrame(analysis_results["data"])
    centroids = np.array(analysis_results["clustering"]["centroids_original"])

    # Create all visualizations
    dist_plot = create_distribution_plots(df)
    elbow_plot = create_elbow_plots(analysis_results["elbow_method"])

    # Animations
    animations = animation_results["animations"]
    metrics_animation = animation_results["metrics_animation"]
    animation_htmls = {}
    for key, fig in animations.items():
        animation_htmls[key] = fig.to_html(full_html=False, include_plotlyjs="cdn")
    metrics_html = metrics_animation.to_html(full_html=False, include_plotlyjs="cdn")

    # Clustering plots
    plot_2d_income_spending = create_2d_cluster_plot(
        df, "Annual Income (k$)", "Spending Score (1-100)", centroids=centroids
    )
    plot_2d_income_age = create_2d_cluster_plot(
        df, "Annual Income (k$)", "Age", centroids=centroids
    )
    plot_3d = create_3d_cluster_plot(df)
    plot_pca = create_pca_plot(
        analysis_results["pca"], analysis_results["clustering"]["labels"]
    )

    # Quality metrics
    silhouette_plot = create_silhouette_plot(analysis_results["silhouette_analysis"])
    characteristics_plot = create_cluster_characteristics_plot(
        analysis_results["cluster_characteristics"]
    )
    statistical_plot = create_statistical_validation_plot(
        analysis_results["statistical_validation"]
    )
    stability_plot = create_stability_plot(analysis_results["stability"])

    # Efficiency
    efficiency_plot = create_efficiency_plot(analysis_results["efficiency"])

    # Business
    business_plot = create_business_segments_plot(
        analysis_results["business_validation"]
    )
    profiles_table = create_cluster_profiles_table(
        analysis_results["business_validation"]
    )

    # Get efficiency metrics for the optimal k
    optimal_k = analysis_results["clustering"]["n_clusters"]
    efficiency_data = analysis_results["efficiency"]
    try:
        k_index = efficiency_data["k"].index(optimal_k)
        execution_time = efficiency_data["time_per_run"][k_index]
        n_iterations = efficiency_data["iterations"][k_index]
    except (ValueError, IndexError):
        execution_time = efficiency_data["time_per_run"][0] if efficiency_data["time_per_run"] else 0
        n_iterations = efficiency_data["iterations"][0] if efficiency_data["iterations"] else 0
    
    efficiency_summary = {
        "execution_time": execution_time,
        "n_iterations": int(n_iterations),
        "memory_usage": 0.0  # Placeholder
    }

    return templates.TemplateResponse(
        "single_page.html",
        {
            "request": request,
            "title": "Customer Segmentation Dashboard",
            "basic_info": analysis_results["basic_info"],
            "distribution_plot": dist_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "elbow_plot": elbow_plot.to_html(full_html=False, include_plotlyjs="cdn"),
            "elbow_data": analysis_results["elbow_method"],
            "optimal_k": optimal_k,
            "animations": animation_htmls,
            "animation_2d": animation_htmls.get("animation_1", ""),
            "animation_metrics": metrics_html,
            "metrics_summary": animation_results["metrics_summary"],
            "cluster_2d": plot_2d_income_spending.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "cluster_3d": plot_3d.to_html(full_html=False, include_plotlyjs="cdn"),
            "pca_plot": plot_pca.to_html(full_html=False, include_plotlyjs="cdn"),
            "cluster_profiles": profiles_table.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "quality_metrics": analysis_results["quality_metrics"],
            "silhouette_plot": silhouette_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "characteristics_plot": characteristics_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "statistical_plot": statistical_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "stability_plot": stability_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "stability_metrics": analysis_results["stability"],
            "efficiency_metrics": efficiency_summary,
            "efficiency_plot": efficiency_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "business_segments": business_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "business_analysis": analysis_results["business_validation"],
            "statistical_validation": analysis_results["statistical_validation"],
        },
    )


@app.get("/overview", response_class=HTMLResponse)
async def overview(request: Request):
    """Data overview page"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    basic_info = analysis_results["basic_info"]

    # Create distribution plots
    df = pd.DataFrame(analysis_results["data"])
    dist_plot = create_distribution_plots(df)

    return templates.TemplateResponse(
        "overview.html",
        {
            "request": request,
            "title": "Data Overview",
            "basic_info": basic_info,
            "distribution_plot": dist_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
        },
    )


@app.get("/elbow-method", response_class=HTMLResponse)
async def elbow_method_page(request: Request):
    """Elbow method analysis page"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    elbow_data = analysis_results["elbow_method"]
    elbow_plot = create_elbow_plots(elbow_data)

    return templates.TemplateResponse(
        "elbow.html",
        {
            "request": request,
            "title": "Optimal Cluster Selection",
            "elbow_plot": elbow_plot.to_html(full_html=False, include_plotlyjs="cdn"),
            "optimal_k": elbow_data,
        },
    )


@app.get("/clustering-results", response_class=HTMLResponse)
async def clustering_results(request: Request):
    """Clustering results visualization"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    df = pd.DataFrame(analysis_results["data"])
    centroids = np.array(analysis_results["clustering"]["centroids_original"])

    # Create various cluster plots
    plot_2d_income_spending = create_2d_cluster_plot(
        df, "Annual Income (k$)", "Spending Score (1-100)", centroids=centroids
    )

    plot_2d_income_age = create_2d_cluster_plot(
        df, "Annual Income (k$)", "Age", centroids=centroids
    )

    plot_3d = create_3d_cluster_plot(df)

    plot_pca = create_pca_plot(
        analysis_results["pca"], analysis_results["clustering"]["labels"]
    )

    return templates.TemplateResponse(
        "clustering.html",
        {
            "request": request,
            "title": "Clustering Results",
            "plot_2d_1": plot_2d_income_spending.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "plot_2d_2": plot_2d_income_age.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "plot_3d": plot_3d.to_html(full_html=False, include_plotlyjs="cdn"),
            "plot_pca": plot_pca.to_html(full_html=False, include_plotlyjs="cdn"),
        },
    )


@app.get("/kmeans-animation", response_class=HTMLResponse)
async def kmeans_animation_page(request: Request):
    """K-means animation page"""
    if animation_results is None:
        return HTMLResponse("Animation not ready. Please refresh.")

    # Get animations
    animations = animation_results["animations"]
    metrics_animation = animation_results["metrics_animation"]
    metrics_summary = animation_results["metrics_summary"]

    # Convert animations to HTML
    animation_htmls = {}
    for key, fig in animations.items():
        animation_htmls[key] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    metrics_html = metrics_animation.to_html(full_html=False, include_plotlyjs="cdn")

    return templates.TemplateResponse(
        "animation.html",
        {
            "request": request,
            "title": "K-Means Algorithm Animation",
            "animations": animation_htmls,
            "metrics_animation": metrics_html,
            "metrics_summary": metrics_summary,
        },
    )


@app.get("/quality-metrics", response_class=HTMLResponse)
async def quality_metrics(request: Request):
    """Quality metrics and validation page"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    quality = analysis_results["quality_metrics"]
    silhouette_plot = create_silhouette_plot(analysis_results["silhouette_analysis"])
    characteristics_plot = create_cluster_characteristics_plot(
        analysis_results["cluster_characteristics"]
    )
    statistical_plot = create_statistical_validation_plot(
        analysis_results["statistical_validation"]
    )
    stability_plot = create_stability_plot(analysis_results["stability"])

    return templates.TemplateResponse(
        "quality.html",
        {
            "request": request,
            "title": "Quality Metrics",
            "quality_metrics": quality,
            "silhouette_plot": silhouette_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "characteristics_plot": characteristics_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "statistical_plot": statistical_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "stability_plot": stability_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
        },
    )


@app.get("/efficiency", response_class=HTMLResponse)
async def efficiency_analysis(request: Request):
    """Efficiency analysis page"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    efficiency_plot = create_efficiency_plot(analysis_results["efficiency"])

    return templates.TemplateResponse(
        "efficiency.html",
        {
            "request": request,
            "title": "Computational Efficiency",
            "efficiency_plot": efficiency_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "efficiency_data": analysis_results["efficiency"],
        },
    )


@app.get("/business-insights", response_class=HTMLResponse)
async def business_insights(request: Request):
    """Business insights and segment profiles"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    business_plot = create_business_segments_plot(
        analysis_results["business_validation"]
    )

    profiles_table = create_cluster_profiles_table(
        analysis_results["business_validation"]
    )

    return templates.TemplateResponse(
        "business.html",
        {
            "request": request,
            "title": "Business Insights",
            "business_plot": business_plot.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "profiles_table": profiles_table.to_html(
                full_html=False, include_plotlyjs="cdn"
            ),
            "business_data": analysis_results["business_validation"],
        },
    )


@app.get("/test-results", response_class=HTMLResponse)
async def test_results(request: Request):
    """Comprehensive test results from test.py"""
    if analysis_results is None:
        return HTMLResponse("Analysis not ready. Please refresh.")

    # Gather all test results
    test_data = {
        "quality_metrics": analysis_results["quality_metrics"],
        "stability": analysis_results["stability"],
        "efficiency": analysis_results["efficiency"],
        "statistical_validation": analysis_results["statistical_validation"],
        "cluster_characteristics": analysis_results["cluster_characteristics"],
        "business_validation": analysis_results["business_validation"],
    }

    return templates.TemplateResponse(
        "test_results.html",
        {
            "request": request,
            "title": "Comprehensive Test Results",
            "test_data": test_data,
        },
    )


@app.get("/api/analysis")
async def get_analysis_data():
    """API endpoint to get analysis data as JSON"""
    if analysis_results is None:
        return {"error": "Analysis not ready"}
    return analysis_results


@app.get("/api/animation")
async def get_animation_data():
    """API endpoint to get animation data as JSON"""
    if animation_results is None:
        return {"error": "Animation not ready"}

    # Return only metrics summary (animations are too large for JSON)
    return {
        "metrics_summary": animation_results["metrics_summary"],
        "history_length": animation_results["history_length"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
