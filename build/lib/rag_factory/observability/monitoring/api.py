"""REST API for monitoring dashboard."""

from flask import Flask, jsonify, request, render_template, send_from_directory
from typing import Dict, Any, Optional
import os

# Import observability components
from rag_factory.observability.logging.logger import get_logger
from rag_factory.observability.metrics.collector import get_collector
from rag_factory.observability.metrics.performance import get_performance_monitor

# Create Flask app
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Disable Flask logging to avoid conflicts
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/')
def index():
    """Dashboard home page."""
    try:
        return render_template('index.html')
    except Exception:
        # If templates not available, return simple HTML
        return """
        <!DOCTYPE html>
        <html>
        <head><title>RAG Factory Monitoring</title></head>
        <body>
            <h1>RAG Factory Monitoring Dashboard</h1>
            <p>API Endpoints:</p>
            <ul>
                <li><a href="/api/health">/api/health</a> - Health check</li>
                <li><a href="/api/metrics/summary">/api/metrics/summary</a> - Metrics summary</li>
                <li><a href="/api/metrics">/api/metrics</a> - All metrics</li>
                <li><a href="/api/system">/api/system</a> - System stats</li>
            </ul>
        </body>
        </html>
        """


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint.

    Returns:
        JSON with health status and version
    """
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "rag-factory-monitoring"
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get metrics for all strategies or a specific strategy.

    Query Parameters:
        strategy: Optional strategy name to filter by

    Returns:
        JSON with metrics data
    """
    strategy = request.args.get('strategy')
    collector = get_collector()

    try:
        metrics = collector.get_metrics(strategy)
        return jsonify({
            "success": True,
            "data": metrics
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/metrics/summary', methods=['GET'])
def get_summary():
    """Get overall metrics summary across all strategies.

    Returns:
        JSON with aggregated metrics
    """
    collector = get_collector()

    try:
        summary = collector.get_summary()
        return jsonify({
            "success": True,
            "data": summary
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/metrics/timeseries', methods=['GET'])
def get_timeseries():
    """Get time-series data for a metric.

    Query Parameters:
        metric: Metric name (e.g., "strategy_name_latency")
        duration: Time window in minutes (default: 60)

    Returns:
        JSON with time-series data points
    """
    metric_name = request.args.get('metric')
    duration = int(request.args.get('duration', 60))

    if not metric_name:
        return jsonify({
            "success": False,
            "error": "metric parameter is required"
        }), 400

    collector = get_collector()

    try:
        data = collector.get_time_series(metric_name, duration)

        return jsonify({
            "success": True,
            "data": [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'labels': point.labels
                }
                for point in data
            ]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get list of all strategies being tracked.

    Returns:
        JSON with list of strategy names
    """
    collector = get_collector()

    try:
        strategies = collector.get_strategy_names()
        return jsonify({
            "success": True,
            "data": strategies
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/system', methods=['GET'])
def get_system_stats():
    """Get current system performance statistics.

    Returns:
        JSON with system metrics (CPU, memory, disk)
    """
    monitor = get_performance_monitor()

    try:
        stats = monitor.get_current_system_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/performance', methods=['GET'])
def get_performance_stats():
    """Get performance statistics for tracked operations.

    Query Parameters:
        operation: Optional operation name to filter by

    Returns:
        JSON with performance statistics
    """
    operation = request.args.get('operation')
    monitor = get_performance_monitor()

    try:
        if operation:
            stats = monitor.get_stats(operation)
        else:
            stats = monitor.get_all_stats()

        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/metrics/reset', methods=['POST'])
def reset_metrics():
    """Reset metrics for a strategy or all strategies.

    Request Body (JSON):
        strategy: Optional strategy name to reset

    Returns:
        JSON with success status
    """
    data = request.get_json() or {}
    strategy = data.get('strategy')

    collector = get_collector()

    try:
        collector.reset_metrics(strategy)
        return jsonify({
            "success": True,
            "message": f"Metrics reset for {strategy if strategy else 'all strategies'}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create and configure Flask app.

    Args:
        config: Optional Flask configuration

    Returns:
        Configured Flask app
    """
    if config:
        app.config.update(config)

    return app


def start_dashboard(
    host: str = '0.0.0.0',
    port: int = 8080,
    debug: bool = False,
    threaded: bool = True
):
    """Start the monitoring dashboard server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8080)
        debug: Enable debug mode (default: False)
        threaded: Enable threading (default: True)

    Example:
        ```python
        from rag_factory.observability.monitoring.api import start_dashboard

        # Start dashboard
        start_dashboard(host='localhost', port=8080)
        ```
    """
    logger = get_logger()
    logger.info(
        f"Starting monitoring dashboard on {host}:{port}",
        host=host,
        port=port,
        debug=debug
    )

    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=threaded,
        use_reloader=False  # Avoid duplicate processes
    )


if __name__ == '__main__':
    start_dashboard()
