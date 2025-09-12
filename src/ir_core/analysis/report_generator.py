# src/ir_core/analysis/report_generator.py

"""
Automated report generation for analysis results.

This module provides comprehensive report generation capabilities
for performance analysis, error analysis, and trend analysis.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from jinja2 import Template

from .core import AnalysisResult


class AnalysisReportGenerator:
    """
    Automated report generator for analysis results.

    Generates comprehensive reports in multiple formats (HTML, JSON, Markdown)
    with performance summaries, error analysis, and trend insights.
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load report templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Template]:
        """Load Jinja2 templates for report generation."""
        templates = {}

        # HTML template for comprehensive reports
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 0.9em; }
        .chart-container { margin: 20px 0; text-align: center; }
        .recommendations { background: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #17a2b8; }
        .recommendations ul { margin: 0; padding-left: 20px; }
        .recommendations li { margin: 8px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on {{ generation_date }}</p>
            <p>{{ description }}</p>
        </div>

        {% for section in sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            {{ section.content | safe }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """

        # Markdown template for simple reports
        markdown_template = """
# {{ title }}

**Generated on:** {{ generation_date }}
**Description:** {{ description }}

{% for section in sections %}
## {{ section.title }}

{{ section.content_markdown | safe }}

{% endfor %}
        """

        templates['html'] = Template(html_template)
        templates['markdown'] = Template(markdown_template)

        return templates

    def generate_performance_report(
        self,
        result: AnalysisResult,
        format: str = 'html',
        include_charts: bool = True
    ) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            result: AnalysisResult to report on
            format: Output format ('html', 'markdown', 'json')
            include_charts: Whether to include chart references

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_report_{timestamp}.{format}"
        filepath = self.output_dir / filename

        # Prepare report data
        report_data = self._prepare_performance_data(result, include_charts)

        if format == 'html':
            content = self.templates['html'].render(**report_data)
        elif format == 'markdown':
            content = self.templates['markdown'].render(**report_data)
        elif format == 'json':
            content = json.dumps(report_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(filepath)

    def generate_error_analysis_report(
        self,
        result: AnalysisResult,
        format: str = 'html'
    ) -> str:
        """
        Generate an error analysis report.

        Args:
            result: AnalysisResult to analyze
            format: Output format

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_analysis_report_{timestamp}.{format}"
        filepath = self.output_dir / filename

        # Prepare error analysis data
        report_data = self._prepare_error_analysis_data(result)

        if format == 'html':
            content = self.templates['html'].render(**report_data)
        elif format == 'markdown':
            content = self.templates['markdown'].render(**report_data)
        elif format == 'json':
            content = json.dumps(report_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(filepath)

    def generate_trend_analysis_report(
        self,
        results: List[AnalysisResult],
        format: str = 'html'
    ) -> str:
        """
        Generate a trend analysis report from multiple results.

        Args:
            results: List of AnalysisResult objects
            format: Output format

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_analysis_report_{timestamp}.{format}"
        filepath = self.output_dir / filename

        # Prepare trend analysis data
        report_data = self._prepare_trend_analysis_data(results)

        if format == 'html':
            content = self.templates['html'].render(**report_data)
        elif format == 'markdown':
            content = self.templates['markdown'].render(**report_data)
        elif format == 'json':
            content = json.dumps(report_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(filepath)

    def _prepare_performance_data(self, result: AnalysisResult, include_charts: bool) -> Dict[str, Any]:
        """Prepare data for performance report."""
        # Calculate key metrics
        metrics_html = f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{result.map_score:.4f}</div>
                <div class="metric-label">Mean Average Precision (MAP)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.retrieval_success_rate:.1%}</div>
                <div class="metric-label">Retrieval Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.total_queries}</div>
                <div class="metric-label">Total Queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.avg_query_length:.1f}</div>
                <div class="metric-label">Average Query Length</div>
            </div>
        </div>
        """

        # Precision@K table
        precision_table = "<table><tr><th>K</th><th>Precision</th></tr>"
        for k in sorted(result.precision_at_k.keys()):
            precision_table += f"<tr><td>{k}</td><td>{result.precision_at_k[k]:.4f}</td></tr>"
        precision_table += "</table>"

        # Domain distribution
        domain_table = "<table><tr><th>Domain</th><th>Count</th><th>Percentage</th></tr>"
        total_domains = sum(result.domain_distribution.values())
        for domain, count in sorted(result.domain_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_domains * 100) if total_domains > 0 else 0
            domain_table += f"<tr><td>{domain}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        domain_table += "</table>"

        # Recommendations
        recommendations_html = "<div class='recommendations'><h3>Recommendations</h3><ul>"
        for rec in result.recommendations[:5]:  # Limit to top 5
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul></div>"

        sections = [
            {
                "title": "Key Performance Metrics",
                "content": metrics_html,
                "content_markdown": f"""
- **MAP**: {result.map_score:.4f}
- **Retrieval Success Rate**: {result.retrieval_success_rate:.1%}
- **Total Queries**: {result.total_queries}
- **Average Query Length**: {result.avg_query_length:.1f}
                """
            },
            {
                "title": "Precision@K Analysis",
                "content": precision_table,
                "content_markdown": "\n".join([f"- P@{k}: {result.precision_at_k[k]:.4f}" for k in sorted(result.precision_at_k.keys())])
            },
            {
                "title": "Domain Distribution",
                "content": domain_table,
                "content_markdown": "\n".join([f"- {domain}: {count} ({count/total_domains*100:.1f}%)" for domain, count in sorted(result.domain_distribution.items(), key=lambda x: x[1], reverse=True)])
            },
            {
                "title": "Recommendations",
                "content": recommendations_html,
                "content_markdown": "\n".join([f"- {rec}" for rec in result.recommendations[:5]])
            }
        ]

        return {
            "title": "Scientific QA Performance Report",
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Comprehensive analysis of retrieval performance metrics and recommendations",
            "sections": sections
        }

    def _prepare_error_analysis_data(self, result: AnalysisResult) -> Dict[str, Any]:
        """Prepare data for error analysis report."""
        # Error categories
        error_table = "<table><tr><th>Error Type</th><th>Count</th><th>Percentage</th></tr>"
        total_errors = sum(result.error_categories.values())
        for error_type, count in sorted(result.error_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            error_table += f"<tr><td>{error_type}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        error_table += "</table>"

        # Domain error rates
        domain_error_table = "<table><tr><th>Domain</th><th>Error Rate</th></tr>"
        for domain, rate in sorted(result.domain_error_rates.items(), key=lambda x: x[1], reverse=True):
            status_class = "status-danger" if rate > 0.5 else "status-warning" if rate > 0.3 else "status-good"
            domain_error_table += f"<tr><td>{domain}</td><td class='{status_class}'>{rate:.1%}</td></tr>"
        domain_error_table += "</table>"

        sections = [
            {
                "title": "Error Categories",
                "content": error_table,
                "content_markdown": "\n".join([f"- {error_type}: {count} ({count/total_errors*100:.1f}%)" for error_type, count in sorted(result.error_categories.items(), key=lambda x: x[1], reverse=True)])
            },
            {
                "title": "Domain Error Rates",
                "content": domain_error_table,
                "content_markdown": "\n".join([f"- {domain}: {rate:.1%}" for domain, rate in sorted(result.domain_error_rates.items(), key=lambda x: x[1], reverse=True)])
            }
        ]

        return {
            "title": "Error Analysis Report",
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Detailed analysis of retrieval errors and failure patterns",
            "sections": sections
        }

    def _prepare_trend_analysis_data(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Prepare data for trend analysis report."""
        if not results:
            return {
                "title": "Trend Analysis Report",
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": "No data available for trend analysis",
                "sections": []
            }

        # Calculate trends
        map_scores = [r.map_score for r in results]
        success_rates = [r.retrieval_success_rate for r in results]

        map_trend = "↗️ Improving" if len(map_scores) > 1 and map_scores[-1] > map_scores[0] else "↘️ Declining" if len(map_scores) > 1 and map_scores[-1] < map_scores[0] else "➡️ Stable"
        success_trend = "↗️ Improving" if len(success_rates) > 1 and success_rates[-1] > success_rates[0] else "↘️ Declining" if len(success_rates) > 1 and success_rates[-1] < success_rates[0] else "➡️ Stable"

        # Trend table
        trend_table = "<table><tr><th>Run</th><th>MAP</th><th>Success Rate</th><th>Query Count</th></tr>"
        for i, result in enumerate(results):
            trend_table += f"<tr><td>Run {i+1}</td><td>{result.map_score:.4f}</td><td>{result.retrieval_success_rate:.1%}</td><td>{result.total_queries}</td></tr>"
        trend_table += "</table>"

        sections = [
            {
                "title": "Performance Trends",
                "content": f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{map_trend}</div>
                        <div class="metric-label">MAP Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{success_trend}</div>
                        <div class="metric-label">Success Rate Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(results)}</div>
                        <div class="metric-label">Total Runs Analyzed</div>
                    </div>
                </div>
                {trend_table}
                """,
                "content_markdown": f"""
**MAP Trend:** {map_trend}
**Success Rate Trend:** {success_trend}
**Total Runs:** {len(results)}

| Run | MAP | Success Rate | Query Count |
|-----|-----|--------------|-------------|
""" + "".join([f"| Run {i+1} | {r.map_score:.4f} | {r.retrieval_success_rate:.1%} | {r.total_queries} |\n" for i, r in enumerate(results)])
            }
        ]

        return {
            "title": "Trend Analysis Report",
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Analysis of performance trends across multiple runs",
            "sections": sections
        }


# Convenience functions
def generate_performance_report(result: AnalysisResult, format: str = 'html') -> str:
    """Convenience function to generate performance report."""
    generator = AnalysisReportGenerator()
    return generator.generate_performance_report(result, format)


def generate_error_report(result: AnalysisResult, format: str = 'html') -> str:
    """Convenience function to generate error analysis report."""
    generator = AnalysisReportGenerator()
    return generator.generate_error_analysis_report(result, format)


def generate_trend_report(results: List[AnalysisResult], format: str = 'html') -> str:
    """Convenience function to generate trend analysis report."""
    generator = AnalysisReportGenerator()
    return generator.generate_trend_analysis_report(results, format)
