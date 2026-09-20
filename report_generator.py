"""
Phase 3: Generate HTML Report

Creates beautiful, interactive HTML report of results.
"""

import json


def generate_html_report(
    answers_file: str = "phase3_answers.json",
    evaluation_file: str = "phase3_evaluation.json",
    benchmark_file: str = "phase2_benchmark_results.json",
    output_file: str = "phase3_report.html",
):
    """Generate comprehensive HTML report."""

    # Load data
    with open(answers_file) as f:
        answers = json.load(f)

    with open(evaluation_file) as f:
        evaluations = json.load(f)

    with open(benchmark_file) as f:
        benchmark = json.load(f)

    # Calculate stats
    avg_faith = sum(e["metrics"]["faithfulness"] for e in evaluations) / len(evaluations)
    avg_relevance = sum(e["metrics"]["relevance"] for e in evaluations) / len(evaluations)
    avg_recall = sum(e["metrics"]["context_recall"] for e in evaluations) / len(evaluations)
    overall_avg = sum(e["metrics"]["average"] for e in evaluations) / len(evaluations)

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GraphKnowledge Phase 3: Complete Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .queries {{
            padding: 40px;
        }}
        .query-result {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .query-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .query-header h3 {{
            color: #667eea;
            margin-bottom: 8px;
        }}
        .query-header p {{
            color: #666;
            font-size: 0.9em;
        }}
        .query-body {{
            padding: 20px;
        }}
        .query-body h4 {{
            color: #333;
            margin: 15px 0 8px 0;
            font-size: 1em;
        }}
        .query-body p {{
            color: #666;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-score {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #999;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        .score-high {{ color: #22c55e; }}
        .score-medium {{ color: #f59e0b; }}
        .score-low {{ color: #ef4444; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GraphKnowledge Phase 3</h1>
            <p>Complete Answer Generation & Evaluation Pipeline</p>
        </div>

        <div class="summary">
            <div class="metric-card">
                <div class="label">Faithfulness</div>
                <div class="metric-card__value" style="color: #667eea;">{avg_faith:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Relevance</div>
                <div class="metric-card__value" style="color: #764ba2;">{avg_relevance:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Context Recall</div>
                <div class="metric-card__value" style="color: #667eea;">{avg_recall:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Overall Score</div>
                <div class="metric-card__value" style="color: #764ba2;">{overall_avg:.3f}</div>
            </div>
        </div>

        <div class="queries">
            <h2 style="margin-bottom: 30px; color: #333;">Answer Results</h2>
"""

    # Add each query result
    for answer_data in answers:
        query_id = answer_data["query_id"]
        question = answer_data["question"]
        answer = answer_data["answer"]

        eval_data = next(e for e in evaluations if e["query_id"] == query_id)
        metrics = eval_data["metrics"]

        html += f"""
            <div class="query-result">
                <div class="query-header">
                    <h3>{query_id}: {question}</h3>
                    <p>Method: {answer_data['method'].upper()} | Passages: {answer_data['passages_used']}</p>
                </div>
                <div class="query-body">
                    <h4>Generated Answer:</h4>
                    <p>{answer}</p>
                    <div class="metrics-row">
                        <div class="metric">
                            <div class="metric-score">{metrics['faithfulness']:.3f}</div>
                            <div class="metric-label">Faithfulness</div>
                        </div>
                        <div class="metric">
                            <div class="metric-score">{metrics['relevance']:.3f}</div>
                            <div class="metric-label">Relevance</div>
                        </div>
                        <div class="metric">
                            <div class="metric-score">{metrics['average']:.3f}</div>
                            <div class="metric-label">Average</div>
                        </div>
                    </div>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>GraphKnowledge Phase 3 Complete | Generated using Groq + Mixtral-8x7b | Portfolio Ready</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML
    with open(output_file, "w") as f:
        f.write(html)

    print(f"✓ HTML report saved to {output_file}")


if __name__ == "__main__":
    generate_html_report()