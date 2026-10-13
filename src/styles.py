"""Custom CSS styles and design helpers for ResolveAI."""

import streamlit as st


def apply_custom_styles():
    """Injects sleek, modern CSS for cards, badges, and clean typography."""
    st.markdown("""
        <style>
            /* Main container polish */
            .main .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2.5rem;
                max-width: 1200px;
            }

            /* Metric cards styling */
            .resolve-card {
                background: #ffffff;
                border-radius: 10px;
                padding: 1.25rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
                margin-bottom: 1rem;
            }

            .metric-box {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.2rem;
            }
            .metric-lbl {
                font-size: 0.85rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 600;
            }

            /* Badges */
            .badge {
                display: inline-block;
                padding: 0.25rem 0.65rem;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 0.4rem;
                margin-bottom: 0.3rem;
            }
            .badge-critical { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
            .badge-high { background-color: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
            .badge-medium { background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
            .badge-low { background-color: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }

            .badge-pos { background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }
            .badge-neu { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
            .badge-neg { background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

            .badge-entity {
                background-color: #ede9fe;
                color: #5b21b6;
                border: 1px solid #c4b5fd;
                font-family: monospace;
            }

            /* Code / tree display */
            pre.tree-view {
                background: #0f172a;
                color: #38bdf8;
                padding: 1rem;
                border-radius: 8px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.85rem;
                overflow-x: auto;
            }

            /* Sidebar custom headers */
            .sidebar-title {
                font-size: 1.4rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 0.2rem;
            }
            .sidebar-sub {
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 1.2rem;
            }
        </style>
    """, unsafe_allow_html=True)


def get_priority_badge(priority: str) -> str:
    """Returns HTML badge for priority level."""
    p_lower = (priority or "low").lower()
    if p_lower == "critical":
        return f'<span class="badge badge-critical">● Critical Priority</span>'
    elif p_lower == "high":
        return f'<span class="badge badge-high">● High Priority</span>'
    elif p_lower == "medium":
        return f'<span class="badge badge-medium">● Medium Priority</span>'
    else:
        return f'<span class="badge badge-low">● Low Priority</span>'


def get_sentiment_badge(sentiment: str, score: float = 0.0) -> str:
    """Returns HTML badge for sentiment label and score."""
    s_lower = (sentiment or "neutral").lower()
    score_str = f"({score:+.2f})" if score is not None else ""
    if s_lower == "positive":
        return f'<span class="badge badge-pos">▲ Positive {score_str}</span>'
    elif s_lower == "negative":
        return f'<span class="badge badge-neg">▼ Negative {score_str}</span>'
    else:
        return f'<span class="badge badge-neu">■ Neutral {score_str}</span>'
