"""UI Helper functions and Talegri-inspired Clean SaaS Design System for Streamlit."""

import streamlit as st
from typing import Dict, Any, List, Optional


def inject_custom_css():
    """Inject the clean, bright, modern Scandinavian SaaS design system inspired by Talegri."""
    st.markdown(
        """
        <style>
        /* Import clean modern fonts: Plus Jakarta Sans & Outfit */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

        :root {
            color-scheme: light;
            --matrix-primary: #4ba3e3;
            --matrix-primary-hover: #3289cb;
            --matrix-text: #1e293b;
            --matrix-muted: #64748b;
            --matrix-border: #e2e8f0;
            --matrix-surface: #ffffff;
            --matrix-background: #f8fafc;
        }

        /* Global Application Background & Base Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--matrix-background) !important;
            color: var(--matrix-text);
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: var(--matrix-background) !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        a {
            color: var(--matrix-primary-hover);
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p {
            color: var(--matrix-text);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea {
            background-color: var(--matrix-surface);
            border-color: var(--matrix-border);
            color: var(--matrix-text);
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stDateInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--matrix-primary) !important;
            box-shadow: 0 0 0 2px rgba(75, 163, 227, 0.18) !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--matrix-primary-hover);
            border-bottom-color: var(--matrix-primary);
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--matrix-border);
            border-radius: 12px;
            overflow: hidden;
            background: var(--matrix-surface);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* Top Navigation Header (Talegri style) */
        .talegri-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 24px;
            background: #ffffff;
            border-bottom: 1px solid #f1f5f9;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.02);
        }
        .talegri-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: 'Outfit', sans-serif;
            font-size: 1.45rem;
            font-weight: 800;
            color: #1e293b;
            letter-spacing: -0.5px;
        }
        .talegri-logo-dots {
            display: flex;
            gap: 3px;
            align-items: center;
        }
        .logo-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
        }
        .logo-dot.red { background: #f87171; }
        .logo-dot.yellow { background: #fbbf24; }
        .logo-dot.blue { background: #38bdf8; }
        .logo-dot.purple { background: #a855f7; }

        .talegri-nav-links {
            display: flex;
            align-items: center;
            gap: 24px;
            font-size: 0.88rem;
            font-weight: 600;
            color: #64748b;
        }
        .talegri-nav-pill {
            background: #4ba3e3;
            color: #ffffff !important;
            padding: 7px 18px;
            border-radius: 9999px;
            font-size: 0.84rem;
            font-weight: 700;
            text-decoration: none;
            box-shadow: 0 2px 8px rgba(75, 163, 227, 0.3);
        }

        /* Talegri Hero Section */
        .talegri-hero {
            position: relative;
            background: #ffffff;
            border-radius: 24px;
            padding: 40px 48px;
            margin-bottom: 30px;
            border: 1px solid #f1f5f9;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04);
            overflow: hidden;
        }
        .talegri-hero-shape {
            position: absolute;
            top: -20px;
            right: -40px;
            width: 420px;
            height: 380px;
            background: linear-gradient(135deg, rgba(224, 242, 254, 0.6) 0%, rgba(240, 249, 255, 0.9) 100%);
            border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
            z-index: 0;
            pointer-events: none;
        }
        .talegri-hero-content {
            position: relative;
            z-index: 1;
            max-width: 680px;
        }
        .talegri-hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.75rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
            letter-spacing: -1px;
            margin: 0 0 16px 0;
        }
        .talegri-hero-title span {
            color: #4ba3e3;
        }
        .talegri-hero-desc {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #475569;
            margin-bottom: 24px;
            font-weight: 400;
        }
        .talegri-pill-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background-color: #4ba3e3;
            color: #ffffff;
            padding: 12px 28px;
            border-radius: 9999px;
            font-size: 0.95rem;
            font-weight: 700;
            box-shadow: 0 4px 14px rgba(75, 163, 227, 0.35);
            border: none;
            transition: all 0.2s ease;
        }

        /* Tech logos / Brand credibility strip */
        .talegri-brand-strip {
            margin-top: 32px;
            padding-top: 20px;
            border-top: 1px solid #f1f5f9;
        }
        .talegri-brand-title {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 12px;
        }
        .talegri-brand-logos {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 0.86rem;
            font-weight: 700;
            color: #64748b;
        }
        .talegri-tech-pill {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #475569;
        }

        /* Stepper Navigation (Talegri style) */
        .talegri-stepper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 9999px;
            padding: 6px 12px;
            margin-bottom: 28px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
            overflow-x: auto;
        }
        .talegri-step-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 18px;
            border-radius: 9999px;
            font-size: 0.84rem;
            font-weight: 700;
            color: #64748b;
            white-space: nowrap;
            transition: all 0.2s ease;
        }
        .talegri-step-item.active {
            background: #4ba3e3;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(75, 163, 227, 0.35);
        }
        .talegri-step-item.completed {
            background: #f0fdf4;
            color: #15803d;
        }
        .talegri-step-circle {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #f1f5f9;
            color: #475569;
            font-size: 0.74rem;
            font-weight: 800;
        }
        .talegri-step-item.active .talegri-step-circle {
            background: #ffffff;
            color: #4ba3e3;
        }

        /* Talegri 4-Column Feature/KPI Cards */
        .talegri-card {
            background: #ffffff;
            border: 1px solid #f1f5f9;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
            text-align: center;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            position: relative;
        }
        .talegri-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.07);
        }
        .talegri-icon-circle {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            margin: 0 auto 14px auto;
            font-size: 1.4rem;
        }
        .talegri-icon-circle.yellow {
            background: #fef3c7;
            color: #d97706;
            border: 2px solid #fde68a;
        }
        .talegri-icon-circle.cyan {
            background: #e0f2fe;
            color: #0284c7;
            border: 2px solid #bae6fd;
        }
        .talegri-icon-circle.purple {
            background: #f3e8ff;
            color: #7e22ce;
            border: 2px solid #e9d5ff;
        }
        .talegri-icon-circle.green {
            background: #dcfce7;
            color: #15803d;
            border: 2px solid #bbf7d0;
        }
        .talegri-card-val {
            font-family: 'Outfit', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.5px;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .talegri-card-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #64748b;
            margin-bottom: 6px;
        }
        .talegri-card-sub {
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.4;
        }

        /* Streamlit Native Button Overrides -> Sky Blue Pills */
        button[kind="primary"] {
            background: #4ba3e3 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 10px 26px !important;
            font-size: 0.92rem !important;
            box-shadow: 0 4px 14px rgba(75, 163, 227, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        button[kind="primary"]:hover {
            background: #3b91d1 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(75, 163, 227, 0.45) !important;
        }
        button[kind="secondary"] {
            background: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 9999px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        button[kind="secondary"]:hover {
            border-color: #4ba3e3 !important;
            color: #4ba3e3 !important;
        }

        /* Callout Boxes */
        .talegri-callout {
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #f1f5f9;
            padding: 20px 24px;
            margin: 16px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }
        .talegri-callout.info {
            border-left: 5px solid #4ba3e3;
        }
        .talegri-callout.warning {
            border-left: 5px solid #f59e0b;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--matrix-surface) !important;
            border-right: 1px solid var(--matrix-border);
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: var(--matrix-surface);
        }

        @media (max-width: 1100px) {
            .block-container {
                padding: 1.5rem 1.5rem 2rem;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 1rem;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                flex: 1 1 calc(50% - 1rem) !important;
                min-width: 0 !important;
            }
            .talegri-hero {
                padding: 32px;
            }
            .talegri-hero-title {
                font-size: 2.25rem;
            }
            .talegri-card {
                padding: 18px;
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding: 1rem 1rem 1.5rem;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                flex-basis: 100% !important;
            }
            .talegri-nav {
                gap: 12px;
                padding: 12px 16px;
                border-radius: 12px;
            }
            .talegri-brand {
                font-size: 1.25rem;
            }
            .talegri-nav-links {
                gap: 0;
                font-size: 0.78rem;
            }
            .talegri-nav-links > span:not(.talegri-nav-pill) {
                display: none;
            }
            .talegri-nav-pill {
                padding: 6px 12px;
                white-space: nowrap;
            }
            .talegri-hero {
                padding: 24px 20px;
                margin-bottom: 20px;
                border-radius: 18px;
            }
            .talegri-hero-shape {
                right: -180px;
                opacity: 0.65;
            }
            .talegri-hero .talegri-hero-title {
                font-size: clamp(1.8rem, 8vw, 2.25rem);
                letter-spacing: -0.5px;
            }
            .talegri-hero-desc {
                font-size: 0.95rem;
            }
            .talegri-brand-strip {
                margin-top: 24px;
                padding-top: 16px;
            }
            .talegri-brand-logos {
                gap: 8px;
            }
            .talegri-tech-pill {
                font-size: 0.72rem;
                padding: 4px 10px;
            }
            .talegri-stepper {
                justify-content: flex-start;
                margin-bottom: 20px;
                border-radius: 16px;
            }
            .talegri-step-item {
                padding: 7px 12px;
                font-size: 0.78rem;
            }
            .talegri-card {
                padding: 18px;
            }
            .talegri-card-val {
                font-size: 1.8rem;
            }
            [data-testid="stDataFrame"],
            [data-testid="stPlotlyChart"] {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_top_navbar():
    """Render top clean Talegri-style navbar with brand logo dots and pill badge."""
    st.markdown(
        """
        <div class="talegri-nav">
            <div class="talegri-brand">
                <div class="talegri-logo-dots">
                    <span class="logo-dot red"></span>
                    <span class="logo-dot yellow"></span>
                    <span class="logo-dot blue"></span>
                    <span class="logo-dot purple"></span>
                </div>
                <span>matrix<b>core</b></span>
            </div>
            <div class="talegri-nav-links">
                <span>Data Quality</span>
                <span>SVD Engine</span>
                <span>Confidence Metric</span>
                <span class="talegri-nav-pill">v2.0 Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hero_banner():
    """Render the bright, clean, Talegri-style hero section."""
    st.markdown(
        """
        <div class="talegri-hero">
            <div class="talegri-hero-shape"></div>
            <div class="talegri-hero-content">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <div class="talegri-logo-dots">
                        <span class="logo-dot red"></span>
                        <span class="logo-dot yellow"></span>
                        <span class="logo-dot blue"></span>
                        <span class="logo-dot purple"></span>
                    </div>
                    <span style="font-weight:700; font-size:0.82rem; color:#4ba3e3; letter-spacing:0.8px; text-transform:uppercase;">
                        Matrix Reconstruction Engine
                    </span>
                </div>
                <h1 class="talegri-hero-title">
                    Welcome to <span>matrix completion</span> done right!
                </h1>
                <p class="talegri-hero-desc">
                    Isolate missing and corrupted entries across any matrix dataset.
                    Reconstruct them with precision using SVD low-rank pattern projection,
                    and quantify confidence for every repaired cell.
                </p>
                <div class="talegri-brand-strip">
                    <div class="talegri-brand-title">Built with state-of-the-art computational foundations</div>
                    <div class="talegri-brand-logos">
                        <span class="talegri-tech-pill">⚡ SVD / TruncatedSVD</span>
                        <span class="talegri-tech-pill">🔍 Scikit-Learn</span>
                        <span class="talegri-tech-pill">📊 Plotly Interactive</span>
                        <span class="talegri-tech-pill">🛡️ Monte Carlo Jackknife</span>
                        <span class="talegri-tech-pill">🚀 NumPy & Pandas</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_step_flow_bar(current_step: int = 1):
    """Render the Talegri rounded pill stepper bar."""
    steps = [
        (1, "Data Input"),
        (2, "Detect Errors"),
        (3, "Pattern Analysis"),
        (4, "Reconstruction"),
        (5, "Confidence"),
        (6, "Dashboard & Export")
    ]
    html = '<div class="talegri-stepper">'
    for num, name in steps:
        if num == current_step:
            cls = "active"
            icon_disp = f"{num}"
        elif num < current_step:
            cls = "completed"
            icon_disp = "✓"
        else:
            cls = ""
            icon_disp = f"{num}"

        html += f"""
        <div class="talegri-step-item {cls}">
            <span class="talegri-step-circle">{icon_disp}</span>
            <span>{name}</span>
        </div>
        """
        if num < len(steps):
            html += '<span style="color:#cbd5e1; font-size:11px;">•</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: Any,
    subtext: str = "",
    theme: str = "default",
    icon: str = "📊"
):
    """Render a Talegri-style clean white card with rounded colored icon circle."""
    circle_color = {
        "success": "green",
        "warning": "yellow",
        "danger": "yellow",
        "purple": "purple",
        "cyan": "cyan",
        "default": "cyan"
    }.get(theme, "cyan")

    st.markdown(
        f"""
        <div class="talegri-card">
            <div class="talegri-icon-circle {circle_color}">{icon}</div>
            <div class="talegri-card-val">{value}</div>
            <div class="talegri-card-label">{label}</div>
            <div class="talegri-card-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
