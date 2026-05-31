import streamlit as st
import requests
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

import textwrap

def apply_lbg_theme():
    """
    Applies the custom Lloyds Banking Group (LBG) styling tokens to the active page.
    This reads from visualization/styles.css and injects it.
    """
    import os
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load styles.css: {str(e)}")
    else:
        # Fallback to base styling if styles.css not found
        st.markdown("""
        <style>
            .stApp {
                background-color: #060913 !important;
                color: #F8FAFC !important;
            }
        </style>
        """, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, delta: str = None, delta_direction: str = "up", icon: str = None, status: str = "normal"):
    """
    Renders a premium HTML/CSS KPI card.
    delta_direction: 'up' (positive/green), 'down' (negative/red), or 'neutral'
    status: 'normal', 'success', 'warning', 'danger'
    """
    delta_html = ""
    if delta:
        if delta_direction == "up":
            delta_class = "delta-positive"
            arrow = "▲"
        elif delta_direction == "down":
            delta_class = "delta-negative"
            arrow = "▼"
        else:
            delta_class = "delta-neutral"
            arrow = "•"
        delta_html = f'<div class="kpi-delta {delta_class}"><span>{arrow}</span> {delta}</div>'
        
    icon_html = f'<span style="float: right; font-size: 1.5rem; opacity: 0.85;">{icon}</span>' if icon else ""
    
    card_style = ""
    if status == "danger":
        card_style = 'style="border-left: 4px solid #EF4444;"'
    elif status == "warning":
        card_style = 'style="border-left: 4px solid #F59E11;"'
    elif status == "success":
        card_style = 'style="border-left: 4px solid #10B981;"'
    elif status == "info":
        card_style = 'style="border-left: 4px solid #00E5FF;"'
        
    html_content = textwrap.dedent(f"""
    <div class="kpi-card" {card_style}>
        {icon_html}
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """).strip()
    st.markdown(html_content, unsafe_allow_html=True)

def render_status_badge(status: str, label: str) -> str:
    """
    Returns HTML string for a colored status badge.
    status: 'danger' (red), 'warning' (yellow), 'success' (green), 'info' (cyan)
    """
    badge_class = f"badge-{status}"
    return f'<span class="status-badge {badge_class}">{label}</span>'

def render_section_header(title: str, subtitle: str = None, icon: str = None):
    """
    Renders section headers with custom subtitle hierarchy.
    """
    icon_prefix = f"{icon} " if icon else ""
    st.markdown(f"### {icon_prefix}{title}")
    if subtitle:
        st.markdown(f'<div style="color: #94A3B8; font-size: 0.9rem; margin-top: -0.6rem; margin-bottom: 1.25rem;">{subtitle}</div>', unsafe_allow_html=True)

def render_alert_banner(title: str, message: str, alert_type: str = "info"):
    """
    Renders a premium operations alert banner.
    alert_type: 'danger', 'warning', 'success', 'info'
    """
    banner_class = f"alert-banner-{alert_type}" if alert_type != "info" else ""
    html_content = textwrap.dedent(f"""
    <div class="alert-banner {banner_class}">
        <div class="alert-title">{title}</div>
        <div class="alert-body">{message}</div>
    </div>
    """).strip()
    st.markdown(html_content, unsafe_allow_html=True)


def init_session_state(defaults: dict):
    """
    Initializes session state parameters if they are not already set.
    """
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def make_api_request(method: str, path: str, json: dict = None, params: dict = None, timeout: float = 5.0):
    """
    Encapsulates backend HTTP calls with centralized logging, error handling, and timeout boundaries.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=json, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
            
        if response.status_code in [200, 201, 202]:
            return response.json()
        else:
            detail = response.json().get("detail", "No error details provided by backend API.")
            logger.error(f"API Error [{response.status_code}] on {path}: {detail}")
            st.error(f"⚠️ **Backend Error ({response.status_code})**: {detail}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"API Timeout on {path}")
        st.error("🔴 **API Request Timeout**: The backend service took too long to respond. Please check if the FastAPI daemon is overloaded.")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"API Connection Error on {path}")
        st.error("🔴 **API Connection Offline**: Unable to establish connection with the backend services at localhost:8000. Ensure the FastAPI Uvicorn server is running.")
        return None
    except Exception as e:
        logger.error(f"Unexpected API call exception on {path}: {str(e)}")
        st.error(f"🔴 **System Error**: An unexpected error occurred: {str(e)}")
        return None
