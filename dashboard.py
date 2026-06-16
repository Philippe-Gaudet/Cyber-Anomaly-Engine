"""Barcode Streamlit dashboard entry point.

Run with:
    streamlit run src/barcode/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `import barcode` works even when this file is
# launched as a standalone script (e.g. `streamlit run src/barcode/dashboard.py`
# from a Python that doesn't have the package installed editable).
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st


def _check_environment() -> None:
    """Surface a clean Streamlit error if the wrong Python is running this app.

    The bare `streamlit` on a fresh macOS install often resolves to the system
    Python 3.13 binary, which has no giotto-tda wheel and therefore cannot
    import `gtda`. Detect that early and tell the user exactly what to run.
    """
    try:
        import gtda  # noqa: F401
    except ModuleNotFoundError:
        st.set_page_config(page_title="Barcode - env error", layout="centered")
        st.error(
            f"""**Environment mismatch.**

Streamlit is running under `{sys.executable}` (Python {sys.version_info.major}.{sys.version_info.minor}),
but `giotto-tda` is not installed there. The project's `.venv` (Python 3.11) is the only
interpreter that can run this dashboard.

**Fix - relaunch with the venv's Streamlit:**

```bash
./run_dashboard.sh
```

or explicitly:

```bash
.venv/bin/streamlit run src/barcode/dashboard.py
```
"""
        )
        st.stop()


_check_environment()

from barcode.ui.components import cicids_available, data_source_strip, sidebar_section
from barcode.ui.pages import attack_analysis, benchmark, live_detection, topology_explorer
from barcode.ui.theme import GRADIENTS, PALETTE, inject_css


def main() -> None:
    st.set_page_config(
        page_title="Barcode",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        f"""
        <div style="position:relative;overflow:hidden;display:flex;align-items:flex-end;
                    justify-content:space-between;gap:24px;
                    padding:22px 26px;margin:0 0 18px 0;border-radius:18px;
                    border:1px solid {PALETTE['border']};
                    background:{GRADIENTS['panel']};backdrop-filter:blur(10px);
                    box-shadow:inset 0 1px 0 rgba(255,255,255,0.05),0 22px 48px -28px rgba(0,0,0,0.85);">
          <span style="position:absolute;left:0;top:0;right:0;height:2px;
                       background:{GRADIENTS['accent']};opacity:0.7;"></span>
          <span style="position:absolute;right:-80px;top:-120px;width:280px;height:280px;border-radius:50%;
                       background:radial-gradient(circle,rgba(139,92,246,0.16),transparent 65%);
                       pointer-events:none;"></span>
          <div style="position:relative;min-width:0;">
            <div style="display:flex;align-items:center;gap:13px;font-family:'IBM Plex Mono',monospace;
                        color:{PALETTE['text']};font-size:30px;line-height:1;font-weight:600;letter-spacing:0;">
              <span style="width:9px;height:32px;border-radius:3px;
                           background:{GRADIENTS['accent']};
                           box-shadow:0 0 22px rgba(45,212,255,0.45);"></span>
              <span>BARCODE</span>
            </div>
            <div style="margin-top:9px;font-family:'DM Sans',sans-serif;color:{PALETTE['text2']};
                        font-size:13px;line-height:1.35;">
              Topological anomaly detection for network traffic
            </div>
          </div>
          <div style="position:relative;display:flex;align-items:center;justify-content:flex-end;
                      gap:8px;flex-wrap:wrap;">
            <span style="display:inline-flex;align-items:center;height:29px;padding:0 12px;border-radius:9px;
                         border:1px solid rgba(255,255,255,0.08);background:{GRADIENTS['raise']};
                         color:{PALETTE['text2']};font-family:'JetBrains Mono',monospace;font-size:11px;
                         box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);">
              CICIDS2017
            </span>
            <span style="display:inline-flex;align-items:center;height:29px;padding:0 12px;border-radius:9px;
                         border:1px solid rgba(255,255,255,0.08);background:{GRADIENTS['raise']};
                         color:{PALETTE['text2']};font-family:'JetBrains Mono',monospace;font-size:11px;
                         box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);">
              Persistent Homology
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            f"""
            <div style="position:relative;overflow:hidden;padding:16px 16px 16px 16px;margin:0 0 10px 0;
                        border-radius:16px;border:1px solid {PALETTE['border']};
                        background:{GRADIENTS['panel']};
                        box-shadow:inset 0 1px 0 rgba(255,255,255,0.05),0 16px 32px -22px rgba(0,0,0,0.85);">
              <span style="position:absolute;left:0;top:0;bottom:0;width:3px;
                           background:{GRADIENTS['accent']};opacity:0.8;"></span>
              <div style="display:flex;align-items:center;gap:12px;">
                <span style="width:9px;height:38px;border-radius:3px;
                             background:{GRADIENTS['accent']};
                             box-shadow:0 0 20px rgba(45,212,255,0.35);"></span>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;color:{PALETTE['text']};
                              font-size:18px;font-weight:600;line-height:1;letter-spacing:0;">
                    BARCODE
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;color:{PALETTE['text2']};
                              font-size:10px;margin-top:6px;text-transform:uppercase;letter-spacing:0.04em;">
                    Operations console
                  </div>
                </div>
              </div>
              <div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:14px;">
                <span style="height:25px;padding:0 9px;display:inline-flex;align-items:center;
                             border-radius:8px;border:1px solid rgba(255,255,255,0.08);
                             background:{GRADIENTS['raise']};box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);
                             color:{PALETTE['text2']};font-family:'JetBrains Mono',monospace;font-size:10px;">
                  Topology
                </span>
                <span style="height:25px;padding:0 9px;display:inline-flex;align-items:center;
                             border-radius:8px;border:1px solid rgba(255,255,255,0.08);
                             background:{GRADIENTS['raise']};box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);
                             color:{PALETTE['text2']};font-family:'JetBrains Mono',monospace;font-size:10px;">
                  Detection
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        real_data = cicids_available()
        sidebar_section("Detector", "Scoring engine")
        model_name = st.selectbox(
            "Detector",
            ["TopoDetector", "IsolationForest", "Autoencoder", "GNN", "Fused"],
            index=0,
            key="global_detector_model",
            label_visibility="collapsed",
        )
        st.session_state["selected_model"] = model_name

        data_source_strip(real_data)

    tabs = st.tabs([
        "Live Detection",
        "\u2003Topology Explorer",
        "\u2003Benchmark",
        "\u2003Attack Analysis",
    ])
    with tabs[0]:
        live_detection.render()
    with tabs[1]:
        topology_explorer.render()
    with tabs[2]:
        benchmark.render()
    with tabs[3]:
        attack_analysis.render()


if __name__ == "__main__":
    main()
