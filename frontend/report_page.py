"""
EAGLE-DAR — Reports Page

View, generate and download damage assessment reports.
"""

from __future__ import annotations
import json
import streamlit as st
from pathlib import Path


def render_report_page() -> None:
    st.title("Damage Assessment Reports")
    st.caption("View saved reports and download PDF/JSON outputs.")

    tab_view, tab_generate = st.tabs(["View Saved Reports", "Generate Report"])

    with tab_view:
        _render_saved_reports()
    with tab_generate:
        _render_generate_report()


def _render_saved_reports() -> None:
    reports_dir = Path("reports")
    if not reports_dir.exists():
        st.info("No reports directory found. Run an assessment first.")
        return

    json_files = sorted(reports_dir.glob("*.json"), reverse=True)
    if not json_files:
        st.info("No saved reports found.")
        return

    selected = st.selectbox("Select Report", [f.name for f in json_files])
    if selected:
        path = reports_dir / selected
        try:
            data = json.loads(path.read_text())
            _render_report_view(data)
        except Exception as exc:
            st.error(f"Could not load report: {exc}")


def _render_report_view(data: dict) -> None:
    st.subheader(f"Report: {data.get('report_id', 'N/A')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buildings", data.get("total_buildings", "N/A"))
    col2.metric("Damaged", data.get("damaged_buildings", "N/A"))
    col3.metric("Severity", data.get("overall_severity", "N/A"))

    if qa := data.get("qa_pairs"):
        st.subheader("Q&A Summary")
        for item in qa:
            with st.expander(item.get("question", "Q")):
                st.write(item.get("answer", ""))

    with st.expander("Full Report JSON"):
        st.json(data)

    st.download_button(
        "Download JSON",
        data=json.dumps(data, indent=2, default=str).encode(),
        file_name=f"{data.get('report_id', 'report')}.json",
        mime="application/json",
    )


def _render_generate_report() -> None:
    st.subheader("Generate Report from Completed Assessment")

    if "inf_report" not in st.session_state:
        st.info("Run an assessment on the Damage Assessment page first.")
        return

    report = st.session_state["inf_report"]
    output_dir = st.text_input("Output Directory", value="reports")

    if st.button("Save JSON Report"):
        try:
            from backend.reports.report_generator import generate_json_report
            path = generate_json_report(report, Path(output_dir) / f"{report.report_id}.json")
            st.success(f"Saved: {path}")
        except Exception as exc:
            st.error(f"Save error: {exc}")

    if st.button("Generate PDF Report"):
        try:
            from backend.reports.report_generator import generate_pdf_report
            heatmap = st.session_state.get("inf_heatmap", {}).get("heatmap")
            path = generate_pdf_report(
                report, heatmap,
                Path(output_dir) / f"{report.report_id}.pdf",
            )
            if path:
                with open(path, "rb") as f:
                    st.download_button("Download PDF", data=f.read(),
                                       file_name=path.name, mime="application/pdf")
            else:
                st.warning("PDF generation requires fpdf2. Install with: pip install fpdf2")
        except Exception as exc:
            st.error(f"PDF error: {exc}")
