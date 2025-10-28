from shiny import App, ui
from pathlib import Path
from modules import page_predict, page_report

www_dir = Path(__file__).parent / "www"

# ✅ head에 들어갈 리소스는 head_content로!
head_block = ui.head_content(
    # Google Font (Noto Sans KR)
    ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
    ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
    ui.tags.link(
        href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap",
        rel="stylesheet",
    ),
    # Bootstrap Icons
    ui.tags.link(
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
        rel="stylesheet",
    ),
    # Plotly
    ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js"),
    # Custom CSS (www/style.css)
    ui.tags.link(href="www/style.css", rel="stylesheet"),
)

app_ui = ui.page_fluid(
    head_block,  # ✅ 이게 실제 <head>로 들어감
    ui.page_navbar(
        ui.nav_panel("실시간 예측", page_predict.predict_ui()),
        ui.nav_panel("분석 리포트", page_report.report_ui()),
        title=ui.span(
            {"class": "brand"},
            ui.tags.i({"class": "bi bi-lightning-charge-fill me-2"}),
            "전기요금 대시보드",
        ),
        navbar_options=ui.navbar_options(
            id="main_nav",
            inverse=True,
        ),
    ),
)

def server(input, output, session):
    page_predict.predict_server(input, output, session)
    page_report.report_server(input, output, session)

app = App(app_ui, server, static_assets=www_dir)
