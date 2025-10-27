from shiny import ui
import datetime as dt


def report_ui():
    return ui.layout_column_wrap(
        1,
        ui.card(
            ui.card_header("필터"),
            ui.layout_columns(
                ui.input_date_range(
                    "rep_range", "분석 기간",
                    start=dt.date.today() - dt.timedelta(days=30),
                    end=dt.date.today(),
                ),
                ui.input_radio_buttons(
                    "rep_gran", "집계 단위", {"D": "일별", "W": "주별", "M": "월별"},
                    selected="D", inline=True,
                ),
                ui.input_slider("topn", "Top N (고가 시간대)", min=5, max=24, value=10),
                col_widths=[6, 3, 3],
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("사용량/요금 추이"),
                ui.div("막대/라인 복축 차트 영역 (Plotly)"),
            ),
            ui.card(
                ui.card_header("요일×시간대 요금 히트맵"),
                ui.div("히트맵 영역 (Plotly)"),
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Top-N 고가 시간대"),
                ui.div("테이블 영역"),
            ),
        ),
    )


def report_server(input, output, session):
    # 로직 없이 UI 틀만 유지
    pass
