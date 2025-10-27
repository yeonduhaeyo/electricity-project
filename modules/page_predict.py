from shiny import ui
import datetime as dt


def predict_ui():
    return ui.layout_column_wrap(
        1,
        ui.card(
            ui.card_header("입력"),
            ui.layout_columns(
                ui.input_date("pred_date", "예측 날짜", value=dt.date.today()),
                ui.input_action_button("pred_run", "예측 실행", class_="btn-primary"),
                ui.input_numeric("pred_usage", "예상 사용량(kWh)", value=50, min=0),
                col_widths=[4, 3, 5],
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("KPI"),
                ui.div("평균/최소/최대/예상비용 등 KPI를 여기에 표시합니다."),
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("시간대별 요금 예측"),
                ui.div("라인 차트 영역 (Plotly)"),
                ui.div({"class": "text-muted small"}, "* 지금은 UI만 구성했습니다."),
            ),
        ),
    )


def predict_server(input, output, session):
    # 로직 없이 UI 틀만 유지
    pass
