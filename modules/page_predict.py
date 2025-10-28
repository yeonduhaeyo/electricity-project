from shiny import ui, render, reactive
import datetime as dt
from utils.ui_components import kpi, section_caption
from viz.plot_placeholders import hourly_prediction

def predict_ui():
    return ui.layout_column_wrap(
        ui.card(
            ui.card_header(ui.span("입력"), ui.span({"class": "badge-accent ms-2"}, "모델 미연결")),
            ui.layout_columns(
                ui.input_date("pred_date", "예측 날짜", value=dt.date.today()),
                ui.input_action_button("pred_run", "예측 실행", class_="btn btn-primary"),
                ui.input_numeric("pred_usage", "예상 사용량(kWh)", value=50, min=0),
                col_widths=[4, 3, 5],
            ),
            section_caption("날짜를 선택하고 ‘예측 실행’을 누르면 예측 영역이 갱신됩니다. (현재는 자리 표시자)"),
        ),
        ui.card(
            ui.card_header("KPI"),
            ui.div(
                kpi("평균 예측단가", "— 원/kWh"),
                kpi("최소 / 최대", "— / — 원/kWh"),
                kpi("예상 비용", "— 원", "입력 사용량 × 평균단가"),
                class_="kpi-row",
            ),
            ui.div({"class": "small-muted mt-2"}, "피크 시간대 강조는 모델 연결 후 활성화됩니다."),
        ),
        ui.card(
            ui.card_header("시간대별 요금 예측"),
            hourly_prediction(),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, "Baseline vs Model 비교, 신뢰구간 밴드는 모델 연결 후 표시됩니다."),
        ),
        width=1,  # <- deprecation 경고 방지(키워드 인자, 마지막에)
    )

def predict_server(input, output, session):
    # 버튼 클릭 상태만 관리 (지금은 자리 표시자 갱신 트리거)
    last_run = reactive.Value(None)

    @reactive.event(input.pred_run)
    def _run():
        last_run.set(input.pred_date())

    @output
    @render.ui
    def pred_kpis():
        _ = last_run()  # 클릭 시 갱신
        return ui.div()  # KPI는 위 카드에서 정적 자리로 대체(필요시 동적 업데이트로 변경)
    
    @output
    @render.ui
    def pred_plot():
        _ = last_run()
        return ui.div()  # Plotly는 viz 모듈 교체 시 렌더
