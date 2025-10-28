from shiny import ui, render, reactive
import datetime as dt
from utils.ui_components import kpi
from viz.plot_placeholders import hourly_prediction


def predict_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="styles.css"),

        # ── 상단 툴바 ──
        ui.card(
            ui.card_header("실시간 제어"),
            ui.div(
                ui.div(
                    ui.input_action_button("btn_start", "시작", class_="btn-soft"),
                    ui.input_action_button("btn_stop",  "멈춤", class_="btn-soft"),
                    ui.input_action_button("btn_reset", "리셋", class_="btn-soft"),
                    class_="btns",
                ),
                ui.div(
                    ui.span("시간 단위 선택", class_="seg-label"),
                    ui.input_radio_buttons(
                        "time_unit",
                        None,
                        {"D": "일별", "H": "시간대별", "Q": "분별(15분)"},
                        selected="Q",
                        inline=True,
                    ),
                    class_="seg",
                ),
                ui.span(
                    ui.span("", class_="dot"),
                    ui.span("측정일시", class_="label"),
                    ui.span(ui.output_text("toolbar_time"), class_="val"),
                    class_="time-chip",
                ),
                class_="toolbar",
            ),
        ),

        # ── 예측 지표 ──
        ui.card(
            ui.card_header("예측 지표"),
            ui.div(
                kpi("실시간 예측 사용량", "— kWh"),
                kpi("실시간 예측 요금", "— 원"),
                kpi("누적 예측 요금", "— 원"),
                kpi("작업 유형", "Light Load"),
                class_="kpi-row",
            ),
        ),

        # ── 시간대별 요금 예측 ──
        ui.card(
            ui.card_header("시간대별 요금 예측"),
            hourly_prediction(),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, ui.output_text("ts_caption")),
        ),

        # ── 누적 사용량 비교 ──
        ui.card(
            ui.card_header("누적 사용량 비교"),
            hourly_prediction(),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, ui.output_text("ts_caption2")),
        ),
    )


def predict_server(input, output, session):
    """툴바 시간 + 측정일시 표기(최소 로직)."""
    latest_ts = reactive.Value(None)
    running = reactive.Value(False)
    now_tick = reactive.Value(None)

    # 외부 실시간 수집에서: session.set_latest_timestamp("YYYY-MM-DD HH:MM:SS") 또는 datetime
    def set_latest_timestamp(ts):
        if ts is None:
            latest_ts.set(None); return
        if isinstance(ts, dt.datetime):
            latest_ts.set(ts); return
        if isinstance(ts, str):
            try:
                latest_ts.set(dt.datetime.fromisoformat(ts))
            except ValueError:
                latest_ts.set(dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            return
        latest_ts.set(dt.datetime.fromisoformat(str(ts)))
    session.set_latest_timestamp = set_latest_timestamp

    @reactive.event(input.btn_start)
    def _start(): running.set(True)

    @reactive.event(input.btn_stop)
    def _stop(): running.set(False)

    @reactive.event(input.btn_reset)
    def _reset():
        running.set(False); latest_ts.set(None); now_tick.set(None)

    @reactive.effect
    def _tick():
        if not running():
            return
        reactive.invalidate_later(1000)
        now_tick.set(dt.datetime.now())

    @output
    @render.text
    def toolbar_time():
        ts = latest_ts()
        if ts:
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        t = now_tick()
        return t.strftime("%Y-%m-%d %H:%M:%S") if t else "—"

    @output
    @render.text
    def ts_caption():
        ts = latest_ts()
        return f"마지막 업데이트: {ts:%Y-%m-%d %H:%M:%S}" if ts else "마지막 업데이트: —"

    @output
    @render.text
    def ts_caption2():
        ts = latest_ts()
        return f"마지막 업데이트: {ts:%Y-%m-%d %H:%M:%S}" if ts else "마지막 업데이트: —"
