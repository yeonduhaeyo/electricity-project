# modules/page_predict.py
from shiny import ui, render, reactive
import datetime as dt
import pandas as pd

from utils.ui_components import kpi
from viz.plot_placeholders import hourly_prediction
from utils.time_streamer import RealTimeStreamer

# ✅ shared에서 test.csv를 로드해 둔 DataFrame 사용
from shared import streaming_df

def predict_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="styles.css"),

        # ── 상단 툴바 ──
        ui.div(
            ui.div(
                # 좌측: 컨트롤 버튼
                ui.div(
                    ui.input_action_button("btn_start", "▶ 시작", class_="btn-control btn-start"),
                    ui.input_action_button("btn_stop",  "⏸ 멈춤", class_="btn-control btn-stop"),
                    ui.input_action_button("btn_reset", "↻ 리셋", class_="btn-control btn-reset"),
                    class_="control-btns",
                ),
                # 우측: 측정일시 칩
                ui.div(
                    ui.span("", class_="status-dot"),
                    ui.div(
                        ui.span("측정일시", class_="time-label"),
                        ui.span(ui.output_text("toolbar_time"), class_="time-value"),
                        class_="time-info",
                    ),
                    class_="time-chip-modern",
                ),
                class_="toolbar-modern",
            ),
            class_="toolbar-container",
        ),

        # ── 예측 지표 (작업 유형 실시간 갱신) ──
        ui.card(
            ui.card_header("예측 지표"),
            ui.div(
                kpi("실시간 예측 사용량", "— kWh"),
                kpi("실시간 예측 요금",   "— 원"),
                kpi("누적 예측 요금",     "— 원"),
                kpi("작업 유형",          ui.output_text("worktype_text")),
                class_="kpi-row",
            ),
        ),

        # ── 시간대별 요금 예측 (자리표시자) ──
        ui.card(
            ui.card_header("시간대별 요금 예측"),
            hourly_prediction(),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, ui.output_text("ts_caption")),
        ),

        # ── 누적 사용량 비교 (자리표시자) ──
        ui.card(
            ui.card_header("누적 사용량 비교"),
            hourly_prediction(),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, ui.output_text("ts_caption2")),
        ),
    )


def predict_server(input, output, session):
        
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "측정일시" not in out.columns:
            out["측정일시"] = pd.NaT
        if "작업유형" not in out.columns:
            out["작업유형"] = "—"
        out = out[["측정일시", "작업유형"]]
        out["측정일시"] = pd.to_datetime(out["측정일시"], errors="coerce")
        out = out.dropna(subset=["측정일시"]).sort_values("측정일시").reset_index(drop=True)
        return out

    src_df = _prepare(streaming_df if isinstance(streaming_df, pd.DataFrame) else pd.DataFrame())
    streamer = RealTimeStreamer(src_df)

    # ---- 상태 관리 ----
    running        = reactive.Value(False)
    latest_ts      = reactive.Value(None)   # datetime | None
    worktype_state = reactive.Value("—")

    # ---- 컨트롤 버튼 ----
    @reactive.effect
    @reactive.event(input.btn_start)
    def _start():
        if len(src_df) == 0:
            print("⚠️ 스트리밍할 데이터가 없습니다.")
            return
        running.set(True)
        print("▶️ 스트리밍 시작")

    @reactive.effect
    @reactive.event(input.btn_stop)
    def _stop():
        running.set(False)
        print("⏸️ 스트리밍 멈춤")

    @reactive.effect
    @reactive.event(input.btn_reset)
    def _reset():
        running.set(False)
        streamer.reset_stream()
        latest_ts.set(None)
        worktype_state.set("—")
        print("🔄 스트리밍 리셋")

    # ---- 메인 루프: 3초마다 1행씩 가져오기 ----
    @reactive.effect
    def _streaming_loop():
        reactive.invalidate_later(3)  # 3초마다 실행
        
        if not running():
            return

        # 다음 배치 가져오기
        batch = streamer.get_next_batch(1)
        
        if batch is None or batch.empty:
            running.set(False)  # 데이터 끝
            print("✅ 스트리밍 완료")
            return

        # 최신 데이터 추출
        row = batch.iloc[-1]
        
        # 측정일시 업데이트
        ts = pd.to_datetime(row["측정일시"])
        if pd.notna(ts):
            latest_ts.set(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        
        # 작업유형 업데이트
        wt = str(row["작업유형"]) if pd.notna(row["작업유형"]) else "—"
        worktype_state.set(wt)
        
        print(f"📊 업데이트: {latest_ts()} | {wt} | 진행률: {streamer.progress():.1f}%")

    # ---- 출력 렌더링 ----
    @output
    @render.text
    def toolbar_time():
        ts = latest_ts()
        return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"

    @output
    @render.text
    def worktype_text():
        return worktype_state() or "—"

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