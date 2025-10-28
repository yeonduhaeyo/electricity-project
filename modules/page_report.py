from shiny import ui, render
import datetime as dt
from utils.ui_components import section_caption, placeholder
from viz.plot_placeholders import usage_rate_trend, weekday_hour_heatmap

def report_ui():
    return ui.layout_column_wrap(
        ui.card(
            ui.card_header("필터"),
            ui.layout_columns(
                ui.input_date_range(
                    "rep_range",
                    "분석 기간",
                    start=dt.date.today() - dt.timedelta(days=30),
                    end=dt.date.today(),
                ),
                ui.input_radio_buttons(
                    "rep_gran",
                    "집계 단위",
                    {"D": "일별", "W": "주별", "M": "월별"},
                    selected="D",
                    inline=True,
                ),
                ui.input_slider("topn", "Top N (고가 시간대)", min=5, max=24, value=10),
                col_widths=[6, 3, 3],
            ),
            section_caption("집계 단위를 바꾸면 트렌드의 리샘플링 기준이 변합니다. (현재는 자리 표시자)"),
        ),
        ui.layout_columns(
            ui.card(ui.card_header("사용량/요금 추이"), usage_rate_trend()),
            ui.card(ui.card_header("요일×시간대 요금 히트맵"), weekday_hour_heatmap()),
        ),
        ui.card(
            ui.card_header("Top-N 고가 시간대"),
            placeholder("여기에 Top-N 테이블이 표시됩니다.", height=260),
            ui.hr({"class": "soft"}),
            ui.div({"class": "small-muted"}, "CSV 내보내기 버튼은 이후 단계에서 추가 가능합니다."),
        ),
        width=1,  # <- deprecation 경고 방지(키워드 인자, 마지막에)
    )

def report_server(input, output, session):
    # 현재는 전부 플레이스홀더 UI만 렌더
    pass
