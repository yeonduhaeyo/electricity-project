# modules/page_report.py
from shiny import ui, render
import datetime as dt
import calendar

# 2024-01 ~ 2024-11 월 선택지
MONTH_CHOICES = {f"2024-{m:02d}": f"2024년 {m}월" for m in range(1, 12)}
DEFAULT_MONTH = "2024-11"

def report_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="styles.css"),

        # ───────── 상단 리본(타이틀/월 선택/금액 배지) ─────────
        ui.div(
            ui.div(
                ui.div(
                    ui.span("전기요금 분석 고지서", class_="billx-title"),
                    ui.span(" · 생산설비 전력 사용 고지", class_="billx-sub"),
                    class_="billx-titlebox",
                ),
                # ⬇️ 날짜 입력 → 드롭다운(월 선택)으로 변경
                ui.div(
                    ui.input_select(
                        "rep_month", "청구월", choices=MONTH_CHOICES, selected=DEFAULT_MONTH
                    ),
                    class_="billx-month",
                ),
                ui.div(
                    ui.div("청구금액", class_="billx-amt-label"),
                    ui.div(ui.output_text("amt_due"), class_="billx-amt-value"),
                    class_="billx-amount-pill",
                ),
                class_="billx-ribbon",
            ),
            class_="billx",
        ),

        # ───────── 본문 그리드 (좌/우) ─────────
        ui.div(
            # 좌측
            ui.div(
                ui.div(
                    ui.div("청구 요약", class_="billx-panel-title"),
                    ui.div(
                        ui.div(ui.div("전력사용량(kWh)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_kwh"), class_="kpi-value"),
                               ui.div("당월 합계", class_="kpi-sub"), class_="kpi"),
                        ui.div(ui.div("전기요금(원)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_cost"), class_="kpi-value"),
                               ui.div("부가 포함", class_="kpi-sub"), class_="kpi"),
                        ui.div(ui.div("평균 단가(원/kWh)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_unit"), class_="kpi-value"),
                               ui.div("전기요금/사용량", class_="kpi-sub"), class_="kpi"),
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("전력 품질 요약", class_="billx-panel-title"),
                    ui.tags.ul({"class": "billx-list"},
                        ui.tags.li(ui.tags.b("지상역률(%)"), ui.span(ui.output_text("pf_lg"))),
                        ui.tags.li(ui.tags.b("진상역률(%)"), ui.span(ui.output_text("pf_ld"))),
                        ui.tags.li(ui.tags.b("지상무효전력량(kVarh)"), ui.span(ui.output_text("q_lg"))),
                        ui.tags.li(ui.tags.b("진상무효전력량(kVarh)"), ui.span(ui.output_text("q_ld"))),
                    ),
                    ui.div({"class": "billx-note"}, "※ 역률 저하는 요금 가산/설비 효율 저하로 이어질 수 있습니다."),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("탄소·환경", class_="billx-panel-title"),
                    ui.div(
                        ui.div(ui.div("탄소배출량(tCO₂)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_co2"), class_="kpi-value"),
                               ui.div("당월 추정", class_="kpi-sub"), class_="kpi"),
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),
                # ✅ 월 달력 · 일일 요금
                ui.div(
                    ui.div("월 달력 · 일일 요금", class_="billx-panel-title"),
                    ui.output_ui("month_calendar"),
                    ui.div({"class": "small-muted mt-2"}, "※ 날짜 밑 수치는 일일 요금 합계(원)입니다."),
                    class_="billx-panel",
                ),
                class_="billx-col",
            ),

            # 우측
            ui.div(
                ui.div(
                    ui.div("검침/사용 정보", class_="billx-panel-title"),
                    ui.tags.ul({"class": "billx-list"},
                        ui.tags.li(ui.tags.b("검침기간"), ui.span(ui.output_text("period"))),
                        ui.tags.li(ui.tags.b("데이터 건수"), ui.span(ui.output_text("rows"))),
                        ui.tags.li(ui.tags.b("주요 작업유형"), ui.span(ui.output_text("worktypes"))),
                    ),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("전월 대비 비교", class_="billx-panel-title"),
                    ui.div({"class": "placeholder"}, "여기에 사용량/요금 MoM 막대 그래프"),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("2024년 추이", class_="billx-panel-title"),
                    ui.div({"class": "placeholder"}, "여기에 12개월 추이(라인/컬럼)"),
                    class_="billx-panel",
                ),
                class_="billx-col",
            ),
            class_="billx-grid",
        ),

        # 하단 바
        ui.div(
            ui.div(
                ui.span(ui.output_text("issue_info"), class_="billx-issue"),
                ui.div(
                    ui.input_action_button("btn_export_pdf", "PDF 저장", class_="btn btn-primary"),
                    ui.input_action_button("btn_export_csv", "CSV 내보내기", class_="btn btn-outline-primary"),
                    class_="billx-actions",
                ),
                class_="billx-footer-inner",
            ),
            class_="billx-footer",
        ),
    )


def report_server(input, output, session):
    # ── KPI 더미 ──
    @output 
    @render.text
    def amt_due():   return "36,860원"
    @output 
    @render.text
    def kpi_kwh():  return "340"
    @output 
    @render.text
    def kpi_cost(): return "36,860"
    @output 
    @render.text
    def kpi_unit(): return "108.4"
    @output 
    @render.text
    def pf_lg():    return "98.6"
    @output 
    @render.text
    def pf_ld():    return "—"
    @output 
    @render.text
    def q_lg():     return "358"
    @output 
    @render.text
    def q_ld():     return "0"
    @output 
    @render.text
    def kpi_co2():  return "0.23"
    @output 
    @render.text
    def period():   return "2025-04-16 ~ 2025-05-15"
    @output 
    @render.text
    def rows():     return "2,880"
    @output 
    @render.text
    def worktypes():return "Light, Normal, Peak"
    @output 
    @render.text
    def issue_info(): return "발행일 2025-05-29 · 공장 전력 데이터 기반 자동 생성"

    # ── 유틸: "YYYY-MM" → (year, month), 해당월 1일/일수/오프셋 ──
    def _parse_year_month(ym: str) -> tuple[int, int]:
        y, m = ym.split("-")
        return int(y), int(m)

    def _first_meta_from_str(ym: str):
        y, m = _parse_year_month(ym)
        first = dt.date(y, m, 1)
        first_weekday, ndays = calendar.monthrange(y, m)  # Mon=0..Sun=6
        offset = (first_weekday + 1) % 7  # 우리 달력은 일요일 시작
        return first, ndays, offset

    # 실데이터 연결 훅: {date: 금액}
    def get_daily_costs(year: int, month: int) -> dict:
        # TODO: df.groupby(df["측정일시"].dt.date)["전기요금(원)"].sum().to_dict()
        return {}

    WEEK_LABELS = ["일","월","화","수","목","금","토"]

    @output
    @render.ui
    def month_calendar():
        # 선택된 "YYYY-MM"
        ym = input.rep_month() or DEFAULT_MONTH
        first, ndays, offset = _first_meta_from_str(ym)
        costs = get_daily_costs(first.year, first.month)

        header = ui.tags.div({"class": "cal-weekdays"}, *[ui.tags.div(x) for x in WEEK_LABELS])

        cells = []
        for _ in range(offset):
            cells.append(ui.tags.div({"class": "cal-cell empty"}))

        today = dt.date.today()
        for d in range(1, ndays + 1):
            date_obj = dt.date(first.year, first.month, d)
            col = (offset + (d - 1)) % 7
            cls = "cal-cell" + (" sun" if col == 0 else " sat" if col == 6 else "")
            if date_obj == today:
                cls += " today"
            val = costs.get(date_obj)
            text = f"{val:,.0f}원" if isinstance(val, (int, float)) else "—"
            cells.append(ui.tags.div({"class": cls},
                                     ui.tags.div(str(d), {"class": "date"}),
                                     ui.tags.div(text, {"class": "cost"})))

        grid = ui.tags.div({"class": "cal-grid"}, *cells)
        return ui.tags.div({"class": "billx-cal"}, header, grid)
