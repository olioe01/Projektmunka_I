"""A 2. DL modell (Embedding + BiLSTM) diái: slides/dl_2_bilstm.pptx

Futtatás a repó gyökeréből:  .venv/bin/python slides/make_bilstm_slides.py
Adatok: results/bilstm.json, a dl_2_bilstm.ipynb kísérleti táblái és slides/bilstm_curves.json
(BiLSTM(32), seed 42, 30 epoch EarlyStopping nélkül – Dropout nélkül és 0,3-mal).
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "dl_2_bilstm.pptx"
result = json.loads((ROOT / "results" / "bilstm.json").read_text(encoding="utf-8"))
baseline = json.loads((ROOT / "results" / "baseline.json").read_text(encoding="utf-8"))
curves = json.loads((ROOT / "slides" / "bilstm_curves.json").read_text(encoding="utf-8"))

DARK = RGBColor(0x12, 0x40, 0x3C)    # mély zöldeskék – címek
TEAL = RGBColor(0x1B, 0x99, 0x8B)    # fő szín
CORAL = RGBColor(0xE4, 0x57, 0x2E)   # kiemelés (a végső modell)
GRAY = RGBColor(0x5C, 0x6B, 0x73)    # másodlagos szöveg
TINT = RGBColor(0xEA, 0xF4, 0xF3)    # kártyák háttere
MUTED = RGBColor(0xA9, 0xC9, 0xC5)   # nem kiemelt oszlopok
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"


def hu(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def text(slide, x, y, w, h, runs, size=14, color=DARK, bold=False, align=PP_ALIGN.LEFT,
         font=BODY, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str vagy bekezdések listája; egy bekezdés: str vagy [(szöveg, {opciók}), ...]."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, opt in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", False)
            f.color.rgb = opt.get("color", color)
        sa = [o["space_after"] for _, o in ([] if isinstance(para, str) else para) if "space_after" in o]
        if sa:
            p.space_after = Pt(sa[0])
    return tb


def box(slide, x, y, w, h, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    if name:
        s.name = name
    return s


def header(slide, title, subtitle):
    text(slide, 0.6, 0.4, 12.1, 0.7, title, size=36, bold=True, font=HEAD, name="Cím")
    text(slide, 0.6, 1.12, 12.1, 0.35, subtitle, size=14, color=GRAY, name="Alcím")


def style_chart(chart):
    chart.font.name = BODY
    chart.font.size = Pt(12)
    chart.font.color.rgb = GRAY


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]

# ------------------------------------------------------------------ 1. dia
s1 = prs.slides.add_slide(blank)
header(s1, "BiLSTM: architektúra és kísérletek",
       "2. DL modell · András · MedPlain: szakmai (1) vagy laikus (0) orvosi mondat")

text(s1, 0.6, 1.8, 4.7, 0.35, "Végső modell", size=20, bold=True, color=TEAL)
layers_ = [
    ("Embedding", "11 894 szó × 100 dim, mask_zero=True", "1 189 400"),
    ("Bidirectional(LSTM(32))", "előre + hátra olvas → 64 dim vektor", "34 048"),
    ("Dropout(0,3)", "regularizáció", "0"),
    ("Dense(1, sigmoid)", "P(szakmai mondat)", "65"),
]
y0, bh, gap = 2.3, 0.78, 0.3
for i, (name, detail, params) in enumerate(layers_):
    y = y0 + i * (bh + gap)
    chosen = i == 1
    b = box(s1, 0.6, y, 4.7, bh, fill=TEAL if chosen else TINT, name=f"Réteg {i + 1}")
    fg = WHITE if chosen else DARK
    sub = WHITE if chosen else GRAY
    text(s1, 0.8, y + 0.1, 3.2, bh - 0.2,
         [[(name, {"bold": True, "size": 15, "color": fg})], [(detail, {"size": 12, "color": sub})]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s1, 3.9, y + 0.1, 1.2, bh - 0.2,
         [[(params, {"bold": True, "size": 13, "color": fg})], [("paraméter", {"size": 10, "color": sub})]],
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(layers_) - 1:
        box(s1, 2.8, y + bh + 0.05, 0.3, gap - 0.1, fill=MUTED, shape=MSO_SHAPE.DOWN_ARROW,
                name=f"Nyíl {i + 1}")
text(s1, 0.6, 6.6, 4.7, 0.5,
     f"Összesen {result['params']:,} paraméter".replace(",", " ") + ", ennek 97%-a az Embeddingben.",
     size=12, color=GRAY)

text(s1, 5.9, 1.8, 6.9, 0.35, "Kísérletek: val accuracy, 3 seed átlaga", size=20, bold=True, color=TEAL)
exps = [  # dl_2_bilstm.ipynb, 1–3. kísérlet
    ("LSTM(64), egyirányú", 0.7778),
    ("BiLSTM(64)", 0.7833),
    ("BiLSTM(32) – végső", 0.7932),
    ("BiLSTM(128)", 0.7674),
    ("BiLSTM(32), Dropout 0", 0.7734),
    ("BiLSTM(32), Dropout 0,5", 0.7907),
    ("+ LSTM dropout 0,2", 0.7921),
    ("+ recurrent_dropout 0,2", 0.7858),
]
cd = CategoryChartData()
cd.categories = [e[0] for e in exps]
cd.add_series("val accuracy", [e[1] for e in exps])
gf = s1.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(5.8), Inches(2.2), Inches(7.0), Inches(3.75), cd)
gf.name = "Kísérletek diagram"
ch = gf.chart
style_chart(ch)
ch.has_legend = False
ch.has_title = False
plot = ch.plots[0]
plot.gap_width = 45
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = '0.000', False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(12)
dl.font.color.rgb = DARK
ser = plot.series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = MUTED
for i, (lab, _) in enumerate(exps):
    if "végső" in lab:
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = CORAL
ca = ch.category_axis
ca.reverse_order = True
ca.format.line.color.rgb = MUTED
ca.tick_labels.font.size = Pt(12)
ca.tick_labels.font.color.rgb = DARK
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0.70, 0.81
va.visible = False
va.has_major_gridlines = False

facts = [("+0,5 pp", "kétirányú vs. egyirányú"), ("32 > 64 > 128", "kevesebb egység jobb"),
         ("+2 pp", "Dropout 0,3 vs. 0"), ("47×", "lassabb a recurrent_dropout")]
fw, fg_ = 1.62, 0.16
for i, (big, small) in enumerate(facts):
    x = 5.9 + i * (fw + fg_)
    box(s1, x, 6.1, fw, 1.0, name=f"Tény {i + 1}")
    text(s1, x + 0.1, 6.17, fw - 0.2, 0.86,
         [[(big, {"bold": True, "size": 18, "color": CORAL if i == 3 else TEAL})],
          [(small, {"size": 11, "color": GRAY})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

s1.notes_slide.notes_text_frame.text = (
    "A modell: Embedding, utána kétirányú LSTM, amely a mondatot előre és hátrafelé is végigolvassa, "
    "majd Dropout és egy szigmoid kimenet. A mask_zero miatt a padding-nullákat az LSTM átugorja.\n"
    "Minden kísérlet a validációs halmazon futott, 3 seed átlagával. A kétirányú változat csak fél "
    "százalékponttal jobb az egyirányúnál. Meglepő módon a 32 egység volt a legjobb: a nagyobb LSTM "
    "csak gyorsabban túltanul. A Dropout réteg 2 százalékpontot hozott. Az LSTM saját recurrent_dropoutja "
    "nem javított, viszont 47-szer lassabb, mert kikapcsolja a GPU-s cuDNN-gyorsítást."
)

# ------------------------------------------------------------------ 2. dia
s2 = prs.slides.add_slide(blank)
header(s2, "Eredmény: a BiLSTM nem veri a baseline-t",
       "Teszthalmaz: 3 048 mondat, 150 cikk – csak a végső modellen, egyszer kiértékelve")

stats = [(hu(result["accuracy"]), "teszt accuracy", f"baseline: {hu(baseline['accuracy'])}"),
         (hu(result["f1"]), "teszt F1", f"baseline: {hu(baseline['f1'])}")]
for i, (big, lab, ref) in enumerate(stats):
    x = 0.6 + i * 2.85
    box(s2, x, 1.8, 2.65, 1.6, name=f"Mutató {i + 1}")
    text(s2, x + 0.2, 1.88, 2.25, 1.45,
         [[(big, {"bold": True, "size": 40, "color": CORAL, "font": HEAD})],
          [(lab, {"size": 14, "color": DARK, "bold": True})],
          [(ref, {"size": 12, "color": GRAY})]], anchor=MSO_ANCHOR.MIDDLE)
text(s2, 0.6, 3.55, 5.5, 0.35,
     [[("3 seed átlaga: ", {"size": 13, "color": GRAY}),
       ("0,776 ± 0,005", {"size": 13, "bold": True}),
       ("  (baseline: 0,796 ± 0,010)", {"size": 13, "color": GRAY})]])

text(s2, 0.6, 4.1, 5.5, 0.35, "Konfúziós mátrix (test)", size=20, bold=True, color=TEAL)
(tn, fp), (fn, tp) = result["confusion_matrix"]
rows = [["", "Jósolt: laikus", "Jósolt: szakmai"],
        ["Valós: laikus", f"{tn:,}".replace(",", " "), str(fp)],
        ["Valós: szakmai", str(fn), f"{tp:,}".replace(",", " ")]]
tbl = s2.shapes.add_table(3, 3, Inches(0.6), Inches(4.55), Inches(5.5), Inches(1.35)).table
tbl.columns[0].width = Inches(1.7)
tbl.columns[1].width = tbl.columns[2].width = Inches(1.9)
for r in range(3):
    for c in range(3):
        cell = tbl.cell(r, c)
        cell.text = rows[r][c]
        cell.margin_left = cell.margin_right = Inches(0.1)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        f = p.runs[0].font if p.runs else p.font
        f.name, f.size = BODY, Pt(14)
        diag = r > 0 and c > 0 and r == c
        err = r > 0 and c > 0 and r != c
        f.bold = r == 0 or c == 0 or diag or err
        f.color.rgb = WHITE if diag else (CORAL if err else DARK)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL if diag else (WHITE if r == 0 or c == 0 else TINT)
text(s2, 0.6, 6.1, 5.5, 0.8,
     f"A szakmai mondatok {fn / (fn + tp):.0%}-át ({fn} db) laikusnak jósolja; "
     f"a laikusoknál csak {fp / (tn + fp):.0%} a hiba.", size=13, color=GRAY)

text(s2, 6.8, 1.8, 6.0, 0.35, "Tanulási görbe: 30 epoch, EarlyStopping nélkül", size=20, bold=True,
     color=TEAL)
cd2 = CategoryChartData()
n = len(curves["Dropout 0,3"]["loss"])
cd2.categories = [str(i) for i in range(1, n + 1)]
cd2.add_series("train loss (Dropout 0,3)", curves["Dropout 0,3"]["loss"])
cd2.add_series("val loss (Dropout 0,3)", curves["Dropout 0,3"]["val_loss"])
cd2.add_series("val loss (Dropout nélkül)", curves["Dropout nélkül"]["val_loss"])
gf2 = s2.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(6.7), Inches(2.2), Inches(6.1), Inches(3.6), cd2)
gf2.name = "Tanulási görbe"
ch2 = gf2.chart
style_chart(ch2)
ch2.has_title = False
ch2.has_legend = True
ch2.legend.position = XL_LEGEND_POSITION.TOP
ch2.legend.include_in_layout = False
ch2.legend.font.size = Pt(11)
for ser, color, dash in zip(ch2.plots[0].series, [GRAY, CORAL, TEAL],
                            [MSO_LINE_DASH_STYLE.DASH, None, None]):
    ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.NONE
    ser.format.line.color.rgb = color
    ser.format.line.width = Pt(2.25)
    if dash:
        ser.format.line.dash_style = dash
va2 = ch2.value_axis
va2.has_major_gridlines = True
va2.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE5, 0xE4)
va2.format.line.fill.background()
va2.tick_labels.number_format, va2.tick_labels.number_format_is_linked = "0.0", False
va2.has_title = True
va2.axis_title.text_frame.text = "loss"
va2.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ca2 = ch2.category_axis
ca2.has_title = True
ca2.axis_title.text_frame.text = "epoch"
ca2.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ca2.format.line.color.rgb = MUTED
# minden 5. epoch felirata
skip = ca2._element.makeelement(qn("c:tickLblSkip"), {"val": "5"})
nml = ca2._element.find(qn("c:noMultiLvlLbl"))
nml.addprevious(skip) if nml is not None else ca2._element.append(skip)

best = min(range(n), key=lambda i: curves["Dropout 0,3"]["val_loss"][i]) + 1
text(s2, 6.8, 6.0, 6.0, 1.1,
     [[("A val loss minimuma az ", {"size": 14}), (f"{best}. epochnál", {"size": 14, "bold": True}),
       (" van, utána a modell túltanul – az EarlyStopping ezt a súlyt tartja meg.",
        {"size": 14, "space_after": 6})],
      [("A szakmaiságot főleg a szókincs jelzi, nem a szórend: a szekvenciális modell nem ad pluszt.",
        {"size": 14})]], color=DARK)

s2.notes_slide.notes_text_frame.text = (
    f"A teszthalmazon a BiLSTM {hu(result['accuracy'])} pontosságot ért el, a baseline "
    f"{hu(baseline['accuracy'])}-et, vagyis körülbelül 2–3 százalékponttal rosszabb, pedig lassabb és "
    "bonyolultabb. Három seed átlagában is ugyanez a sorrend, tehát nem véletlen.\n"
    "A konfúziós mátrix szerint főleg a szakmai mondatokat téveszti el: azok 29%-át laikusnak jósolja.\n"
    "A tanulási görbén látszik, hogy a validációs loss már az első epoch után nőni kezd, a train loss "
    "pedig tovább csökken: erős túltanulás. A görbe a végső beállítás (BiLSTM 32, seed 42) újrafuttatása "
    "30 epochon EarlyStopping nélkül.\n"
    "Értelmezés: a szakmai és a laikus mondatot itt elsősorban a szakszavak jelenléte különbözteti meg, "
    "nem a szavak sorrendje, ezért az egyszerű átlagoló baseline elég, a BiLSTM nem tud többet kihozni."
)

prs.save(OUT)
print("Mentve:", OUT)
