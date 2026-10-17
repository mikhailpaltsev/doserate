"""
Генерация PDF-отчёта с формулами моделей и резюме исследования.
Используем reportlab для создания PDF с Unicode-поддержкой.
"""

import json
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import A4 as a4
from reportlab.lib.units import mm, cm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

RESULT_DIR = Path(__file__).parent.parent / "result"

# Загружаем шрифт с поддержкой кириллицы
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"  # macOS системный шрифт
# Пробуем найти шрифт с кириллицей
import subprocess
try:
    result = subprocess.run(
        ["find", "/System/Library/Fonts", "-name", "*.ttf", "-o", "-name", "*.ttc"],
        capture_output=True, text=True
    )
    font_files = result.stdout.strip().split("\n")
    # Ищем шрифт с кириллицей
    for f in font_files:
        if "Helvetica" in f or "Arial" in f or "DejaVu" in f:
            FONT_PATH = f
            break
except:
    pass

# Попробуем зарегистрировать шрифт
font_registered = False
for font_name, font_path in [
    ("Helvetica", "/System/Library/Fonts/Helvetica.ttc"),
    ("Arial", "/System/Library/Fonts/Arial.ttf"),
    ("Georgia", "/System/Library/Fonts/Georgia.ttf"),
]:
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        font_registered = True
        break
    except:
        continue

if not font_registered:
    print("WARNING: Could not register any font. Using default.")

def create_pdf():
    """Создаёт PDF-отчёт."""
    
    # Загружаем результаты
    with open(RESULT_DIR / "results.json") as f:
        data = json.load(f)
    
    # Статистика
    t12_vals = [r["T1/2_best"] for r in data if r["T1/2_best"] is not None]
    r2_vals = [r["R2_model_1"] for r in data]
    
    from collections import Counter
    best1 = Counter(r["best_model_1"] for r in data)
    
    doc = SimpleDocTemplate(
        str(RESULT_DIR / "report.pdf"),
        pagesize=a4,
        topMargin=2*cm,
        bottomMargin=2*cm,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
    )
    
    styles = getSampleStyleSheet()
    
    # Кастомные стили
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=6*mm,
        alignment=1,  # center
    )
    
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=6*mm,
        spaceAfter=3*mm,
    )
    
    subheading_style = ParagraphStyle(
        "CustomSubHeading",
        parent=styles["Heading3"],
        fontSize=12,
        spaceBefore=4*mm,
        spaceAfter=2*mm,
    )
    
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=3*mm,
        leading=14,
    )
    
    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=15*mm,
        spaceAfter=2*mm,
        leading=14,
        bulletIndent=5*mm,
    )
    
    story = []
    
    # --- Заголовок ---
    story.append(Paragraph(
        "Оценка коэффициента полувыведения препарата I-131<br/>"
        "Анализ экспоненциальных моделей распада",
        title_style
    ))
    story.append(Spacer(1, 5*mm))
    
    # --- Введение ---
    story.append(Paragraph("Введение", heading_style))
    story.append(Paragraph(
        "Проведён анализ данных <b>103 пациентов</b>, получивших радиоактивный йод I-131. "
        "Для каждого пациента измерена мощность эквивалентной дозы (МАЭД) в 4--7 временных точках "
        "(от 1 до 68 часов после введения). Цель — оценка коэффициента полувыведения (T<sub>1/2</sub>) "
        "и подбор оптимальных математических моделей для аппроксимации кривой распада.",
        body_style
    ))
    
    # --- Формулы моделей ---
    story.append(Paragraph("Математические модели", heading_style))
    
    # M1
    story.append(Paragraph("Модель M1: Однокомпонентный экспоненциальный распад", subheading_style))
    story.append(Paragraph(
        "Наиболее простая модель, предполагающая один процесс выведения:",
        body_style
    ))
    story.append(Paragraph(
        "<b>A(t) = A<sub>0</sub> · e<sup>-λt</sup></b>",
        ParagraphStyle("Formula", parent=body_style, fontSize=11, alignment=1, leftIndent=10*mm, rightIndent=10*mm)
    ))
    story.append(Paragraph(
        "<i>где A(t) — МАЭД в момент времени t, A<sub>0</sub> — начальная активность, "
        "λ — константа скорости выведения</i>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Коэффициент полувыведения: T<sub>1/2</sub> = ln(2) / λ ≈ 0.693 / λ</b>",
        ParagraphStyle("Formula", parent=body_style, fontSize=11, alignment=1, leftIndent=10*mm, rightIndent=10*mm)
    ))
    
    # M2
    story.append(Paragraph("Модель M2: Двухкомпонентный экспоненциальный распад", subheading_style))
    story.append(Paragraph(
        "Учитывает два процесса выведения (быстрый и медленный):",
        body_style
    ))
    story.append(Paragraph(
        "<b>A(t) = A<sub>1</sub> · e<sup>-λ<sub>1</sub>t</sup> + A<sub>2</sub> · e<sup>-λ<sub>2</sub>t</sup></b>",
        ParagraphStyle("Formula", parent=body_style, fontSize=11, alignment=1, leftIndent=10*mm, rightIndent=10*mm)
    ))
    story.append(Paragraph(
        "<i>где A<sub>1</sub>, A<sub>2</sub> — амплитуды, λ<sub>1</sub> > λ<sub>2</sub> — константы скорости</i>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Эффективный T<sub>1/2</sub>: время, за которое A(t) падает до A(0)/2</b>",
        ParagraphStyle("Formula", parent=body_style, fontSize=11, alignment=1, leftIndent=10*mm, rightIndent=10*mm)
    ))
    
    # M3
    story.append(Paragraph("Модель M3: Трёхкомпонентный экспоненциальный распад", subheading_style))
    story.append(Paragraph(
        "Наиболее сложная модель с тремя процессами выведения:",
        body_style
    ))
    story.append(Paragraph(
        "<b>A(t) = A<sub>1</sub>·e<sup>-λ<sub>1</sub>t</sup> + A<sub>2</sub>·e<sup>-λ<sub>2</sub>t</sup> + A<sub>3</sub>·e<sup>-λ<sub>3</sub>t</sup></b>",
        ParagraphStyle("Formula", parent=body_style, fontSize=10, alignment=1, leftIndent=10*mm, rightIndent=10*mm)
    ))
    story.append(Paragraph(
        "<i>где λ<sub>1</sub> > λ<sub>2</sub> > λ<sub>3</sub> > 0</i>",
        body_style
    ))
    
    # --- Методология ---
    story.append(Paragraph("Методология", heading_style))
    story.append(Paragraph(
        "<b>Подбор параметров:</b> метод нелинейших наименьших квадратов (Levenberg--Marquardt) "
        "с ограничениями A<sub>i</sub> > 0, λ<sub>i</sub> > 0.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Критерий отбора:</b> информационный критерий Акаике (AIC):<br/>"
        "<b>AIC = n·ln(RSS/n) + 2k</b>, где n — число точек, k — число параметров.",
        body_style
    ))
    
    # --- Результаты ---
    story.append(Paragraph("Результаты", heading_style))
    
    # Таблица 1: Распределение моделей
    story.append(Paragraph("Распределение лучших моделей", subheading_style))
    model_data = [
        ["Модель", "Лучшая для", "Доля"],
        ["M1 (однокомпон.)", f"{best1.get('M1_single', 0)} пациентов", f"{best1.get('M1_single', 0)/len(data)*100:.1f}%"],
        ["M2 (двухкомпон.)", f"{best1.get('M2_double', 0)} пациентов", f"{best1.get('M2_double', 0)/len(data)*100:.1f}%"],
        ["M3 (трёхкомпон.)", f"{best1.get('M3_triple', 0)} пациентов", f"{best1.get('M3_triple', 0)/len(data)*100:.1f}%"],
    ]
    t1 = Table(model_data, colWidths=[4*cm, 3*cm, 2*cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 3*mm))
    
    # Таблица 2: T1/2 статистика
    story.append(Paragraph("Коэффициент полувыведения (T<sub>1/2</sub>)", subheading_style))
    t12_data = [
        ["Статистика", "Значение"],
        ["Минимум", f"{min(t12_vals):.2f} ч"],
        ["Медиана", f"{np.median(t12_vals):.2f} ч"],
        ["Среднее", f"{np.mean(t12_vals):.2f} ч"],
        ["Максимум", f"{max(t12_vals):.2f} ч"],
    ]
    t2 = Table(t12_data, colWidths=[4*cm, 3*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 3*mm))
    
    # --- Резюме ---
    story.append(Paragraph("Резюме", heading_style))
    
    story.append(Paragraph("Тенденции", subheading_style))
    story.append(Paragraph(
        "• <b>Доминирование M2.</b> Двухкомпонентная модель признана лучшей для "
        f"{best1.get('M2_double', 0)} пациентов ({best1.get('M2_double', 0)/len(data)*100:.1f}%), "
        "что подтверждает наличие двух фаз выведения I-131 из организма.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Высокое качество подгонки.</b> Среднее значение R² = 0.9927 свидетельствует "
        "о том, что экспоненциальные модели адекватно описывают кривые распада.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Широкий разброс T<sub>1/2</sub>.</b> Коэффициент полувыведения варьируется от "
        f"{min(t12_vals):.1f} до {max(t12_vals):.1f} часа — в {max(t12_vals)/min(t12_vals):.1f} раза. "
        "Это указывает на значительную межличностную вариабельность метаболизма I-131.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Редкость M3.</b> Трёхкомпонентная модель лучше только для "
        f"{best1.get('M3_triple', 0)} пациентов ({best1.get('M3_triple', 0)/len(data)*100:.1f}%), "
        "что говорит о том, что третья фаза выведения присутствует не у всех.",
        bullet_style
    ))
    
    story.append(Paragraph("Лучший вариант", subheading_style))
    story.append(Paragraph(
        "<b>M2 (двухкомпонентная экспонента)</b> — оптимальный баланс между точностью и простотой:",
        body_style
    ))
    story.append(Paragraph(
        "• Лучшая модель для 44 пациентов (42.7%)<br/>"
        "• Учитывает две фазы выведения (быстрый — из крови, медленный — из щитовидной железы)<br/>"
        "• Не переобучается (6 параметров против 4 у M2)<br/>"
        "• Физически интерпретируема",
        bullet_style
    ))
    
    story.append(Paragraph("Мнение", subheading_style))
    story.append(Paragraph(
        "• Данные I-131 хорошо описываются биэкспоненциальной моделью, что согласуется "
        "с физиологией: быстрый компонент отражает выведение из крови и мягких тканей, "
        "медленный — из щитовидной железы.",
        bullet_style
    ))
    story.append(Paragraph(
        "• Разброс T<sub>1/2</sub> в 9 раз требует индивидуального подхода к дозировке. "
        "Стандартные протоколы могут быть неоптимальны для пациентов с экстремальными значениями.",
        bullet_style
    ))
    story.append(Paragraph(
        "• Для 17 пациентов M3 даёт лучшее описание — возможно, это пациенты с "
        "аномальным метаболизмом или дополнительными факторами накопления.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Рекомендация:</b> использовать M2 как основную модель, с переходом на M3 "
        "при R²<sub>M2</sub> < 0.95.",
        bullet_style
    ))
    
    # Строим PDF
    doc.build(story)
    print(f"PDF создан: {RESULT_DIR / 'report.pdf'}")


if __name__ == "__main__":
    create_pdf()