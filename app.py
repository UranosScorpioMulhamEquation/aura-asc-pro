from datetime import datetime, date, time, timedelta
import os
import io
import base64
from typing import Dict, List, Tuple, Any, Optional
import geonamescache
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import swisseph as swe
from timezonefinder import TimezoneFinder
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 0. تهيئة مسار ملفات الـ Ephemeris والإعدادات
# ==========================================
EPHE_PATH = "./ephe"
if not os.path.exists(EPHE_PATH):
    os.makedirs(EPHE_PATH)

swe.set_ephe_path(EPHE_PATH)

# ضبط النظام الفلكي ليكون تروبيكال (استوائي) حصرياً
swe.set_sid_mode(0, 0, 0)

# ==========================================
# 1. جداول علم الحرف والأبراج (التروبيكال)
# ==========================================
ABJAD_DICT = {
    'ا': 1, 'أ': 1, 'إ': 1, 'آ': 1, 'ء': 1,
    'ب': 2,
    'ج': 3,
    'د': 4,
    'ه': 5, 'ة': 5,
    'و': 6,
    'ز': 7,
    'ح': 8,
    'ط': 9,
    'ي': 10, 'ى': 10,
    'ك': 20,
    'ل': 30,
    'م': 40,
    'ن': 50,
    'س': 60,
    'ع': 70,
    'ف': 80,
    'ص': 90,
    'ق': 100,
    'ر': 200,
    'ش': 300,
    'ت': 400,
    'ث': 500,
    'خ': 600,
    'ذ': 700,
    'ض': 800,
    'ظ': 900,
    'غ': 1000
}

DETAILED_ZODIAC_DEGREES = {
    1: {'name': 'الحمل (Aries)', 'ruler': 'المريخ', 'index': 0, 'start_deg': 0},
    2: {'name': 'الثور (Taurus)', 'ruler': 'الزهرة', 'index': 1, 'start_deg': 30},
    3: {'name': 'الجوزاء (Gemini)', 'ruler': 'عطارد', 'index': 2, 'start_deg': 60},
    4: {'name': 'السرطان (Cancer)', 'ruler': 'القمر', 'index': 3, 'start_deg': 90},
    5: {'name': 'الأسد (Leo)', 'ruler': 'الشمس', 'index': 4, 'start_deg': 120},
    6: {'name': 'العذراء (Virgo)', 'ruler': 'عطارد', 'index': 5, 'start_deg': 150},
    7: {'name': 'الميزان (Libra)', 'ruler': 'الزهرة', 'index': 6, 'start_deg': 180},
    8: {'name': 'العقرب (Scorpio)', 'ruler': 'المريخ', 'index': 7, 'start_deg': 210},
    9: {'name': 'القوس (Sagittarius)', 'ruler': 'المشتري', 'index': 8, 'start_deg': 240},
    10: {'name': 'الجدي (Capricorn)', 'ruler': 'زحل', 'index': 9, 'start_deg': 270},
    11: {'name': 'الدلو (Aquarius)', 'ruler': 'زحل', 'index': 10, 'start_deg': 300},
    12: {'name': 'الحوت (Pisces)', 'ruler': 'المشتري', 'index': 11, 'start_deg': 330}
}

PLANET_RULES = {
    'المريخ': ['الحمل', 'العقرب'],
    'الزهرة': ['الثور', 'الميزان'],
    'عطارد': ['الجوزاء', 'العذراء'],
    'القمر': ['السرطان'],
    'الشمس': ['الأسد'],
    'المشتري': ['القوس', 'الحوت'],
    'زحل': ['الجدي', 'الدلو']
}

zodiacs = [
    'الحمل (Aries)', 'الثور (Taurus)', 'الجوزاء (Gemini)', 'السرطان (Cancer)',
    'الأسد (Leo)', 'العذراء (Virgo)', 'الميزان (Libra)', 'العقرب (Scorpio)',
    'القوس (Sagittarius)', 'الجدي (Capricorn)', 'الدلو (Aquarius)', 'الحوت (Pisces)'
]

def format_degree(deg: float) -> str:
    """تنسيق درجة الزاوية إلى درجة ودقيقة واسم البرج"""
    sign_idx = int(deg // 30) % 12
    rem = deg % 30
    d_int = int(rem)
    m_int = int((rem - d_int) * 60)
    return f"{d_int}° {m_int:02d}' - {zodiacs[sign_idx]}"

def calculate_name_value(name_part: str, is_composite_first_box: bool = False) -> int:
    """حساب جمل الاسم مع تطبيق قاعدة 400 للتاء المربوطة/الهاء"""
    total = 0
    name_part = name_part.strip()
    if not name_part:
        return 0
    for i, char in enumerate(name_part):
        if char in ['ة', 'ه']:
            if is_composite_first_box and i == len(name_part) - 1:
                total += 400
            else:
                total += 5
        elif char in ABJAD_DICT:
            total += ABJAD_DICT[char]
    return total

def get_utc_julian_day(date_obj: date, time_obj: time, tz_str: str) -> float:
    """حساب اليوم الجولياني بالتوقيت العالمي المنسق UTC"""
    local_tz = pytz.timezone(tz_str)
    dt = datetime.combine(date_obj, time_obj)
    local_dt = local_tz.localize(dt)
    utc_dt = local_dt.astimezone(pytz.utc)
    return swe.julday(
        utc_dt.year,
        utc_dt.month,
        utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0,
    )

def calculate_tropical_positions(jd: float) -> Dict[str, int]:
    """حساب مواقع الكواكب الاستوائية عبر Swiss Ephemeris"""
    planet_constants = {
        'الشمس': swe.SUN,
        'القمر': swe.MOON,
        'عطارد': swe.MERCURY,
        'الزهرة': swe.VENUS,
        'المريخ': swe.MARS,
        'المشتري': swe.JUPITER,
        'زحل': swe.SATURN
    }
    
    planets_tropical = {}
    for p_name, p_id in planet_constants.items():
        res, _ = swe.calc_ut(jd, p_id)
        longitude = res[0]
        z_idx = int(longitude // 30) + 1
        planets_tropical[p_name] = z_idx
        
    res_nodes, _ = swe.calc_ut(jd, swe.MEAN_NODE)
    rahu_lon = res_nodes[0]
    planets_tropical['راهو'] = int(rahu_lon // 30) + 1
    planets_tropical['كيتو'] = int((rahu_lon + 180) % 360 // 30) + 1

    return planets_tropical

def get_esoteric_zodiac_anchored(
    fn_p1: str, fn_p2: str, mn_p1: str, mn_p2: str, 
    is_fn_comp: bool, is_mn_comp: bool, birth_date_obj: date, 
    asc_deg: float
) -> Tuple[str, int, str, int]:
    """استخراج طالع طاقة الهالة الروحية (الأقرب لطالع طاقة الجسد Asc) بناءً على مجموع الجمل وتاريخ الميلاد"""
    val_fn1 = calculate_name_value(fn_p1, is_composite_first_box=is_fn_comp)
    val_fn2 = calculate_name_value(fn_p2, is_composite_first_box=False) if is_fn_comp else 0
    
    val_mn1 = calculate_name_value(mn_p1, is_composite_first_box=is_mn_comp)
    val_mn2 = calculate_name_value(mn_p2, is_composite_first_box=False) if is_mn_comp else 0
    
    total_name_val = val_fn1 + val_fn2 + val_mn1 + val_mn2
    date_str = birth_date_obj.strftime('%d%m%Y')
    digits_sum = sum(int(char) for char in date_str if char.isdigit())
    total_sum = total_name_val + digits_sum
    
    zodiac_remainder_idx = total_sum % 12
    if zodiac_remainder_idx == 0:
        zodiac_remainder_idx = 12
        
    quotient = total_sum // 12
    quotient_idx = (quotient % 12) + 1
    
    candidate_indices = list({zodiac_remainder_idx, quotient_idx})
    
    best_zodiac_idx = candidate_indices[0]
    min_distance_asc = 360.0
    
    for c_idx in candidate_indices:
        z_data = DETAILED_ZODIAC_DEGREES[c_idx]
        z_center_deg = z_data['start_deg'] + 15.0
        distance = abs(asc_deg - z_center_deg)
        if distance > 180:
            distance = 360 - distance
        if distance < min_distance_asc:
            min_distance_asc = distance
            best_zodiac_idx = c_idx

    selected_zodiac_data = DETAILED_ZODIAC_DEGREES[best_zodiac_idx]
    zodiac_name = selected_zodiac_data['name']
    ruler = selected_zodiac_data['ruler']
    
    return zodiac_name, selected_zodiac_data['index'], ruler, total_sum

# ==========================================
# 2. خدمات البيانات الجغرافية والزمنية
# ==========================================
@st.cache_data
def load_geo_data() -> Tuple[Any, Dict[str, str], List[str], Dict[str, List[Dict[str, Any]]]]:
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()

    country_map = {c["name"]: code for code, c in countries.items()}
    sorted_countries = sorted(list(country_map.keys()))

    cities_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for c_code in country_map.values():
        c_cities = [city for city in cities.values() if city["countrycode"] == c_code]
        c_cities_sorted = sorted(c_cities, key=lambda x: x.get("population", 0), reverse=True)
        cities_by_country[c_code] = c_cities_sorted

    return gc, country_map, sorted_countries, cities_by_country

gc, country_map, country_list, cities_by_country = load_geo_data()

@st.cache_resource
def get_tz_finder() -> TimezoneFinder:
    return TimezoneFinder()

tf = get_tz_finder()

# ==========================================
# 3. رسم الخريطة الدائرية الاستوائية المتوافقة
# ==========================================
def draw_circular_chart(asc_degree: float, planets_in_signs: Dict[str, int], ep_deg: float):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.axis('off')

    outer = plt.Circle((0, 0), 1.3, color='#f59e0b', fill=False, lw=2.5)
    inner = plt.Circle((0, 0), 0.9, color='#3b82f6', fill=False, lw=1.5)
    ax.add_patch(outer)
    ax.add_patch(inner)

    zodiac_names_ar = [
        'الحمل', 'الثور', 'الجوزاء', 'السرطان',
        'الأسد', 'العذراء', 'الميزان', 'العقرب',
        'القوس', 'الجدي', 'الدلو', 'الحوت'
    ]

    # رسم الأبراج الاستوائية الثابتة (0 إلى 360 درجة بدقة تامة)
    for i in range(12):
        angle_deg = 180 - (i * 30) - 15  # منتصف البرج
        angle_rad = np.radians(angle_deg)
        
        line_angle_deg = 180 - (i * 30)
        line_rad = np.radians(line_angle_deg)
        x1 = 0.9 * np.cos(line_rad)
        y1 = 0.9 * np.sin(line_rad)
        x2 = 1.3 * np.cos(line_rad)
        y2 = 1.3 * np.sin(line_rad)
        ax.plot([x1, x2], [y1, y2], color='#3b82f6', lw=1.5)
        
        tx = 1.12 * np.cos(angle_rad)
        ty = 1.12 * np.sin(angle_rad)
        ax.text(tx, ty, f"{i+1}\n{zodiac_names_ar[i]}", color='#f3f4f6', fontsize=8, fontweight='bold', ha='center', va='center')

    # مؤشر طالع طاقة الجسد (Ascendant) باستخدام الطول المطلق الصحيح
    asc_angle_deg = 180 - asc_degree
    asc_rad = np.radians(asc_angle_deg)
    ax.plot([0, 1.45 * np.cos(asc_rad)], [0, 1.45 * np.sin(asc_rad)], color='#10b981', lw=3, linestyle='-', label='طالع طاقة الجسد')
    ax.text(1.58 * np.cos(asc_rad), 1.58 * np.sin(asc_rad), f"طالع الجسد: {asc_degree:.1f}°", color='#10b981', fontsize=8, fontweight='bold', ha='center', va='center')

    # مؤشر ساعة الميلاد (East Point - EP) باستخدام الطول المطلق الصحيح
    ep_angle_deg = 180 - ep_deg
    ep_rad = np.radians(ep_angle_deg)
    ax.plot([0, 1.35 * np.cos(ep_rad)], [0, 1.35 * np.sin(ep_rad)], color='#ec4899', lw=2, linestyle='--', label='ساعة الميلاد')
    ax.text(1.48 * np.cos(ep_rad), 1.48 * np.sin(ep_rad), f"ساعة الميلاد: {ep_deg:.1f}°", color='#ec4899', fontsize=8, fontweight='bold', ha='center', va='center')

    sign_to_planets = {i: [] for i in range(1, 13)}
    for p_name, p_z_idx in planets_in_signs.items():
        sign_to_planets[p_z_idx].append(p_name)
        
    for i in range(12):
        p_list = sign_to_planets[i+1]
        if p_list:
            text_angle = 180 - (i * 30) - 15
            text_rad = np.radians(text_angle)
            px = 0.65 * np.cos(text_rad)
            py = 0.65 * np.sin(text_rad)
            ax.text(px, py, "\n".join(p_list), color='#38bdf8', fontsize=8, ha='center', va='center')

    plt.title("الخريطة الفلكية الاستوائية والدائرية (Tropical Wheel & Esoteric Chart)", color='#f59e0b', fontsize=12, pad=15, fontweight='bold')
    return fig

# ==========================================
# 4. واجهة المستخدم (Streamlit App)
# ==========================================
st.set_page_config(page_title="رادار الطوالع الفلكية", layout="wide")

st.title("🌟 النظام المتقدم لاستخراج الطوالع المختلفة")
st.markdown("تطوير استرو رادار")
st.markdown("---")

st.sidebar.header('📋 بيانات المستفيد باللغة العربية حصرا')

fn_type = st.sidebar.radio("نوع الاسم الأول:", ["مفرد", "مركب"], key="fn_type", horizontal=True)
if fn_type == "مفرد":
    fn_p1 = st.sidebar.text_input("الاسم الأول المفرد:", value="محمد", key="fn1")
    fn_p2 = ""
else:
    fn_p1 = st.sidebar.text_input("الشطر الأول من الاسم الأول:", value="عبد", key="fn1_c")
    fn_p2 = st.sidebar.text_input("الشطر الثاني من الاسم الأول:", value="الرحمن", key="fn2_c")

mn_type = st.sidebar.radio("نوع اسم الأم:", ["مفرد", "مركب"], key="mn_type", horizontal=True)
if mn_type == "مفرد":
    mn_p1 = st.sidebar.text_input("اسم الأم المفرد:", value="فاطمة", key="mn1")
    mn_p2 = ""
else:
    mn_p1 = st.sidebar.text_input("الشطر الأول من اسم الأم:", value="فاطمة", key="mn1_c")
    mn_p2 = st.sidebar.text_input("الشطر الثاني من اسم الأم:", value="الزهراء", key="mn2_c")

birth_date = st.sidebar.date_input(
    "تاريخ الميلاد:", 
    value=date(2020, 1, 1),
    min_value=date(1950, 1, 1), 
    max_value=date.today()
)

selected_country_name = st.sidebar.selectbox('دولة الميلاد', country_list)
c_code = country_map.get(selected_country_name)

country_cities = cities_by_country.get(c_code, [])
if not country_cities:
    country_cities = [{'name': 'المدينة المركزية', 'latitude': 33.5138, 'longitude': 36.2765, 'population': 1000000}]

city_options = [f"{c['name']} (السكان: {c.get('population', 0):,})" for c in country_cities]
selected_city_str = st.sidebar.selectbox('مدينة الميلاد (مرتبة بالكثافة السكانية)', city_options)

selected_city_obj = country_cities[city_options.index(selected_city_str)]
city_lat = selected_city_obj['latitude']
city_lon = selected_city_obj['longitude']

auto_tz = tf.timezone_at(lng=city_lon, lat=city_lat)
if not auto_tz:
    auto_tz = 'UTC'

timezone_str = st.sidebar.text_input('المنطقة الزمنية (تلقائي أو تعديل يدوي)', value=auto_tz)

st.sidebar.info(
    f'📍 الإحداثيات المختارة:\nخط العرض: {city_lat}° | خط الطول: {city_lon}°\n🌍 المنطقة الزمنية: {timezone_str}'
)

time_intervals = [
    'من 10 ص - 1 م',
    'من 1 م - 4 م',
    'من 4 م - 7 م',
    'من 7 م - 10 م',
    'من 10 م الى 1 فجرا',
    'من 1 فجرا - 4 فجرا',
    'من 4 فجرا - 7 ص',
    'من 7 ص الى 10 ص',
]
selected_interval = st.sidebar.selectbox(
    'الفترة الزمنية المتوقعة (فترات 3 ساعات)', time_intervals
)

# ==========================================
# 5. محرك الحساب والتحليل الفلكي عند الضغط
# ==========================================
if st.button('🚀 احسب الطالع المرجح وساعة الميلاد'):
    if not fn_p1 or not mn_p1:
        st.error("الرجاء إدخال الحقول المطلوبة بشكل صحيح.")
    else:
        interval_map = {
            'من 10 ص - 1 م': (10.0, 13.0),
            'من 1 م - 4 م': (13.0, 16.0),
            'من 4 م - 7 م': (16.0, 19.0),
            'من 7 م - 10 م': (19.0, 22.0),
            'من 10 م الى 1 فجرا': (22.0, 25.0),
            'من 1 فجرا - 4 فجرا': (1.0, 4.0),
            'من 4 فجرا - 7 ص': (4.0, 7.0),
            'من 7 ص الى 10 ص': (7.0, 10.0),
        }

        start_h, end_h = interval_map[selected_interval]
        mid_h = (start_h + end_h) / 2.0
        h_local = mid_h % 24
        current_date = birth_date
        if mid_h >= 24:
            current_date = birth_date + timedelta(days=1)
        elif mid_h < 0:
            current_date = birth_date - timedelta(days=1)

        hour_int = int(h_local)
        minute_int = int((h_local - hour_int) * 60)
        time_obj = time(hour=hour_int, minute=minute_int, second=0)

        try:
            jd = get_utc_julian_day(current_date, time_obj, timezone_str)
            cusps, ascmc = swe.houses(jd, city_lat, city_lon, b'P')
            
            asc_deg = ascmc[0]  # طالع طاقة الجسد (الطول الفلكي المطلق 0-360)
            ep_deg = ascmc[4]   # ساعة الميلاد EP (الطول الفلكي المطلق 0-360)

            asc_str = format_degree(asc_deg)
            ep_str = format_degree(ep_deg)
            time_str = f"{int(h_local%24):02d}:{minute_int:02d}"

            is_fn_comp = (fn_type == "مركب")
            is_mn_comp = (mn_type == "مركب")

            esoteric_zodiac_name, zodiac_idx_num, ruler, total_sum = get_esoteric_zodiac_anchored(
                fn_p1, fn_p2, mn_p1, mn_p2, is_fn_comp, is_mn_comp, birth_date, asc_deg
            )

            ruled_signs = PLANET_RULES.get(ruler, [esoteric_zodiac_name])
            ruled_signs_str = " أو ".join(ruled_signs)

            full_fn = f"{fn_p1} {fn_p2}".strip()
            full_mn = f"{mn_p1} {mn_p2}".strip()
            full_display_name = f"{full_fn} بن/بنت {full_mn}"

            st.success(f"✅ تم حساب التحليل بنجاح للمستفيد: **{full_display_name}**")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 🔮 النتائج الحرفية والفلكية الاستوائية")
                st.markdown(f"**الاسم الكامل:** {full_display_name}")
                st.markdown(f"**الوقت المحسوب (منتصف الفترة):** {time_str}")
                st.markdown(f"**مجموع الجمل الكبير + تاريخ الميلاد:** {total_sum}")
                st.markdown(f"**طالع طاقة الجسد:** {asc_str}")
                st.markdown(f"**ساعة الميلاد:** {ep_str}")
                st.markdown(f"**طالع طاقة الهالة الروحية:** {esoteric_zodiac_name}")
                st.markdown(f"**حاكم طالع الهالة الروحية:** {ruler} (يحكم بالعادة: **{ruled_signs_str}**)")
                
            with col2:
                st.markdown(f"### 🌌 الخريطة الفلكية الاستوائية الدائرية (Tropical Wheel)")
                planets_tropical = calculate_tropical_positions(jd)
                chart_fig = draw_circular_chart(asc_deg, planets_tropical, ep_deg)
                st.pyplot(chart_fig)

            # تحويل الرسم البياني إلى Base64 لتضمينه مباشرة داخل تقرير HTML
            buf = io.BytesIO()
            chart_fig.savefig(buf, format='png', bbox_inches='tight', facecolor=chart_fig.get_facecolor(), edgecolor='none')
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            chart_img_html = f"data:image/png;base64,{chart_base64}"

            # تقرير HTML القابل للطباعة مع تضمين صورة الخريطة الفلكية المتوافقة
            html_report = f"""
            <!DOCTYPE html>
            <html lang="ar" dir="rtl">
            <head>
                <meta charset="UTF-8">
                <title>التقرير الفلكي الاستوائي - {full_display_name}</title>
                <style>
                    body {{ font-family: 'Amiri', Tahoma, Arial, sans-serif; background: #f9f9f9; color: #333; direction: rtl; padding: 20px; }}
                    .container {{ max-width: 800px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    h1 {{ text-align: center; color: #b45309; border-bottom: 2px solid #f59e0b; padding-bottom: 10px; }}
                    .section {{ margin-top: 20px; padding: 15px; background: #f8fafc; border-radius: 8px; }}
                    .highlight {{ font-weight: bold; color: #1e293b; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                    th, td {{ border: 1px solid #cbd5e0; padding: 10px; text-align: right; }}
                    th {{ background-color: #3b82f6; color: #fff; }}
                    .chart-container {{ text-align: center; margin-top: 25px; }}
                    .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; border: 2px solid #f59e0b; }}
                    .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #64748b; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>التقرير الشامل للطوالع</h1>
                    
                    <div class="section">
                        <h3>بيانات المستفيد والنتائج المتقدمة (Tropical System)</h3>
                        <table>
                            <tr>
                                <th>المؤشر الفلكي / الحرفي</th>
                                <th>القيمة والنتيجة</th>
                            </tr>
                            <tr>
                                <td>الاسم واسم الأم</td>
                                <td class="highlight">{full_display_name}</td>
                            </tr>
                            <tr>
                                <td>تاريخ ومكان الميلاد والوقت</td>
                                <td>{current_date.strftime('%Y-%m-%d')} ({selected_interval} -> {time_str}) | {selected_country_name} - {selected_city_str}</td>
                            </tr>
                            <tr>
                                <td>طالع طاقة الجسد</td>
                                <td class="highlight">{asc_str}</td>
                            </tr>
                            <tr>
                                <td>ساعة الميلاد</td>
                                <td class="highlight">{ep_str}</td>
                            </tr>
                            <tr>
                                <td>طالع طاقة الهالة الروحية</td>
                                <td class="highlight">{esoteric_zodiac_name}</td>
                            </tr>
                            <tr>
                                <td>حاكم طالع الهالة الروحية</td>
                                <td class="highlight">{ruler} (يحكم: {ruled_signs_str})</td>
                            </tr>
                        </table>
                    </div>

                    <div class="section chart-container">
                        <h3>الخريطة الفلكية الاستوائية والدائرية</h3>
                        <img src="{chart_img_html}" alt="الخريطة الفلكية الدائرية">
                    </div>
                    
                    <div class="footer">
                        <p>تم إصدار هذا التقرير آلياً عبر محرك التنجيم الاستوائي المتقدم (Tropical & Esoteric Engine).</p>
                    </div>
                </div>
            </body>
            </html>
            """

            st.markdown("---")
            st.subheader("📄 تحميل التقرير النهائي")
            st.download_button(
                label="📥 تحميل تقرير HTML الاستوائي المتكامل والجاهز للطباعة مع الخريطة",
                data=html_report,
                file_name=f"Tropical_Astro_Report_{fn_p1.replace(' ', '_')}.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"حدث خطأ أثناء إجراء الحسابات الفلكية: {str(e)}")