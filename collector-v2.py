"""
학교 태양광 발전량 예측 수집기 (ECMWF 예보 + 선형 회귀)

1. Open-Meteo에서 학교 좌표의 ECMWF IFS 시간별 일사량·기온 예보를 받아 날짜별로 집계
   - 일사량: 하루 합계 (kWh/m²),  기온: 0~23시 평균 (°C)
2. model_coefficients.json 계수로 D1(오늘)·D2(내일)·D3(모레) 발전량 예측
   발전량 = intercept + day_sin·sin(2π·연중일수/365.25) + day_cos·cos(2π·연중일수/365.25)
            + kwh_m2·일사량 + kwh_m2_x_tmean·일사량·기온
   - D1 → 1일 전 모델, D2 → 2일 전 모델, D3 → 3일 전 모델
3. Firebase에 업로드 (로컬에는 아무것도 저장하지 않음)
   - /v2/latest               : 최신 3일 예측, 실행할 때마다 갱신 (웹사이트가 읽는 형식)
   - /v2/history/{날짜}        : 그날 처음 낸 당일(D1) 예측, 30일 지난 데이터는 자동 삭제
   - /v2/forecast_log/{날짜}   : 그날에 대해 1·2·3일 전에 처음 낸 예측 (정확도 평가용)
   (예전 모델 데이터는 /solar 에 그대로 남아 있음)

필요 파일 (같은 폴더): serviceAccountKey.json, model_coefficients.json
권장 실행 시각: 매일 03:30, 09:30, 15:30, 21:30 KST (ECMWF 6시간 갱신 직후)
결측 처리:
   - 일사량: 밤 시간 결측은 0, 낮 시간 결측은 앞뒤 값으로 보간 (낮 결측 4시간 초과 시 그날 제외)
   - 기온: 결측 시간은 앞뒤 값으로 보간 (결측 4시간 초과 시 그날 제외)
   - D1~D3 중 계산 가능한 날만 업로드, 없는 날은 latest에서 빠짐 (웹사이트는 있는 날만 표시)
종료 코드: 업로드함 0 (일부 날짜가 빠져도 0), 업로드 못 함 1 (이때 Firebase는 변경되지 않음)
데이터 출처: Weather data by Open-Meteo.com (CC BY 4.0), ECMWF IFS
"""

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import firebase_admin
from firebase_admin import credentials, db

# -------------------------------------------------------------
# 설정
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # cron 등에서 실행 위치와 무관하게 파일을 찾기 위함
KEY_PATH = BASE_DIR / "serviceAccountKey.json"
COEF_PATH = BASE_DIR / "model_coefficients.json"
FIREBASE_URL = "https://resistor-solar-default-rtdb.asia-southeast1.firebasedatabase.app/"
DB_ROOT = "v2"                 # 저장 위치 (기존 예전 모델 데이터는 /solar 에 그대로 남음)

LAT, LON = 37.461653631314825, 126.71344567084134
MODEL = "ecmwf_ifs"           # 학습에 쓴 예보 모델과 반드시 같아야 함
KST = ZoneInfo("Asia/Seoul")   # 서버 시간대와 관계없이 한국 날짜 기준
HISTORY_DAYS = 30

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
RETRIES = 3

NIGHT_HOURS = set(range(0, 6)) | set(range(20, 24))   # 일사량이 사실상 0인 시간 (KST)
MAX_FILLED_DAY_HOURS = 4                               # 낮 시간 결측이 이보다 많으면 그날은 제외
MAX_FILLED_TEMP_HOURS = 4                              # 기온 결측이 이보다 많으면 그날은 제외
COEF_KEYS = ("intercept", "day_sin", "day_cos", "kwh_m2", "kwh_m2_x_tmean")


# -------------------------------------------------------------
# 예측 모델
# -------------------------------------------------------------
def load_coefficients():
    with open(COEF_PATH, encoding="utf-8") as f:
        coef = json.load(f)
    if coef.get("model") != MODEL:
        raise ValueError(f"coefficient model '{coef.get('model')}' != collector model '{MODEL}'")
    for lead in ("1", "2", "3"):
        k = coef.get("leads", {}).get(lead, {})
        if not all(isinstance(k.get(name), (int, float)) for name in COEF_KEYS):
            raise ValueError(f"missing coefficients for lead {lead} (need {', '.join(COEF_KEYS)})")
    return coef


def predict_power(coef, lead, date_str, x_kwh_m2, tmean):
    """학습(Orange) 때와 같은 식. 음수는 0"""
    k = coef["leads"][str(lead)]
    doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday   # 1월 1일 = 1
    angle = 2 * math.pi * doy / 365.25
    y = (k["intercept"]
         + k["day_sin"] * math.sin(angle)
         + k["day_cos"] * math.cos(angle)
         + k["kwh_m2"] * x_kwh_m2
         + k["kwh_m2_x_tmean"] * x_kwh_m2 * tmean)
    return max(0.0, y)


# -------------------------------------------------------------
# ECMWF 일사량·기온 예보
# -------------------------------------------------------------
def fetch_daily_forecast():
    """
    날짜(YYYY-MM-DD) → {"x": (일사량 kWh/m², 보간 시간 수) 또는 None,
                         "tmean": (일평균기온 °C, 보간 시간 수) 또는 None}
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "models": MODEL,
        "hourly": "shortwave_radiation,temperature_2m",   # 일사량 W/m²(직전 1시간 평균), 기온 °C
        "timezone": "Asia/Seoul",
        "forecast_days": 4,
    }
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            res = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            res.raise_for_status()
            hourly = res.json()["hourly"]
            break
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Open-Meteo request failed after {RETRIES} tries: {last_err}")

    rad = defaultdict(dict)    # 날짜 → {시: 값}
    temp = defaultdict(dict)
    for t, r, c in zip(hourly["time"], hourly["shortwave_radiation"], hourly["temperature_2m"]):
        rad[t[:10]][int(t[11:13])] = r
        temp[t[:10]][int(t[11:13])] = c

    return {d: {"x": daily_sum(rad[d]), "tmean": daily_mean_temp(temp[d])} for d in rad}


def interpolate(vals):
    """None 칸을 앞뒤 값으로 선형 보간 (한쪽만 있으면 그 값). 값이 하나도 없으면 None"""
    known = [h for h, v in enumerate(vals) if v is not None]
    if not known:
        return None
    out = list(vals)
    for h, v in enumerate(vals):
        if v is not None:
            continue
        prev = max((k for k in known if k < h), default=None)
        nxt = min((k for k in known if k > h), default=None)
        if prev is None:
            out[h] = vals[nxt]
        elif nxt is None:
            out[h] = vals[prev]
        else:
            out[h] = vals[prev] + (vals[nxt] - vals[prev]) * (h - prev) / (nxt - prev)
    return out


def daily_mean_temp(hours):
    """시간별 기온 → (0~23시 평균 °C, 보간 시간 수). 결측이 많으면 None"""
    vals = [hours.get(h) for h in range(24)]
    missing = sum(v is None for v in vals)
    if missing > MAX_FILLED_TEMP_HOURS:
        return None
    vals = interpolate(vals)
    if vals is None:
        return None
    return round(sum(vals) / 24, 2), missing


def daily_sum(hours):
    """
    시간별 값 → (하루 합계 kWh/m², 보간한 낮 시간 수). 계산할 수 없으면 None.
    - 밤 시간(NIGHT_HOURS)의 빈 값은 0으로 채움
    - 낮 시간의 빈 값은 앞뒤 값으로 선형 보간
    - 낮 시간이 MAX_FILLED_DAY_HOURS보다 많이 비면 신뢰할 수 없으므로 None
    """
    vals = [hours.get(h) for h in range(24)]
    vals = [0.0 if v is None and h in NIGHT_HOURS else v for h, v in enumerate(vals)]

    missing = sum(v is None for v in vals)
    if missing > MAX_FILLED_DAY_HOURS:
        return None
    vals = interpolate(vals)
    if vals is None:
        return None
    return round(sum(vals) / 1000, 3), missing


# -------------------------------------------------------------
# Firebase
# -------------------------------------------------------------
def set_if_absent(ref, value):
    """값이 없을 때만 저장 (트랜잭션이라 동시에 실행돼도 안전)"""
    ref.transaction(lambda current: value if current is None else current)


def upload_to_firebase(data, now):
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(KEY_PATH)), {"databaseURL": FIREBASE_URL})

    # 없는 날(d1~d3)은 키 자체가 빠진 채로 저장됨 → 웹사이트는 있는 날만 표시
    db.reference(f"{DB_ROOT}/latest").set(data)

    # history·forecast_log는 "그날 처음 낸 예측"만 기록 (하루 여러 번 실행해도 덮어쓰지 않음)
    # → 저녁 실행분처럼 거의 실황에 가까운 값이 들어가 정확도가 부풀려지는 것을 막음
    ref_history = db.reference(f"{DB_ROOT}/history")
    if "d1" in data:
        set_if_absent(ref_history.child(data["d1"]["date"]), data["d1"])

    ref_log = db.reference(f"{DB_ROOT}/forecast_log")
    for key, lead in (("d1", 1), ("d2", 2), ("d3", 3)):
        if key not in data:
            continue
        set_if_absent(ref_log.child(data[key]["date"]).child(f"lead{lead}"), {
            "power_kwh": data[key]["power_kwh"],
            "kwh_m2": data[key]["kwh_m2"],
            "tmean": data[key]["tmean"],
            "issued_at": data["updated_at"],
        })

    cutoff = (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    old = [k for k in (ref_history.get(shallow=True) or {}) if k < cutoff]
    for k in old:
        ref_history.child(k).delete()
    return len(old)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    now = datetime.now(KST)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        coef = load_coefficients()
        daily = fetch_daily_forecast()

        payload = {"updated_at": stamp, "model": MODEL}
        lines, skipped = [], []
        for i, key in enumerate(("d1", "d2", "d3")):
            date_str = (now + timedelta(days=i)).strftime("%Y-%m-%d")
            day = daily.get(date_str, {})
            if not day.get("x") or not day.get("tmean"):
                what = "radiation" if not day.get("x") else "temperature"
                skipped.append(key.upper())
                lines.append(f"  {key.upper()} {date_str}  no {what} data, skipped")
                continue
            (x, filled_x), (tmean, filled_t) = day["x"], day["tmean"]
            payload[key] = {
                "date": date_str,
                "kwh_m2": x,
                "mj_m2": round(x * 3.6, 2),
                "tmean": tmean,
                "power_kwh": round(predict_power(coef, i + 1, date_str, x, tmean), 2),
            }
            mae = coef["leads"][str(i + 1)].get("mae_kwh")
            if isinstance(mae, (int, float)):
                payload[key]["mae_kwh"] = mae   # 그래프 오차 범위(±)용, 없으면 웹사이트 기본값 사용
            filled = filled_x + filled_t
            if filled:
                payload[key]["filled_hours"] = filled   # 보간한 시간 수 (참고용)
            note = f"  ({filled}h interpolated)" if filled else ""
            lines.append(f"  {key.upper()} {date_str}  {x:.2f} kWh/m2  {tmean:5.1f} C  ->  {payload[key]['power_kwh']:.1f} kWh{note}")

        if len(skipped) == 3:
            raise ValueError("no usable forecast for D1-D3, nothing uploaded")

        deleted = upload_to_firebase(payload, now)

        status = f"PARTIAL (skipped {', '.join(skipped)})" if skipped else "OK"
        print(f"[{stamp} KST] {status}  model={MODEL}")
        print("\n".join(lines))
        if deleted:
            print(f"  history: removed {deleted} entries older than {HISTORY_DAYS} days")
        return 0

    except Exception as e:
        print(f"[{stamp} KST] FAIL  {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
