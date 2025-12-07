# utils/scheduler.py

from __future__ import annotations
import numpy as np
import pandas as pd

NBO_TZ = "Africa/Nairobi"

# Rated power (kW) and required contiguous duration in HOURS (not slots)
DEVICE_SPECS_HOURS = {
    "Laundry_Machine_kW": dict(power=3.0,  dur_hours=4.0),   # 4 hours
    "Dryer_kW":            dict(power=3.0,  dur_hours=2.0),   # 2 hours
    "Dishwasher_kW":       dict(power=2.0,  dur_hours=1.5),   # 1.5 hours  (11:00–14:00)
    "Oven_kW":             dict(power=4.0,  dur_hours=6.0),   # 6 hours    (finish by 12:00)
    "Water_Heater_kW":     dict(power=5.0,  dur_hours=2.0),   # 2 hours    (finish by 09:00)
    "Ventilation_kW":      dict(power=1.5,  dur_hours=2.0),   # 2 hours
}

# Defining helper functions to make the work easier
def _to_nairobi(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True).dt.tz_convert(NBO_TZ)
    return out

def _tomorrow_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    now_nbo = pd.Timestamp.now(tz=NBO_TZ)
    day_start = now_nbo.normalize() + pd.Timedelta(days=1)
    return day_start, day_start + pd.Timedelta(days=1)

def _daylight_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    day_00, _day_24 = _tomorrow_bounds()
    return day_00 + pd.Timedelta(hours=6), day_00 + pd.Timedelta(hours=18)

def _window_sum(series: pd.Series, start: int, length: int) -> float:
    return float(series.iloc[start:start + length].sum())

def _infer_slot_minutes(index: pd.DatetimeIndex) -> int:
    """Infer the median step in minutes from the index; fallback to 30 if unknown."""
    if len(index) < 2:
        return 30
    deltas = pd.Series(index[1:] - index[:-1], dtype="timedelta64[ns]")
    median_delta = deltas.median()
    minutes = int(round(median_delta / pd.Timedelta(minutes=1)))
    return max(minutes, 1)  # avoid 0

# The actual Scheduler:

def schedule_loads(load_df: pd.DataFrame, solar_df: pd.DataFrame) -> pd.DataFrame:
    # Normalizing & keeping only tomorrow
    df = _to_nairobi(load_df, "timestamp")
    pv = _to_nairobi(solar_df, "timestamp")

    day_00, day_24 = _tomorrow_bounds()
    df = df[(df["timestamp"] >= day_00) & (df["timestamp"] < day_24)].copy()
    pv = pv[(pv["timestamp"] >= day_00) & (pv["timestamp"] < day_24)].copy()

    # Aligning on timestamp
    df = df.sort_values("timestamp").set_index("timestamp")
    pv = pv.sort_values("timestamp").set_index("timestamp")
    aligned = df.join(pv[["predicted_solar_production"]], how="inner")

    # These are the core columns
    if "base_load_kW" not in aligned.columns:
        aligned["base_load_kW"] = 0.0
    if "total_load_kW" not in aligned.columns:
        aligned["total_load_kW"] = aligned["base_load_kW"].astype(float)

    # Clear any existing device values for tomorrow (preventing any splits/carryover). This is bad 
    for dev in DEVICE_SPECS_HOURS.keys():
        aligned[dev] = 0.0
    # If Food_Warmers_kW existed historically, clear that too
    if "Food_Warmers_kW" in aligned.columns:
        aligned["Food_Warmers_kW"] = 0.0

    n = len(aligned)
    if n == 0:
        return aligned.reset_index()

    # Infer cadence and convert durations (hours -> slots) 
    slot_minutes = _infer_slot_minutes(aligned.index)
    slot_hours = slot_minutes / 60.0
    # Convert PV Wh/slot -> instantaneous kW per slot duration
    aligned["solar_kW"] = aligned["predicted_solar_production"] / (slot_hours * 1000.0)

    # Daylight mask (strict)
    six_am, six_pm = _daylight_bounds()
    noon       = day_00 + pd.Timedelta(hours=12)
    elev_start = day_00 + pd.Timedelta(hours=11)  # dishwasher earliest start
    elev_end   = day_00 + pd.Timedelta(hours=14)  # dishwasher latest finish
    morning_9  = day_00 + pd.Timedelta(hours=9)   # heater latest finish

    allowed_mask = (aligned.index >= six_am) & (aligned.index < six_pm)
    allowed_bool = np.asarray(allowed_mask, dtype=bool)

    # PV headroom after base load
    aligned["remaining_kW"] = (aligned["solar_kW"] - aligned["base_load_kW"]).clip(lower=0.0)
    remaining = aligned["remaining_kW"].copy()

    # Greedy: largest power first to reduce conflicts
    devices_sorted = sorted(DEVICE_SPECS_HOURS.keys(), key=lambda d: -DEVICE_SPECS_HOURS[d]["power"])

    def _slot_ok_with_window(dev: str, i: int, dur_slots: int) -> bool:
        """Check daylight + device-specific time window."""
        if not allowed_bool[i:i + dur_slots].all():
            return False

        ts_start = aligned.index[i]
        ts_end   = aligned.index[i + dur_slots - 1] + pd.Timedelta(minutes=slot_minutes)

        if dev == "Oven_kW":
            return ts_end <= noon
        elif dev == "Dishwasher_kW":
            return (ts_start >= elev_start) and (ts_end <= elev_end)
        elif dev == "Water_Heater_kW":
            return ts_end <= morning_9
        else:
            return True  # daylight only

    for dev in devices_sorted:
        power = float(DEVICE_SPECS_HOURS[dev]["power"])
        dur_hours = float(DEVICE_SPECS_HOURS[dev]["dur_hours"])
        dur_slots = max(int(round(dur_hours * 60.0 / slot_minutes)), 1)

        if dur_slots <= 0 or n < dur_slots:
            continue

        best_i, best_score = None, -1e18

        # Pass A: strict (device window + headroom for all slots) 
        for i in range(0, n - dur_slots + 1):
            if not _slot_ok_with_window(dev, i, dur_slots):
                continue
            if (remaining.iloc[i:i + dur_slots] - power < -1e-12).any():
                continue
            score = _window_sum(remaining, i, dur_slots)
            if score > best_score:
                best_score, best_i = score, i

        # Pass B: relaxed (device window only; ignore headroom) 
        if best_i is None:
            for i in range(0, n - dur_slots + 1):
                if not _slot_ok_with_window(dev, i, dur_slots):
                    continue
                score = _window_sum(aligned["solar_kW"], i, dur_slots)
                if score > best_score:
                    best_score, best_i = score, i

        # Final fallback: general daylight (ignore special window) 
        if best_i is None and dev in {"Oven_kW", "Dishwasher_kW", "Water_Heater_kW"}:
            # Try daylight + headroom
            for i in range(0, n - dur_slots + 1):
                if not allowed_bool[i:i + dur_slots].all():
                    continue
                if (remaining.iloc[i:i + dur_slots] - power < -1e-12).any():
                    continue
                score = _window_sum(remaining, i, dur_slots)
                if score > best_score:
                    best_score, best_i = score, i
            # If still none, daylight-only
            if best_i is None:
                for i in range(0, n - dur_slots + 1):
                    if not allowed_bool[i:i + dur_slots].all():
                        continue
                    score = _window_sum(aligned["solar_kW"], i, dur_slots)
                    if score > best_score:
                        best_score, best_i = score, i

        if best_i is None:
            # No valid window found — skip this device
            continue

        # Place single contiguous block
        idx_slice = aligned.index[best_i:best_i + dur_slots]
        aligned.loc[idx_slice, dev] = power
        aligned.loc[idx_slice, "total_load_kW"] += power

        # Consume headroom for subsequent devices (never below 0)
        remaining.iloc[best_i:best_i + dur_slots] = (
            remaining.iloc[best_i:best_i + dur_slots] - power
        ).clip(lower=0.0)

    # Return clean frame
    return aligned.drop(columns=["solar_kW", "remaining_kW"], errors="ignore").reset_index()
