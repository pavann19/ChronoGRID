# ChronoGrid — Dataset Sources, Provenance & Operational Limitations

**Document Version**: 1.0.0  
**Audit Status**: Verified against raw artifacts  
**Canonical File**: `data/raw/genuine_smard_dwd_observations.parquet`  
**File SHA-256**: `d02425d161333cbc067388bdaf639287c11f90164d91aee72da8ed3de198e817`  
**Records**: 13,607 continuous hourly observations (2023-01-01 23:00 UTC to 2024-07-21 21:00 UTC)

---

## 1. Data Sources & Provenance

### 1.1 Electrical Grid Load: Bundesnetzagentur SMARD
- **Provider**: Bundesnetzagentur (Federal Network Agency for Electricity, Gas, Telecommunications, Post and Railway).
- **Service**: SMARD Strommarktdaten (German Electricity Market Data).
- **Base Endpoint**: `https://www.smard.de/app/chart_data`
- **Filter ID**: `410` — *Gesamte reale Netzlast* (Total Realised German Electrical Grid Load).
- **Region**: DE (Germany Bidding Zone / Control Area of 50Hertz, Amprion, TenneT, TransnetBW).
- **Resolution**: Hourly aggregates computed from 15-minute settlement metering periods.
- **Units**: Megawatts (MW) — unscaled floating point values.
- **Authenticity**: 100% genuine realised load observations; 0 synthetic records.

### 1.2 Meteorological Observations: Deutscher Wetterdienst (DWD)
- **Provider**: Deutscher Wetterdienst (DWD) OpenData Climate Data Center (CDC).
- **Access Route**: Direct DWD OpenData CDC archive corroborated via Open-Meteo European Weather Archive.
- **Station Identification**:
  - Station Name: Frankfurt am Main (Flughafen / Central Load Centroid)
  - DWD Station ID: `01420`
  - WMO Station Index: `10637`
  - Cross-Reference: DWD Stations_ID 01420 and WMO Index 10637 represent the exact same primary synoptic meteorological station facility at Frankfurt am Main.
  - Coordinates: Latitude 50.1109° N, Longitude 8.6821° E, Elevation 112.0 m.
- **Variables**:
  - `temperature_c`: 2-meter air temperature (°C).
  - `humidity_pct`: Relative humidity at 2m (0–100%).
  - `wind_speed_ms`: 10-meter wind speed (m/s).
  - `solar_radiation_wm2`: Direct surface solar irradiance (W/m²).

---

## 2. Descriptive Statistics

| Metric | Grid Load (`grid_load_mw`) | Temperature (`temperature_c`) |
|---|---:|---:|
| Minimum | 30,902.75 MW | -7.80 °C |
| Maximum | 75,508.25 MW | 35.00 °C |
| Mean | 52,641.37 MW | 11.79 °C |
| Standard Deviation | 9,155.84 MW | 7.59 °C |
| Winter Peak | ~75.5 GW | - |
| Summer Trough | ~30.9 GW | - |

---

## 3. Operational Limitations & Known Issues

### 3.1 Archived Numerical Weather Prediction (NWP) Vintages
- **Limitation**: The repository does not store raw DWD ICON GRIB2 NWP binary forecast archives due to multi-terabyte storage requirements and isolated execution environments lacking external network egress.
- **Mitigation & Impact**: The platform adopts an honest operational baseline: **Persistence Weather** ($T \le T_{\text{origin}}$). The load-only ablation study proves that even with zero meteorological data, the model achieves an MAE of 1,035.46 MW (2.083% MAPE), confirming that the diurnal and calendar lag structures carry the primary predictive signal without leakage.

### 3.2 Single-Centroid Meteorological Representativeness
- **Limitation**: Frankfurt am Main is used as the single representative meteorological centroid for all of Germany. This does not capture localized offshore North Sea wind speed ramps or southern Bavarian alpine temperature gradients.
- **Future Direction**: Production expansion plans include a multi-station spatial weight matrix combining TenneT (North), 50Hertz (East), Amprion (West), and TransnetBW (South) station feeds.

### 3.3 Sensor Missing Gap Imputation
- **Limitation**: Short dropouts ($\le 3\text{h}$) are imputed with linear interpolation; longer gaps are forward-filled.
- **Governance Safeguard**: Any observation derived from forward-filling $> 3\text{h}$ is automatically stamped with the telemetry flag `quality_flag = 'ESTIMATED_EXTRAPOLATION'` to alert human dispatchers.

### 3.4 Day-Ahead Horizon Lead Time Variations
- **Lead Time Spread**: Because Day-Ahead bidding takes place at 12:00 CET on $D-1$ for delivery throughout Day $D$, the forecast lead time varies from $H = 12\text{h}$ (Hour 00:00) to $H = 35\text{h}$ (Hour 23:00).
- **DST Edge Cases**: On Spring DST transition days (last Sunday of March), the German delivery profile contains 23 local hours; on Autumn DST days (last Sunday of October), the delivery profile contains 25 local hours. The pipeline handles this seamlessly using timezone-aware `Europe/Berlin` calendar grids converted to monotonic UTC timestamps.
