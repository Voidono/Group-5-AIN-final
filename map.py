# ========================================================
#  ORIGINAL MAP WITH NEW LOGIC: TYPHOON PREDICTION + BOAT ESCAPE
#  Geopandas for detailed map, LSTM from ipynb, animation from map.py
# ========================================================

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from math import radians, cos, sin, sqrt, atan2
import warnings
warnings.filterwarnings("ignore")

# ================== LOAD TYPHOON DATA (IBTRACS) ==================
DATA_FILE = "ibtracs.WP.list.v04r00.csv"
if not os.path.exists(DATA_FILE):
    print("Please download IBTrACS from https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ and place in folder.")
    exit()

print("Loading typhoon data...")
df = pd.read_csv(DATA_FILE, low_memory=False)
df = df[df['BASIN'] == 'WP']
df = df[['LAT', 'LON', 'USA_WIND']].dropna()
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
df = df.dropna()

# LIMIT DATA FOR FASTER TRAINING - only use recent data
df = df.tail(5000)  # Use only last 5000 records
print(f"Using {len(df)} data points for training")

# ================== TRAIN LSTM MODEL FOR TYPHOON PREDICTION ==================
def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

print("Creating sequences...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

X, y = create_sequences(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], 3))
print(f"Training sequences: {X.shape[0]}")

print("Building LSTM model...")
model = Sequential([
    LSTM(64, activation='relu', input_shape=(7, 3), return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
print("Training model (this may take 1-2 minutes)...")
model.fit(X, y, epochs=5, batch_size=128, verbose=1)  # Reduced to 5 epochs, larger batch
print("Model training complete!")

# ================== PREDICT TYPHOON PATH ==================
def predict_path(history_lat, history_lon, history_wind, steps=20):
    seq = np.array(list(zip(history_lat, history_lon, history_wind)))
    seq_scaled = scaler.transform(seq[-7:]).reshape(1, 7, 3)
    
    pred = []
    current = seq_scaled.copy()
    for _ in range(steps):
        next_step = model.predict(current, verbose=0)
        pred.append(next_step[0])
        current = np.roll(current, -1, axis=1)
        current[0, -1] = next_step
    
    pred = scaler.inverse_transform(np.array(pred))
    return pred[:,0], pred[:,1]

history_lat = [11.96, 12.5, 13.2, 14.0, 14.5, 14.76, 15.1]
history_lon = [127.98, 126.5, 125.8, 124.9, 124.2, 123.03, 122.0]
history_wind = [35, 38, 42, 48, 52, 56, 58]

print("Predicting typhoon path...")
pred_lat, pred_lon = predict_path(history_lat, history_lon, history_wind)
full_lat = history_lat + pred_lat.tolist()
full_lon = history_lon + pred_lon.tolist()
print(f"Predicted {len(pred_lat)} future positions")

# ================== HAVERSINE FUNCTION (MOVED UP) ==================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# ================== BOAT ESCAPE SIMULATION ==================
BOAT_START = (15.50, 115.00)
DA_NANG = (16.07, 108.22)
boat_path = [BOAT_START]
boat_lat, boat_lon = BOAT_START
distance = haversine(boat_lat, boat_lon, *DA_NANG)

print("Calculating boat escape route...")
while distance > 10:
    best_score = -float('inf')
    best_pos = None
    for angle in range(0, 360, 45):
        dy = cos(radians(angle)) * (18 * 3 / 111)
        dx = sin(radians(angle)) * (18 * 3 / 111)
        new_lat = boat_lat + dy
        new_lon = boat_lon + dx
        d_port = haversine(new_lat, new_lon, *DA_NANG)
        storm_idx = min(len(full_lat)-1, 7 + len(boat_path))
        d_storm = haversine(new_lat, new_lon, full_lat[storm_idx], full_lon[storm_idx])
        score = d_storm - d_port
        if score > best_score:
            best_score = score
            best_pos = (new_lat, new_lon)
    if best_pos:
        boat_lat, boat_lon = best_pos
        boat_path.append((boat_lat, boat_lon))
        distance = haversine(boat_lat, boat_lon, *DA_NANG)
print(f"Boat path calculated: {len(boat_path)} waypoints")

# ================== ORIGINAL MAP WITH GEOPANDAS ==================
world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
vietnam = world[world['ADMIN'] == 'Vietnam']
other = world[world['ADMIN'] != 'Vietnam']

fig, ax = plt.subplots(figsize=(14, 16))
ax.set_facecolor('#a6cce3')

other.plot(ax=ax, color='#f0f0f0', edgecolor='white')
vietnam.plot(ax=ax, color='#FFD700', edgecolor='#d32f2f', linewidth=1.2)

# Ports and Islands (original)
danh_sach_cang = [
    {"name": "Hải Phòng", "lon": 106.7, "lat": 20.85},
    {"name": "Đà Nẵng", "lon": 108.22, "lat": 16.1},
    {"name": "Quy Nhơn", "lon": 109.22, "lat": 13.76},
    {"name": "Cam Ranh", "lon": 109.15, "lat": 11.9},
    {"name": "Vũng Tàu", "lon": 107.08, "lat": 10.35}
]
for cang in danh_sach_cang:
    plt.text(cang["lon"], cang["lat"], "⚓", fontsize=12, color='#00008B', ha='center', va='center', zorder=7)
    plt.text(cang["lon"]+0.2, cang["lat"], cang["name"], fontsize=9, weight='bold', color='black', 
             path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=8)

plt.text(111.2, 16.5, "★", fontsize=20, color='red', ha='center', va='center', zorder=6)
plt.text(111.6, 16.5, "Hoàng Sa (VN)", color='red', weight='bold', fontsize=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=7)
plt.text(114.5, 10.0, "★", fontsize=20, color='red', ha='center', va='center', zorder=6)
plt.text(114.9, 10.0, "Trường Sa (VN)", color='red', weight='bold', fontsize=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=7)

ax.set_xlim(102, 120)
ax.set_ylim(6, 23)
ax.grid(True, linestyle='--', alpha=0.3, color='black')
plt.title("BÃO MARIA 2025 - TÀU CÁ VỀ ĐÀ NẴNG AN TOÀN", fontsize=18, weight='bold', pad=15, color='#8B0000')

# Plot typhoon and boat paths (new logic)
ax.plot(full_lon, full_lat, color='gray', linestyle=':', linewidth=2, alpha=0.5, zorder=5)
ax.plot([p[1] for p in boat_path], [p[0] for p in boat_path], color='cyan', linewidth=3, zorder=6)

# Animation (original + new)
storm_circle = plt.Circle((0, 0), 0.1, color='red', alpha=0.6, zorder=10)
ax.add_patch(storm_circle)
storm_eye, = ax.plot([], [], 'yo', markersize=4, markeredgecolor='black', zorder=11)
boat_marker, = ax.plot([], [], 's', color='lime', markersize=10, zorder=8)

time_box = ax.text(0.02, 0.95, "00H", transform=ax.transAxes, fontsize=12, weight='bold', color='white',
                   bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'), va='top')

def get_storm_color(value):
    if value < 0.3: return '#FFD700'
    elif value < 0.6: return '#FF8C00'
    elif value < 0.85: return '#FF0000'
    else: return '#800080'

# Interpolate for smooth animation
t_new = np.linspace(0, len(full_lat)-1, 120)
spl_x = make_interp_spline(range(len(full_lon)), full_lon, k=3)
spl_y = make_interp_spline(range(len(full_lat)), full_lat, k=3)
track_x = spl_x(t_new)
track_y = spl_y(t_new)
intensity_curve = 1 - np.linspace(-1, 1, 120)**2
track_s = 0.3 + intensity_curve * 0.9
track_c = intensity_curve

def update(frame):
    idx = min(frame, len(track_x)-1)
    cx, cy = track_x[idx], track_y[idx]
    size = track_s[idx]
    color_val = track_c[idx]
    
    storm_circle.center = (cx, cy)
    storm_circle.set_radius(size)
    storm_circle.set_color(get_storm_color(color_val))
    storm_eye.set_data([cx], [cy])
    
    boat_idx = min(frame // 2, len(boat_path)-1)
    boat_marker.set_data([boat_path[boat_idx][1]], [boat_path[boat_idx][0]])
    
    current_hour = int((frame / len(track_x)) * 48)
    cap_bao = "8-9" if color_val < 0.3 else "10-12" if color_val < 0.6 else "13-15" if color_val < 0.85 else "16+"
    
    time_box.set_text(f"Thời gian: {current_hour:02d}H\nCấp độ: {cap_bao}")
    time_box.set_bbox(dict(facecolor=get_storm_color(color_val), alpha=0.8, boxstyle='round,pad=0.5'))
    
    return storm_circle, storm_eye, boat_marker, time_box

ani = animation.FuncAnimation(fig, update, frames=120, interval=80, blit=False)
ani.save("typhoon_escape_original_map.gif", writer='pillow', fps=15)
plt.close()

print("HOÀN TẤT! GIF đã lưu: typhoon_escape_original_map.gif")