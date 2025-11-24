"""
================================================================================
CODE 1: HỆ THỐNG DỰ ĐOÁN BÃO - PHIÊN BẢN ĐẦY ĐỦ
================================================================================
Tính năng:
1. Train model LSTM và lưu (.keras)
2. Trích xuất bão từ dataset với format chuẩn
3. Dự đoán bão từ N bước đầu
4. Export CSV chuẩn cho A* navigation

OUTPUT FORMAT (giống IBTrACS):
SID,NAME,SEASON,LAT,LON,WMO_WIND,DIST2LAND,ISO_TIME,FORECAST_TYPE
2024MARIA,MARIA,2024,15.2,118.5,45,400,2024-11-01 00:00:00,observed
2024MARIA,MARIA,2024,15.6,117.8,47,350,2024-11-01 03:00:00,observed
...
2024MARIA,MARIA,2024,16.2,115.1,52,200,2024-11-01 24:00:00,predicted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. DATA PROCESSOR
# ===========================================================================

class TyphoonDataProcessor:
    def __init__(self, sequence_length=8):
        self.sequence_length = sequence_length
        self.scaler_lat = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_lon = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_wind = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_dist = MinMaxScaler(feature_range=(-1, 1))

    def load_and_clean(self, csv_path):
        print("\n📂 Đang đọc dữ liệu...")
        df = pd.read_csv(csv_path, low_memory=False)

        df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df['WMO_WIND'] = pd.to_numeric(df['WMO_WIND'], errors='coerce')
        df['DIST2LAND'] = pd.to_numeric(df['DIST2LAND'], errors='coerce')

        # LỌC CHO TRAINING: TẤT CẢ BÃO VÙNG TÂY THÁI BÌNH DƯƠNG
        df = df[
            (df['SEASON'] >= 2000) &  # Từ năm 2000
            (df['BASIN'] == 'WP')      # Tây Thái Bình Dương
        ].copy()

        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df.sort_values(['SID', 'ISO_TIME'])

        print(f"✓ Tìm thấy {df['SID'].nunique()} cơn bão (TẤT CẢ Tây Thái Bình Dương)")
        print(f"✓ Tổng {len(df):,} điểm quan sát")

        return df

    def create_sequences(self, df):
        print("\n🔧 Đang tạo training sequences...")
        X_sequences = []
        y_sequences = []
        storm_ids = []

        for sid in df['SID'].unique():
            storm = df[df['SID'] == sid].copy()
            if len(storm) < self.sequence_length + 1:
                continue

            lats = storm['LAT'].values
            lons = storm['LON'].values
            winds = storm['WMO_WIND'].fillna(50).values
            dists = storm['DIST2LAND'].fillna(200).values

            for i in range(len(storm) - self.sequence_length):
                seq_lat = lats[i:i + self.sequence_length]
                seq_lon = lons[i:i + self.sequence_length]
                seq_wind = winds[i:i + self.sequence_length]
                seq_dist = dists[i:i + self.sequence_length]

                target_lat = lats[i + self.sequence_length]
                target_lon = lons[i + self.sequence_length]

                X = np.column_stack([seq_lat, seq_lon, seq_wind, seq_dist])
                y = np.array([target_lat, target_lon])

                X_sequences.append(X)
                y_sequences.append(y)
                storm_ids.append(sid)

        X = np.array(X_sequences)
        y = np.array(y_sequences)

        print(f"✓ Tạo được {len(X):,} sequences")
        return X, y, storm_ids

    def normalize(self, X, y, fit=True):
        X_norm = X.copy()
        y_norm = y.copy()

        if fit:
            self.scaler_lat.fit(X[:, :, 0].reshape(-1, 1))
            self.scaler_lon.fit(X[:, :, 1].reshape(-1, 1))
            self.scaler_wind.fit(X[:, :, 2].reshape(-1, 1))
            self.scaler_dist.fit(X[:, :, 3].reshape(-1, 1))

        X_norm[:, :, 0] = self.scaler_lat.transform(X[:, :, 0].reshape(-1, 1)).reshape(X[:, :, 0].shape)
        X_norm[:, :, 1] = self.scaler_lon.transform(X[:, :, 1].reshape(-1, 1)).reshape(X[:, :, 1].shape)
        X_norm[:, :, 2] = self.scaler_wind.transform(X[:, :, 2].reshape(-1, 1)).reshape(X[:, :, 2].shape)
        X_norm[:, :, 3] = self.scaler_dist.transform(X[:, :, 3].reshape(-1, 1)).reshape(X[:, :, 3].shape)

        y_norm[:, 0] = self.scaler_lat.transform(y[:, 0].reshape(-1, 1)).reshape(-1)
        y_norm[:, 1] = self.scaler_lon.transform(y[:, 1].reshape(-1, 1)).reshape(-1)

        return X_norm, y_norm

    def denormalize_output(self, y_norm):
        y = y_norm.copy()
        y[:, 0] = self.scaler_lat.inverse_transform(y[:, 0].reshape(-1, 1)).reshape(-1)
        y[:, 1] = self.scaler_lon.inverse_transform(y[:, 1].reshape(-1, 1)).reshape(-1)
        return y

# ===========================================================================
# 2. MODEL
# ===========================================================================

class TyphoonLSTM:
    def __init__(self, sequence_length=8, n_features=4):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None

    def build_model(self):
        print("\n🏗️  Xây dựng mô hình LSTM...")
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True,
                       input_shape=(self.sequence_length, self.n_features), name='lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False, name='lstm_2'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, name='output')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])
        self.model = model
        print("✓ Mô hình đã sẵn sàng!")
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        print("\n🎓 Bắt đầu huấn luyện...")
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        print("✓ Huấn luyện hoàn tất!")
        return history

    def predict_trajectory(self, initial_sequence, n_steps=24):
        predictions = []
        current_seq = initial_sequence.copy()

        for step in range(n_steps):
            next_point = self.model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
            predictions.append(next_point)

            new_step = np.array([
                next_point[0], next_point[1],
                current_seq[-1, 2], current_seq[-1, 3]
            ])
            current_seq = np.vstack([current_seq[1:], new_step])

        return np.array(predictions)

    def save_model(self, model_path, scaler_path, processor):
        """Lưu model (.keras) và scaler"""
        print(f"\n💾 Đang lưu model...")
        self.model.save(model_path)  # Tự động lưu .keras

        scaler_data = {
            'scaler_lat': processor.scaler_lat,
            'scaler_lon': processor.scaler_lon,
            'scaler_wind': processor.scaler_wind,
            'scaler_dist': processor.scaler_dist,
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"✅ Đã lưu:")
        print(f"   - Model: {model_path}")
        print(f"   - Scaler: {scaler_path}")

    @staticmethod
    def load_model(model_path, scaler_path):
        """Load model (.keras) và scaler"""
        print(f"\n📥 Đang load model...")
        lstm = TyphoonLSTM()
        lstm.model = keras.models.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)

        processor = TyphoonDataProcessor()
        processor.scaler_lat = scaler_data['scaler_lat']
        processor.scaler_lon = scaler_data['scaler_lon']
        processor.scaler_wind = scaler_data['scaler_wind']
        processor.scaler_dist = scaler_data['scaler_dist']

        print(f"✅ Load thành công!")
        return lstm, processor

# ===========================================================================
# 3. TRAINING WORKFLOW
# ===========================================================================

def train_and_save_model(csv_path, model_path, scaler_path):
    """Train model và lưu"""
    print("="*80)
    print("🌀 TRAINING MÔ HÌNH DỰ ĐOÁN BÃO")
    print("="*80)

    processor = TyphoonDataProcessor(sequence_length=8)
    df = processor.load_and_clean(csv_path)
    X, y, storm_ids = processor.create_sequences(df)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    X_train_norm, y_train_norm = processor.normalize(X_train, y_train, fit=True)
    X_val_norm, y_val_norm = processor.normalize(X_val, y_val, fit=False)

    model = TyphoonLSTM(sequence_length=8, n_features=4)
    model.build_model()
    history = model.train(X_train_norm, y_train_norm, X_val_norm, y_val_norm, epochs=50)

    model.save_model(model_path, scaler_path, processor)

    print("\n" + "="*80)
    print("✅ HOÀN TẤT!")
    print("="*80)

    return model, processor, df

# ===========================================================================
# 4. TRÍCH XUẤT BÃO TỪ DATASET
# ===========================================================================

def list_available_storms(df):
    """Liệt kê bão ẢNH HƯỞNG VIỆT NAM để dự đoán"""
    print("\n" + "="*80)
    print("📋 DANH SÁCH BÃO ẢNH HƯỞNG VIỆT NAM (để dự đoán)")
    print("="*80)

    storms = []
    for sid in df['SID'].unique():
        storm = df[df['SID'] == sid]
        if len(storm) < 8:
            continue

        # LỌC: Bão đi qua vùng Việt Nam
        vn_points = storm[
            (storm['LAT'] >= 8) & (storm['LAT'] <= 23) &
            (storm['LON'] >= 102) & (storm['LON'] <= 120) &
            (storm['DIST2LAND'] < 500)
        ]

        if len(vn_points) < 3:  # Ít nhất 3 điểm trong vùng VN
            continue

        name = storm['NAME'].iloc[0] if 'NAME' in storm.columns else 'UNKNOWN'
        season = int(storm['SEASON'].iloc[0]) if 'SEASON' in storm.columns else 0
        n_points = len(storm)
        start_lat = storm['LAT'].iloc[0]
        start_lon = storm['LON'].iloc[0]

        storms.append({
            'SID': sid,
            'NAME': name,
            'SEASON': season,
            'POINTS': n_points,
            'START_LAT': start_lat,
            'START_LON': start_lon
        })

    storms_df = pd.DataFrame(storms)
    storms_df = storms_df.sort_values(['SEASON', 'NAME'], ascending=[False, True])

    print(f"\n✓ Tổng số bão ảnh hưởng VN: {len(storms_df)}")
    print(f"\n{storms_df.head(20).to_string(index=False)}")
    print(f"\n... và {len(storms_df) - 20} bão khác")

    return storms_df

def extract_storm_by_name(df, storm_name, output_csv=None):
    """
    Trích xuất bão theo tên

    Args:
        df: DataFrame IBTrACS
        storm_name: Tên bão (VD: "MARIA", "NORU")
        output_csv: Đường dẫn lưu file (optional)

    Returns:
        DataFrame với format chuẩn
    """
    print(f"\n🔍 Đang tìm bão: {storm_name}")

    # Tìm bão
    matches = df[df['NAME'].str.upper() == storm_name.upper()]

    if len(matches) == 0:
        print(f"❌ Không tìm thấy bão {storm_name}")
        return None

    # Nếu có nhiều bão cùng tên, lấy bão gần nhất
    if matches['SID'].nunique() > 1:
        print(f"⚠️  Tìm thấy {matches['SID'].nunique()} bão tên {storm_name}")
        print("   Lấy bão gần nhất...")
        latest_season = matches['SEASON'].max()
        matches = matches[matches['SEASON'] == latest_season]

    sid = matches['SID'].iloc[0]
    storm = df[df['SID'] == sid].copy()

    # Format chuẩn
    storm_export = storm[['SID', 'NAME', 'SEASON', 'LAT', 'LON',
                          'WMO_WIND', 'DIST2LAND', 'ISO_TIME']].copy()
    storm_export['FORECAST_TYPE'] = 'observed'

    print(f"✅ Tìm thấy bão:")
    print(f"   SID: {sid}")
    print(f"   Tên: {storm['NAME'].iloc[0]}")
    print(f"   Năm: {int(storm['SEASON'].iloc[0])}")
    print(f"   Số điểm: {len(storm)}")
    print(f"   Quỹ đạo: ({storm['LAT'].iloc[0]:.2f}°N, {storm['LON'].iloc[0]:.2f}°E)")
    print(f"           → ({storm['LAT'].iloc[-1]:.2f}°N, {storm['LON'].iloc[-1]:.2f}°E)")

    if output_csv:
        storm_export.to_csv(output_csv, index=False)
        print(f"💾 Đã lưu: {output_csv}")

    return storm_export

# ===========================================================================
# 5. DỰ ĐOÁN BÃO TỪ N BƯỚC ĐẦU
# ===========================================================================

def predict_from_storm_steps(model, processor, storm_df, start_step=0,
                             n_input_steps=8, n_predict_steps=20, output_csv=None):
    """
    Dự đoán bão từ N bước đầu

    Args:
        model: TyphoonLSTM model
        processor: TyphoonDataProcessor
        storm_df: DataFrame bão (từ extract_storm_by_name)
        start_step: Bắt đầu từ bước nào (0 = từ đầu)
        n_input_steps: Số bước làm input (8)
        n_predict_steps: Số bước dự đoán (20)
        output_csv: Đường dẫn lưu

    Returns:
        DataFrame đầy đủ (observed + predicted)
    """
    print(f"\n🔮 DỰ ĐOÁN BÃO")
    print(f"   Input: Bước {start_step} → {start_step + n_input_steps - 1}")
    print(f"   Predict: {n_predict_steps} bước ({n_predict_steps * 3}h)")

    # Kiểm tra đủ dữ liệu
    if len(storm_df) < start_step + n_input_steps:
        print(f"❌ Không đủ dữ liệu! Cần ít nhất {start_step + n_input_steps} điểm")
        return None

    # Lấy N bước
    input_data = storm_df.iloc[start_step:start_step + n_input_steps].copy()

    lats = input_data['LAT'].values
    lons = input_data['LON'].values
    winds = input_data['WMO_WIND'].fillna(50).values
    dists = input_data['DIST2LAND'].fillna(200).values

    X_raw = np.column_stack([lats, lons, winds, dists]).reshape(1, n_input_steps, 4)

    # Normalize
    X_norm = X_raw.copy()
    X_norm[0, :, 0] = processor.scaler_lat.transform(X_raw[0, :, 0].reshape(-1, 1)).reshape(-1)
    X_norm[0, :, 1] = processor.scaler_lon.transform(X_raw[0, :, 1].reshape(-1, 1)).reshape(-1)
    X_norm[0, :, 2] = processor.scaler_wind.transform(X_raw[0, :, 2].reshape(-1, 1)).reshape(-1)
    X_norm[0, :, 3] = processor.scaler_dist.transform(X_raw[0, :, 3].reshape(-1, 1)).reshape(-1)

    # Predict
    pred_norm = model.predict_trajectory(X_norm[0], n_steps=n_predict_steps)
    predictions = processor.denormalize_output(pred_norm)

    print(f"✅ Dự đoán xong!")

    # Tạo DataFrame kết quả
    # 1. Phần observed
    result_df = input_data[['SID', 'NAME', 'SEASON', 'LAT', 'LON',
                            'WMO_WIND', 'DIST2LAND', 'ISO_TIME']].copy()
    result_df['FORECAST_TYPE'] = 'observed'

    # 2. Phần predicted
    last_time = pd.to_datetime(input_data['ISO_TIME'].iloc[-1])
    pred_data = []

    for i, (lat, lon) in enumerate(predictions):
        pred_data.append({
            'SID': input_data['SID'].iloc[0],
            'NAME': input_data['NAME'].iloc[0],
            'SEASON': input_data['SEASON'].iloc[0],
            'LAT': lat,
            'LON': lon,
            'WMO_WIND': winds[-1],  # Giả sử giữ nguyên
            'DIST2LAND': max(0, dists[-1] - i*10),  # Ước lượng
            'ISO_TIME': last_time + timedelta(hours=(i+1)*3),
            'FORECAST_TYPE': 'predicted'
        })

    pred_df = pd.DataFrame(pred_data)
    result_df = pd.concat([result_df, pred_df], ignore_index=True)

    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"💾 Đã lưu: {output_csv}")

    # Vẽ
    _plot_forecast(result_df)

    return result_df

def _plot_forecast(forecast_df):
    """Vẽ bão observed + predicted"""
    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([100, 130, 5, 25], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#c8e6c9', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=2, edgecolor='#2e7d32', zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Observed
    obs = forecast_df[forecast_df['FORECAST_TYPE'] == 'observed']
    ax.plot(obs['LON'], obs['LAT'], 'b-', linewidth=3,
           label=f'Quan sát ({len(obs)} điểm)', transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(obs['LON'], obs['LAT'], c='blue', s=150, marker='o',
              edgecolors='white', linewidths=2, transform=ccrs.PlateCarree(), zorder=6)

    # Predicted
    pred = forecast_df[forecast_df['FORECAST_TYPE'] == 'predicted']
    ax.plot(pred['LON'], pred['LAT'], 'r--', linewidth=3,
           label=f'Dự đoán ({len(pred)} điểm = {len(pred)*3}h)', transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(pred['LON'], pred['LAT'], c='red', s=150, marker='^',
              edgecolors='white', linewidths=2, transform=ccrs.PlateCarree(), zorder=6)

    storm_name = forecast_df['NAME'].iloc[0]
    storm_year = int(forecast_df['SEASON'].iloc[0])
    ax.set_title(f'🌀 BÃO {storm_name} ({storm_year})', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    plt.show()



# ===========================================================================
# 6. MAIN - WORKFLOW ĐẦY ĐỦ
# ===========================================================================

if __name__ == "__main__":
    # ===== CẤU HÌNH ĐƯỜNG DẪN =====
    csv_path = 'ibtracs.WP.list.v04r00.csv'
    model_path = 'typhoon_model.keras'  # ← .keras
    scaler_path = 'typhoon_scaler.pkl'

    print("\n" + "="*80)
    print("🌀 HỆ THỐNG DỰ ĐOÁN BÃO - WORKFLOW ĐẦY ĐỦ")
    print("="*80)

    # ===== BƯỚC 1: TRAIN MODEL (CHẠY 1 LẦN ĐẦU) =====
    #print("\n[Bước 1] TRAINING MODEL")
    #print("-" * 80)

    #model, processor, df = train_and_save_model(csv_path, model_path, scaler_path)

    # ===== BƯỚC 2: LOAD MODEL (CÁC LẦN SAU) =====
    print("\n[Bước 2] LOAD MODEL ĐÃ TRAIN")
    print("-" * 80)
    model, processor = TyphoonLSTM.load_model(model_path, scaler_path)
    processor_temp = TyphoonDataProcessor()
    df = processor_temp.load_and_clean(csv_path)

    # ===== BƯỚC 3: LIỆT KÊ BÃO =====
    print("\n[Bước 3] LIỆT KÊ BÃO TRONG DATASET")
    print("-" * 80)
    storms_list = list_available_storms(df)

    # ===== BƯỚC 4: TRÍCH XUẤT BÃO THEO TÊN =====
    print("\n[Bước 4] TRÍCH XUẤT BÃO THEO TÊN")
    print("-" * 80)

    # VD: Nhập tên bão
    storm_name = input("\n📝 Nhập tên bão (VD: NORU, MARIA, RAI): ").strip()

    storm_df = extract_storm_by_name(
        df,
        storm_name,
        output_csv=f'storm_{storm_name}_observed.csv'
    )

    if storm_df is None:
        print("❌ Không tìm thấy bão!")
    else:
        # ===== BƯỚC 5: DỰ ĐOÁN TỪ N BƯỚC ĐẦU =====
        print("\n[Bước 5] DỰ ĐOÁN TỪ N BƯỚC ĐẦU")
        print("-" * 80)

        # Cho phép chọn bước
        print(f"\n📊 Bão có {len(storm_df)} điểm")
        print("   Chọn N bước đầu để dự đoán tiếp:")

        n_input = int(input("   Số bước input (8-16): ") or "8")
        n_predict = int(input("   Số bước dự đoán (10-30): ") or "20")

        forecast_df = predict_from_storm_steps(
            model, processor, storm_df,
            start_step=0,
            n_input_steps=n_input,
            n_predict_steps=n_predict,
            output_csv=f'storm_{storm_name}_forecast.csv'
        )

        print("\n" + "="*80)
        print("✅ HOÀN TẤT!")
        print("="*80)
        print(f"\nFile đã lưu:")
        print(f"  1. storm_{storm_name}_observed.csv")
        print(f"  2. storm_{storm_name}_forecast.csv")
        print(f"\n→ Dùng file số 2 làm input cho Code 2 (A* navigation)")

"""
================================================================================
CODE 2: HỆ THỐNG ĐIỀU HƯỚNG TÀU CÁ TRÁNH BÃO (A*) - PHIÊN BẢN HOÀN THIỆN
================================================================================
Tính năng:
1. Tự động chọn cảng gần nhất an toàn
2. Tránh đất liền (Hải Nam, VN)
3. Bão bắt đầu từ bước quan sát đầu tiên
4. A* navigation với land mask
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. CONSTANTS & LOAD LAND POLYGONS
# ===========================================================================

BOAT_SPEED_KMH = 18.0
TIMESTEP_HOURS = 3
BOAT_MOVE_DISTANCE = BOAT_SPEED_KMH * TIMESTEP_HOURS

TYPHOON_DANGER_RADIUS = 200
TYPHOON_WARNING_RADIUS = 400

VIETNAM_PORTS = {
    'Hòn Gai': (20.96, 107.08),
    'Hải Phòng': (20.87, 106.68),
    'Cửa Lò': (18.80, 105.73),
    'Đà Nẵng': (16.07, 108.22),
    'Quy Nhơn': (13.77, 109.22),
    'Nha Trang': (12.25, 109.19),
    'Vũng Tàu': (10.35, 107.08),
}

def load_land_polygons(json_path=None):
    """
    Trả về polygon đất liền mặc định (không cần file JSON).
    """

    # Polygon bạn đưa ra
    land_points = [
        (12.6, 102.0),
        (10.0, 104.83333333333333),
        (8.5, 104.66666666666667),
        (8.6, 105.0),
        (9.5, 106.28333333333333),
        (10.333333333333334, 106.88333333333334),
        (10.583333333333334, 107.45),
        (11.466666666666667, 109.0),
        (14.266666666666667, 109.18333333333334),
        (15.266666666666667, 108.83333333333333),
        (16.0, 108.11666666666666),
        (17.7, 106.483333333333334),
        (18.916666666666668, 105.56666666666666),
        (19.883333333333333, 106.0),
        (20.75, 106.76666666666667),
        (21.533333333333335, 107.88333333333334),
        (21.666666666666668, 108.76666666666667),
        (21.2, 109.75),
        (21.0, 109.51666666666667),
        (19.266666666666666, 108.56666666666666),
        (18.45, 108.63333333333334),
        (18.0, 109.5),
        (18.666666666666668, 110.5),
        (19.833333333333332, 111.0),
        (22.566666666666666, 115.65),
        (24.5, 118.0),
        (24.5, 102.0)
    ]

    print("✅ Đã load polygon Đất liền (offline, không cần JSON)")
    print(f"   - Số điểm: {len(land_points)}")

    return {
        "Đất liền": land_points
    }


# Load polygons khi khởi động
LAND_POLYGONS = load_land_polygons()

# ===========================================================================
# 2. GEOGRAPHIC UTILITIES
# ===========================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def bearing_to(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371
    lat, lon, bearing = map(np.radians, [lat, lon, bearing_deg])
    new_lat = np.arcsin(
        np.sin(lat) * np.cos(distance_km/R) +
        np.cos(lat) * np.sin(distance_km/R) * np.cos(bearing)
    )
    new_lon = lon + np.arctan2(
        np.sin(bearing) * np.sin(distance_km/R) * np.cos(lat),
        np.cos(distance_km/R) - np.sin(lat) * np.sin(new_lat)
    )
    return np.degrees(new_lat), np.degrees(new_lon)

def is_point_in_polygon(lat, lon, polygon):
    """Kiểm tra điểm có nằm trong polygon không (Ray casting algorithm)"""
    n = len(polygon)
    inside = False

    p1_lat, p1_lon = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon[i % n]
        if lon > min(p1_lon, p2_lon):
            if lon <= max(p1_lon, p2_lon):
                if lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        xinters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    if p1_lat == p2_lat or lat <= xinters:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon

    return inside

def is_on_land(lat, lon):
    """Kiểm tra vị trí có trên đất liền không"""
    for land_name, polygon in LAND_POLYGONS.items():
        if is_point_in_polygon(lat, lon, polygon):
            return True
    return False

def distance_to_land(lat, lon):
    """Tính khoảng cách đến đất liền gần nhất"""
    min_dist = float('inf')

    for land_name, polygon in LAND_POLYGONS.items():
        for land_lat, land_lon in polygon:
            dist = haversine_distance(lat, lon, land_lat, land_lon)
            min_dist = min(min_dist, dist)

    return min_dist

# ===========================================================================
# 3. TYPHOON DATA LOADER
# ===========================================================================

@dataclass
class TyphoonState:
    time_step: int
    lat: float
    lon: float
    wind_speed: float
    forecast_type: str

    def get_position(self):
        return (self.lat, self.lon)

class TyphoonForecastLoader:
    """Load dự đoán bão - BẮT ĐẦU TỪ BƯỚC QUAN SÁT ĐẦU TIÊN"""

    def __init__(self, csv_path):
        print(f"\n📂 Đang đọc dự đoán bão: {csv_path}")
        self.df = pd.read_csv(csv_path)

        required_cols = ['LAT', 'LON', 'WMO_WIND', 'FORECAST_TYPE']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột: {col}")

        # Parse TẤT CẢ từ đầu (observed + predicted)
        self.states = []
        for i, row in self.df.iterrows():
            self.states.append(TyphoonState(
                time_step=i,
                lat=float(row['LAT']),
                lon=float(row['LON']),
                wind_speed=float(row['WMO_WIND']) if pd.notna(row['WMO_WIND']) else 50.0,
                forecast_type=str(row['FORECAST_TYPE'])
            ))

        self.observed_count = sum(1 for s in self.states if s.forecast_type == 'observed')

        print(f"✅ Load thành công:")
        print(f"   - Tổng điểm: {len(self.states)}")
        print(f"   - Quan sát: {self.observed_count}")
        print(f"   - Dự đoán: {len(self.states) - self.observed_count}")
        print(f"   - Vị trí đầu: ({self.states[0].lat:.2f}°N, {self.states[0].lon:.2f}°E)")
        print(f"   - Bão bắt đầu từ bước 0 (quan sát đầu tiên)")

    def get_typhoon_at_step(self, time_step):
        """BÃO BẮT ĐẦU TỪ BƯỚC 0"""
        if time_step >= len(self.states):
            return None
        return self.states[time_step]

    def get_visible_forecast(self, current_step, lookahead=4):
        """Thông tin bão mà tàu biết"""
        visible = []
        for i in range(lookahead):
            typhoon = self.get_typhoon_at_step(current_step + i)
            if typhoon:
                visible.append(typhoon)
        return visible

    def is_active(self, current_step):
        return self.get_typhoon_at_step(current_step) is not None

    def get_all_forecast(self):
        return self.states

# ===========================================================================
# 4. AUTO SELECT NEAREST SAFE PORT
# ===========================================================================

def select_nearest_safe_port(boat_lat, boat_lon, typhoon_loader):
    """
    Tự động chọn cảng gần nhất và an toàn

    Tiêu chí:
    1. Gần nhất
    2. Không bị bão đi qua
    3. Có thể đến được
    """
    print(f"\n🔍 Đang tìm cảng gần nhất an toàn...")
    print(f"   Vị trí tàu: ({boat_lat:.2f}°N, {boat_lon:.2f}°E)")

    port_scores = []

    for port_name, (port_lat, port_lon) in VIETNAM_PORTS.items():
        # 1. Khoảng cách
        dist = haversine_distance(boat_lat, boat_lon, port_lat, port_lon)

        # 2. Kiểm tra bão có đi qua cảng không
        min_typhoon_dist = float('inf')
        for typhoon in typhoon_loader.get_all_forecast():
            ty_dist = haversine_distance(port_lat, port_lon, typhoon.lat, typhoon.lon)
            min_typhoon_dist = min(min_typhoon_dist, ty_dist)

        # 3. Score: ưu tiên cảng gần + xa bão
        score = dist - min_typhoon_dist * 0.3  # Penalty nếu bão gần cảng

        port_scores.append({
            'name': port_name,
            'lat': port_lat,
            'lon': port_lon,
            'distance': dist,
            'min_typhoon_dist': min_typhoon_dist,
            'score': score
        })

        print(f"   {port_name:15s}: {dist:6.0f}km | Bão gần nhất: {min_typhoon_dist:6.0f}km | Score: {score:7.1f}")

    # Chọn cảng có score thấp nhất
    best_port = min(port_scores, key=lambda x: x['score'])

    print(f"\n✅ Chọn cảng: {best_port['name']}")
    print(f"   Khoảng cách: {best_port['distance']:.0f} km")
    print(f"   Bão gần nhất cảng: {best_port['min_typhoon_dist']:.0f} km")

    return best_port['name']

# ===========================================================================
# 5. A* PATHFINDING WITH LAND AVOIDANCE
# ===========================================================================

class TyphoonAvoidanceAStar:
    """A* với tránh đất liền"""

    def __init__(self, goal_lat, goal_lon, typhoon_loader):
        self.goal_lat = goal_lat
        self.goal_lon = goal_lon
        self.typhoon_loader = typhoon_loader

    def heuristic(self, lat, lon, time_step):
        """
        h(n) = distance_to_goal + typhoon_risk + land_penalty
        """
        # 1. Khoảng cách đến cảng
        dist_to_goal = haversine_distance(lat, lon, self.goal_lat, self.goal_lon)

        # 2. Rủi ro bão
        typhoon_risk = 0
        visible_typhoons = self.typhoon_loader.get_visible_forecast(time_step, lookahead=4)

        for typhoon in visible_typhoons:
            ty_lat, ty_lon = typhoon.get_position()
            dist_to_typhoon = haversine_distance(lat, lon, ty_lat, ty_lon)

            if dist_to_typhoon < TYPHOON_DANGER_RADIUS:
                typhoon_risk += 10000
            elif dist_to_typhoon < TYPHOON_WARNING_RADIUS:
                typhoon_risk += 500

        # 3. Penalty đất liền
        land_penalty = 0
        if is_on_land(lat, lon):
            land_penalty = 50000  # CỰC KỲ TRÁNH ĐẤT LIỀN
        else:
            dist_land = distance_to_land(lat, lon)
            if dist_land < 30:  # Quá gần bờ (<30km)
                land_penalty = 1000

        return dist_to_goal + typhoon_risk + land_penalty

    def get_neighbors(self, lat, lon, move_distance):
        """8 hướng - CHỈ LẤY ĐIỂM TRÊN BIỂN"""
        directions = [
            (0, 'Bắc'), (45, 'Đông Bắc'), (90, 'Đông'), (135, 'Đông Nam'),
            (180, 'Nam'), (225, 'Tây Nam'), (270, 'Tây'), (315, 'Tây Bắc'),
        ]

        neighbors = []
        for bearing, name in directions:
            new_lat, new_lon = destination_point(lat, lon, bearing, move_distance)

            # Kiểm tra trong vùng biển hợp lệ
            if 8 <= new_lat <= 23 and 102 <= new_lon <= 120:
                # Kiểm tra KHÔNG trên đất liền
                if not is_on_land(new_lat, new_lon):
                    neighbors.append((new_lat, new_lon, name))

        return neighbors

    def find_next_move(self, current_lat, current_lon, current_time_step):
        """Tìm bước tốt nhất - TRÁNH ĐẤT LIỀN"""
        neighbors = self.get_neighbors(current_lat, current_lon, BOAT_MOVE_DISTANCE)

        if len(neighbors) == 0:
            # Không có hướng nào an toàn, thử giảm khoảng cách
            neighbors = self.get_neighbors(current_lat, current_lon, BOAT_MOVE_DISTANCE / 2)

        if len(neighbors) == 0:
            return None  # Thực sự kẹt

        best_move = None
        best_f_cost = float('inf')

        for new_lat, new_lon, direction in neighbors:
            g = BOAT_MOVE_DISTANCE
            h = self.heuristic(new_lat, new_lon, current_time_step + 1)
            f = g + h

            if f < best_f_cost:
                best_f_cost = f
                best_move = (new_lat, new_lon, direction)

        return best_move

# ===========================================================================
# 6. BOAT SIMULATION
# ===========================================================================

@dataclass
class BoatState:
    lat: float
    lon: float
    time_step: int
    direction: str = "Bắt đầu"
    status: str = "Di chuyển"
    distance_to_port: float = 0.0

def simulate_boat_evacuation(
    boat_start_lat,
    boat_start_lon,
    port_name,
    typhoon_loader,
    max_steps=50
):
    """Mô phỏng - TRÁNH ĐẤT LIỀN + VỀ CẢNG KHI GẦN"""
    port_lat, port_lon = VIETNAM_PORTS[port_name]
    astar = TyphoonAvoidanceAStar(port_lat, port_lon, typhoon_loader)

    # Kiểm tra vị trí xuất phát hợp lệ
    if is_on_land(boat_start_lat, boat_start_lon):
        print(f"❌ VỊ TRÍ XUẤT PHÁT TRÊN ĐẤT LIỀN! ({boat_start_lat:.2f}°N, {boat_start_lon:.2f}°E)")
        return [], False

    current_lat, current_lon = boat_start_lat, boat_start_lon
    time_step = 0

    dist_start = haversine_distance(current_lat, current_lon, port_lat, port_lon)
    boat_trajectory = [BoatState(current_lat, current_lon, 0, "Bắt đầu", "Di chuyển", dist_start)]

    print("\n" + "="*80)
    print("🚢 MÔ PHỎNG TÀU CÁ TRÁNH BÃO")
    print("="*80)
    print(f"📍 Xuất phát: ({current_lat:.2f}°N, {current_lon:.2f}°E)")
    print(f"⚓ Cảng đích: {port_name} ({port_lat:.2f}°N, {port_lon:.2f}°E)")
    print(f"📏 Khoảng cách: {dist_start:.0f} km")
    print("="*80 + "\n")

    while time_step < max_steps:
        dist_to_port = haversine_distance(current_lat, current_lon, port_lat, port_lon)

        # ===== FIX: VỀ CẢNG KHI ĐỦ GẦN =====
        # Nếu khoảng cách < 1.5 lần khoảng cách di chuyển, về luôn
        if dist_to_port < BOAT_MOVE_DISTANCE * 1.5:
            boat_trajectory.append(
                BoatState(port_lat, port_lon, time_step, "Cập cảng", "Đã về cảng", 0)
            )
            print(f"\n✅ Bước {time_step}: ĐỦ GẦN CẢNG ({dist_to_port:.1f}km) - VỀ CẢNG AN TOÀN!")
            return boat_trajectory, True

        if not typhoon_loader.is_active(time_step):
            print(f"\n🌤️  Bước {time_step}: Bão đã tan, đi thẳng về cảng")

            # Tính hướng về cảng
            bearing = bearing_to(current_lat, current_lon, port_lat, port_lon)

            # Nếu gần, đi chính xác đến cảng
            if dist_to_port < BOAT_MOVE_DISTANCE:
                current_lat, current_lon = port_lat, port_lon
            else:
                current_lat, current_lon = destination_point(
                    current_lat, current_lon, bearing, BOAT_MOVE_DISTANCE
                )

            # Kiểm tra không lên đất
            if is_on_land(current_lat, current_lon):
                print(f"⚠️  Điểm tiếp theo trên đất, tìm hướng khác...")
                time_step += 1
                continue

            boat_trajectory.append(
                BoatState(current_lat, current_lon, time_step, "Về cảng", "Di chuyển", dist_to_port)
            )
            time_step += 1
            continue

        next_move = astar.find_next_move(current_lat, current_lon, time_step)

        if next_move is None:
            print(f"\n❌ Bước {time_step}: KHÔNG TÌM THẤY ĐƯỜNG AN TOÀN!")
            boat_trajectory.append(
                BoatState(current_lat, current_lon, time_step, "Kẹt", "Gặp nguy hiểm", dist_to_port)
            )
            return boat_trajectory, False

        new_lat, new_lon, direction = next_move
        current_lat, current_lon = new_lat, new_lon

        boat_trajectory.append(
            BoatState(current_lat, current_lon, time_step, direction, "Di chuyển", dist_to_port)
        )

        if time_step % 3 == 0:
            print(f"📍 Bước {time_step} (+{time_step*3}h): {direction:12s} | Còn {dist_to_port:6.0f} km")

        time_step += 1

    print(f"\n⏰ HẾT THỜI GIAN!")
    return boat_trajectory, False

# ===========================================================================
# 7. SAVE & GIF (với land polygons)
# ===========================================================================

def save_boat_trajectory(boat_trajectory, output_csv):
    data = []
    for state in boat_trajectory:
        data.append({
            'time_step': state.time_step,
            'lat': state.lat,
            'lon': state.lon,
            'direction': state.direction,
            'status': state.status,
            'distance_to_port': state.distance_to_port
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Đã lưu quỹ đạo tàu: {output_csv}")

def create_evacuation_gif(boat_trajectory, typhoon_loader, port_name, success, output_gif):
    print(f"\n🎬 Đang tạo GIF animation...")

    port_lat, port_lon = VIETNAM_PORTS[port_name]
    all_typhoons = typhoon_loader.get_all_forecast()

    obs_typhoons = [t for t in all_typhoons if t.forecast_type == 'observed']
    pred_typhoons = [t for t in all_typhoons if t.forecast_type == 'predicted']

    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    def init_map():
        ax.clear()
        ax.set_extent([102, 122, 8, 23], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#c8e6c9', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=2, edgecolor='#2e7d32', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='#666', linestyle='--', alpha=0.5, zorder=2)

        # Vẽ land polygons (để visualize vùng tránh)
        for land_name, polygon in LAND_POLYGONS.items():
            lats, lons = zip(*polygon)
            poly = Polygon(list(zip(lons, lats)), transform=ccrs.PlateCarree(),
                          facecolor='red', alpha=0.1, edgecolor='red',
                          linewidth=1, linestyle='--', zorder=2)
            ax.add_patch(poly)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Tất cả cảng
        for name, (lat, lon) in VIETNAM_PORTS.items():
            if name == port_name:
                ax.scatter(lon, lat, c='gold', s=800, marker='H',
                          edgecolors='black', linewidths=3,
                          transform=ccrs.PlateCarree(), zorder=12)
            else:
                ax.scatter(lon, lat, c='gray', s=400, marker='h',
                          edgecolors='white', linewidths=2,
                          transform=ccrs.PlateCarree(), zorder=11, alpha=0.6)

            ax.text(lon + 0.3, lat + 0.3, name, fontsize=9,
                   transform=ccrs.PlateCarree(), zorder=13)

    def animate(frame):
        init_map()

        # Bão quan sát
        if len(obs_typhoons) > 0:
            obs_lats = [t.lat for t in obs_typhoons]
            obs_lons = [t.lon for t in obs_typhoons]
            ax.plot(obs_lons, obs_lats, 'orange', linewidth=2, linestyle=':',
                   alpha=0.7, transform=ccrs.PlateCarree(), zorder=4, label='Bão (quan sát)')
            ax.scatter(obs_lons, obs_lats, c='orange', s=60, marker='o',
                      edgecolors='white', linewidths=1, transform=ccrs.PlateCarree(), zorder=5)

        # Bão dự đoán
        pred_until = min(frame, len(pred_typhoons))
        if pred_until > 0:
            pred_lats = [t.lat for t in pred_typhoons[:pred_until]]
            pred_lons = [t.lon for t in pred_typhoons[:pred_until]]

            ax.plot(pred_lons, pred_lats, 'r-', linewidth=3,
                   alpha=0.8, transform=ccrs.PlateCarree(), zorder=6, label='Bão (dự báo)')
            ax.scatter(pred_lons, pred_lats, c='red', s=100, marker='o',
                      edgecolors='white', linewidths=2, transform=ccrs.PlateCarree(), zorder=7)

        # Vị trí bão hiện tại
        current_typhoon_idx = min(frame, len(all_typhoons) - 1)
        ty = all_typhoons[current_typhoon_idx]
        ax.scatter(ty.lon, ty.lat, c='#c62828', s=500, marker='*',
                  edgecolors='white', linewidths=3, transform=ccrs.PlateCarree(), zorder=10)

        circle = Circle((ty.lon, ty.lat), TYPHOON_DANGER_RADIUS/111,
                       transform=ccrs.PlateCarree(), facecolor='red', alpha=0.15,
                       edgecolor='red', linewidth=2, linestyle='--', zorder=4)
        ax.add_patch(circle)

        # Tàu
        boat_until = min(frame + 1, len(boat_trajectory))
        if boat_until > 0:
            boat_lats = [s.lat for s in boat_trajectory[:boat_until]]
            boat_lons = [s.lon for s in boat_trajectory[:boat_until]]

            color = '#2e7d32' if success else '#1565c0'
            ax.plot(boat_lons, boat_lats, color=color, linewidth=4,
                   transform=ccrs.PlateCarree(), zorder=8, label='Tàu cá')
            ax.scatter(boat_lons, boat_lats, c=color, s=120, marker='s',
                      edgecolors='white', linewidths=2, transform=ccrs.PlateCarree(), zorder=9)

            current_boat = boat_trajectory[boat_until - 1]
            ax.scatter(current_boat.lon, current_boat.lat, c='blue', s=600, marker='*',
                      edgecolors='yellow', linewidths=4, transform=ccrs.PlateCarree(), zorder=14)

        hours = frame * 3
        status_emoji = "✅" if success and boat_until == len(boat_trajectory) else "🚢"
        ax.set_title(f'{status_emoji} TÀU CÁ TRÁNH BÃO - Thời gian: +{hours}h',
                    fontsize=20, fontweight='bold', pad=15)

        if boat_until > 0:
            info = boat_trajectory[boat_until - 1]
            ax.text(0.02, 0.98,
                   f"Trạng thái: {info.status}\nHướng: {info.direction}\nCòn: {info.distance_to_port:.0f}km",
                   transform=ax.transAxes, fontsize=12, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    n_frames = max(len(boat_trajectory), len(all_typhoons))
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=500, repeat=True)

    writer = PillowWriter(fps=2)
    anim.save(output_gif, writer=writer)
    plt.close()

    print(f"✅ Đã lưu GIF: {output_gif}")

# ===========================================================================
# 8. MAIN - WORKFLOW
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚢 HỆ THỐNG ĐIỀU HƯỚNG TÀU CÁ TRÁNH BÃO - PHIÊN BẢN HOÀN THIỆN")
    print("="*80)

    # ===== BƯỚC 1: LOAD DỰ ĐOÁN BÃO =====
    print("\n[Bước 1] LOAD DỰ ĐOÁN BÃO")
    print("-" * 80)

    typhoon_csv = input("📂 Nhập đường dẫn file CSV bão (từ Code 1): ").strip()
    if not typhoon_csv:
        typhoon_csv = 'storm_NORU_forecast.csv'  # Default
        print(f"   Dùng mặc định: {typhoon_csv}")

    typhoon_loader = TyphoonForecastLoader(typhoon_csv)

    # ===== BƯỚC 2: CHỌN VỊ TRÍ TÀU =====
    print("\n[Bước 2] CHỌN VỊ TRÍ TÀU")
    print("-" * 80)
    print("1. Ngẫu nhiên (trong vùng biển VN)")
    print("2. Nhập tọa độ thủ công")

    choice = input("\nChọn (1/2): ").strip()

    if choice == '1':
        # Random - đảm bảo KHÔNG trên đất
        max_attempts = 100
        for attempt in range(max_attempts):
            boat_lat = np.random.uniform(12, 19)
            boat_lon = np.random.uniform(108, 116)

            if not is_on_land(boat_lat, boat_lon):
                print(f"✅ Vị trí ngẫu nhiên: ({boat_lat:.2f}°N, {boat_lon:.2f}°E)")
                break
        else:
            print("❌ Không tìm được vị trí ngẫu nhiên hợp lệ!")
            boat_lat, boat_lon = 15.5, 112.0  # Fallback
            print(f"   Dùng mặc định: ({boat_lat:.2f}°N, {boat_lon:.2f}°E)")
    else:
        # Manual
        boat_lat = float(input("Nhập vĩ độ (8-23): "))
        boat_lon = float(input("Nhập kinh độ (102-120): "))

        if is_on_land(boat_lat, boat_lon):
            print(f"⚠️  VỊ TRÍ TRÊN ĐẤT LIỀN! Tự động điều chỉnh...")
            # Di chuyển ra biển (tăng lon)
            while is_on_land(boat_lat, boat_lon) and boat_lon < 120:
                boat_lon += 0.1
            print(f"   Điều chỉnh thành: ({boat_lat:.2f}°N, {boat_lon:.2f}°E)")
        else:
            print(f"✅ Vị trí: ({boat_lat:.2f}°N, {boat_lon:.2f}°E)")

    # ===== BƯỚC 3: TỰ ĐỘNG CHỌN CẢNG GẦN NHẤT AN TOÀN =====
    print("\n[Bước 3] TỰ ĐỘNG CHỌN CẢNG GẦN NHẤT AN TOÀN")
    print("-" * 80)

    port_name = select_nearest_safe_port(boat_lat, boat_lon, typhoon_loader)

    # ===== BƯỚC 4: CHẠY SIMULATION =====
    print("\n[Bước 4] CHẠY SIMULATION")
    print("-" * 80)

    boat_trajectory, success = simulate_boat_evacuation(
        boat_lat, boat_lon, port_name, typhoon_loader, max_steps=50
    )

    if len(boat_trajectory) == 0:
        print("\n❌ MÔ PHỎNG THẤT BẠI!")
    else:
        # ===== BƯỚC 5: LƯU KẾT QUẢ =====
        print("\n[Bước 5] LƯU KẾT QUẢ")
        print("-" * 80)

        import os
        from IPython.display import Image, display   # ⭐ NEW: để show GIF

        csv_filename = os.path.basename(typhoon_csv)
        if 'storm_' in csv_filename:
            storm_name = csv_filename.split('_')[1].split('.')[0]
        elif 'forecast' in csv_filename:
            storm_name = csv_filename.split('_')[0]
        else:
            storm_name = 'unknown'

        output_csv = f'boat_trajectory_{storm_name}.csv'
        output_gif = f'boat_evacuation_{storm_name}.gif'

        save_boat_trajectory(boat_trajectory, output_csv)
        create_evacuation_gif(boat_trajectory, typhoon_loader, port_name, success, output_gif)

        # ⭐⭐⭐ NEW: Hiển thị ngay GIF sau khi tạo
        print("\n🖼️  HIỂN THỊ KẾT QUẢ GIF:")
        display(Image(filename=output_gif))

        # ===== BƯỚC 6: THỐNG KÊ =====
        print("\n" + "="*80)
        print("📊 THỐNG KÊ")
        print("="*80)
        print(f"✓ Kết quả: {'THÀNH CÔNG ✅' if success else 'THẤT BẠI ❌'}")
        print(f"✓ Số bước: {len(boat_trajectory)}")
        print(f"✓ Thời gian: {len(boat_trajectory)*3}h")
        print(f"✓ Quãng đường: {len(boat_trajectory)*BOAT_MOVE_DISTANCE:.0f} km")
        print(f"✓ Cảng đích: {port_name}")
        print("="*80)

        print(f"\n📁 File đã lưu:")
        print(f"   - Quỹ đạo: {output_csv}")
        print(f"   - GIF: {output_gif}")
