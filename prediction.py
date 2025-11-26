# prediction.py
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
    if storms_df.empty:
        print("\n✓ No storms affecting Vietnam found.")
        return storms_df
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

    X_raw = np.column_stack([lats, lons, winds, dists]).reshape(1, n_input_steps, 4).astype(float)

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

if __name__ == "__main__":
    # ===== CẤU HÌNH ĐƯỜNG DẪN =====
    csv_path = 'ibtracs.WP.list.v04r00.csv'
    model_path = 'typhoon_model.keras'  # ← .keras
    scaler_path = 'typhoon_scaler.pkl'

    print("\n" + "="*80)
    print("🌀 HỆ THỐNG DỰ ĐOÁN BÃO - WORKFLOW ĐẦY ĐỦ")
    print("="*80)

    # ===== BƯỚC 1: TRAIN MODEL (CHẠY 1 LẦN ĐẦU) =====
    # print("\n[Bước 1] TRAINING MODEL")
    # print("-" * 80)

    # model, processor, df = train_and_save_model(csv_path, model_path, scaler_path)

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
        print(f"\n→ Dùng file số 2 làm input cho simulation.py (A* navigation)")