# prediction.py
"""
================================================================================
CODE 1: HỆ THỐNG DỰ ĐOÁN BÃO - PHIÊN BẢN ĐẦY ĐỦ (SỬ DỤNG HMM)
================================================================================
Tính năng:
1. Train model HMM và lưu (.pkl)
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
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

class GaussianHMM:
    def __init__(self, n_states, n_features, random_state=None):
        self.n_states = n_states
        self.n_features = n_features
        self.random_state = random_state
        
        # Initialize parameters
        rng = np.random.RandomState(random_state)
        
        # Initial state probabilities
        self.start_prob = np.ones(n_states) / n_states
        
        # Transition probabilities
        self.transmat = rng.dirichlet(np.ones(n_states), size=n_states)
        
        # Means and variances for each state
        self.means = rng.randn(n_states, n_features)
        self.variances = np.ones((n_states, n_features)) * 1.0
        
    def _gaussian_pdf(self, x, mean, variance):
        """Compute Gaussian probability density function."""
        exponent = -0.5 * np.sum(((x - mean) / np.sqrt(variance)) ** 2)
        coefficient = 1.0 / (np.sqrt((2 * np.pi) ** self.n_features * np.prod(variance)))
        return coefficient * np.exp(exponent)
    
    def _forward(self, obs):
        """Forward algorithm in log space."""
        T = len(obs)
        N = self.n_states
        log_alpha = np.full((T, N), -np.inf)
        
        log_emit = np.zeros(N)
        for j in range(N):
            log_emit[j] = np.log(self._gaussian_pdf(obs[0], self.means[j], self.variances[j]) + 1e-10)
        log_alpha[0] = np.log(self.start_prob + 1e-10) + log_emit
        
        for t in range(1, T):
            log_emit = np.zeros(N)
            for j in range(N):
                log_emit[j] = np.log(self._gaussian_pdf(obs[t], self.means[j], self.variances[j]) + 1e-10)
                log_terms = log_alpha[t-1] + np.log(self.transmat[:, j] + 1e-10)
                log_alpha[t, j] = logsumexp(log_terms) + log_emit[j]
        
        return log_alpha
    
    def _backward(self, obs):
        """Backward algorithm in log space."""
        T = len(obs)
        N = self.n_states
        log_beta = np.zeros((T, N))
        
        for t in range(T-2, -1, -1):
            log_emit = np.zeros(N)
            for j in range(N):
                log_emit[j] = np.log(self._gaussian_pdf(obs[t+1], self.means[j], self.variances[j]) + 1e-10)
            for i in range(N):
                log_terms = np.log(self.transmat[i, :] + 1e-10) + log_emit + log_beta[t+1]
                log_beta[t, i] = logsumexp(log_terms)
        
        return log_beta
    
    def _baum_welch(self, obs_list):
        """Baum-Welch for multiple sequences."""
        N = self.n_states
        xi_sum = np.zeros((N, N))
        obs_count_sum = np.zeros(N)
        weighted_obs_sum = np.zeros((N, self.n_features))
        weighted_sq_sum = np.zeros((N, self.n_features))
        start_gamma_sum = np.zeros(N)
        
        for obs in obs_list:
            if len(obs) < 2:
                continue
            T = len(obs)
            log_alpha = self._forward(obs)
            log_beta = self._backward(obs)
            log_lik = logsumexp(log_alpha[-1])
            if np.isinf(log_lik) or np.isnan(log_lik):
                continue  # Skip bad sequences
            
            log_gamma = log_alpha + log_beta - log_lik
            gamma = np.exp(log_gamma)
            
            start_gamma_sum += gamma[0]
            
            obs_count_sum += np.sum(gamma, axis=0)
            weighted_obs_sum += np.dot(gamma.T, obs)
            weighted_sq_sum += np.dot(gamma.T, obs**2)
            
            log_emit = np.zeros((T, N))
            for t in range(T):
                for j in range(N):
                    log_emit[t, j] = np.log(self._gaussian_pdf(obs[t], self.means[j], self.variances[j]) + 1e-10)
            
            for t in range(T-1):
                log_xi = log_alpha[t, :, np.newaxis] + \
                         np.log(self.transmat + 1e-10) + \
                         log_emit[t+1, np.newaxis, :] + \
                         log_beta[t+1, np.newaxis, :] - log_lik
                xi_sum += np.exp(log_xi)
        
        # M-step
        # Transition matrix
        row_sums = np.sum(xi_sum, axis=1, keepdims=True)
        self.transmat = xi_sum / row_sums
        nan_rows = np.isnan(self.transmat).any(axis=1)
        self.transmat[nan_rows] = 1.0 / N
        
        # Start probabilities
        sum_start = np.sum(start_gamma_sum)
        if sum_start > 0:
            self.start_prob = start_gamma_sum / sum_start
        else:
            self.start_prob = np.ones(N) / N
        
        # Means and variances
        for i in range(N):
            if obs_count_sum[i] > 0:
                self.means[i] = weighted_obs_sum[i] / obs_count_sum[i]
                var = weighted_sq_sum[i] / obs_count_sum[i] - (self.means[i] ** 2)
                self.variances[i] = np.maximum(var, 1e-6)
    
    def fit(self, obs_list, n_iter=100):
        if not isinstance(obs_list, list):
            obs_list = [obs_list]
        for _ in tqdm(range(n_iter)):
            old_loglik = self.score(obs_list)
            self._baum_welch(obs_list)
            new_loglik = self.score(obs_list)
            if abs(new_loglik - old_loglik) < 1e-6:
                break
    
    def score(self, obs):
        if isinstance(obs, list):
            return sum(self.score(o) for o in obs if len(o) >= 1)
        log_alpha = self._forward(obs)
        return logsumexp(log_alpha[-1])
    
    def predict(self, obs):
        """Viterbi in log space."""
        T = len(obs)
        N = self.n_states
        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        
        log_emit = np.zeros((T, N))
        for t in range(T):
            for j in range(N):
                log_emit[t, j] = np.log(self._gaussian_pdf(obs[t], self.means[j], self.variances[j]) + 1e-10)
        
        log_delta[0] = np.log(self.start_prob + 1e-10) + log_emit[0]
        psi[0] = 0
        
        for t in range(1, T):
            for j in range(N):
                log_probs = log_delta[t-1] + np.log(self.transmat[:, j] + 1e-10)
                psi[t, j] = np.argmax(log_probs)
                log_delta[t, j] = log_probs[psi[t, j]] + log_emit[t, j]
        
        best_last_state = np.argmax(log_delta[-1])
        path = np.zeros(T, dtype=int)
        path[T-1] = best_last_state
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path

    def sample(self, n_steps=1, current_state=None):
        """Sample future observations from the model, starting from a given state."""
        predictions = []
        if current_state is None:
            current_state = np.random.choice(range(self.n_states), p=self.start_prob)
        
        for _ in range(n_steps):
            # Transition to next state
            p = self.transmat[current_state]
            p = p / np.sum(p) if np.sum(p) > 0 else np.ones(self.n_states) / self.n_states  # Fix if sum 0
            if np.any(np.isnan(p)):
                p = np.nan_to_num(p, nan=1.0/self.n_states)
            current_state = np.random.choice(range(self.n_states), p=p)
            
            # Emit observation
            std = np.sqrt(self.variances[current_state])
            std[std == 0] = 1e-3  # Avoid zero std
            next_obs = np.random.normal(self.means[current_state], std)
            
            predictions.append(next_obs)
        
        return np.array(predictions)

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
        all_obs = []
        for sid in df['SID'].unique():
            storm = df[df['SID'] == sid].copy()
            if len(storm) < self.sequence_length + 1:
                continue
            features = np.column_stack([
                storm['LAT'].values,
                storm['LON'].values,
                storm['WMO_WIND'].fillna(50).values,
                storm['DIST2LAND'].fillna(200).values
            ]).astype(float)
            all_obs.append(features)
        print(f"✓ Tạo được {len(all_obs)} sequences")
        return all_obs

    def normalize(self, obs, fit=True):
        obs_norm = obs.copy()
        if fit:
            self.scaler_lat.fit(obs[:, 0].reshape(-1, 1))
            self.scaler_lon.fit(obs[:, 1].reshape(-1, 1))
            self.scaler_wind.fit(obs[:, 2].reshape(-1, 1))
            self.scaler_dist.fit(obs[:, 3].reshape(-1, 1))

        obs_norm[:, 0] = self.scaler_lat.transform(obs[:, 0].reshape(-1, 1)).reshape(-1)
        obs_norm[:, 1] = self.scaler_lon.transform(obs[:, 1].reshape(-1, 1)).reshape(-1)
        obs_norm[:, 2] = self.scaler_wind.transform(obs[:, 2].reshape(-1, 1)).reshape(-1)
        obs_norm[:, 3] = self.scaler_dist.transform(obs[:, 3].reshape(-1, 1)).reshape(-1)

        return obs_norm

    def denormalize_output(self, y_norm):
        y = y_norm.copy()
        y[:, 0] = self.scaler_lat.inverse_transform(y[:, 0].reshape(-1, 1)).reshape(-1)
        y[:, 1] = self.scaler_lon.inverse_transform(y[:, 1].reshape(-1, 1)).reshape(-1)
        return y

class TyphoonHMM:
    def __init__(self, n_states=8, n_features=4, random_state=42):
        self.n_states = n_states
        self.n_features = n_features
        self.model = GaussianHMM(n_states, n_features, random_state)

    def train(self, obs_list_norm, epochs=50):
        print("\n🎓 Bắt đầu huấn luyện HMM...")
        self.model.fit(obs_list_norm, n_iter=epochs)
        print("✓ Huấn luyện hoàn tất!")

    def predict_trajectory(self, initial_sequence_norm, n_steps=24):
        # Infer states on initial sequence
        states = self.model.predict(initial_sequence_norm)
        last_state = states[-1]
        
        # Sample future observations (full features)
        pred_norm = self.model.sample(n_steps, current_state=last_state)
        
        # Extract only LAT/LON (first 2 features)
        predictions = pred_norm[:, :2]  # Assuming [lat, lon, wind, dist]
        return predictions

    def save_model(self, model_path, scaler_path, processor):
        """Lưu model (.pkl) và scaler"""
        print(f"\n💾 Đang lưu model...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

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
        """Load model (.pkl) và scaler"""
        print(f"\n📥 Đang load model...")
        with open(model_path, 'rb') as f:
            hmm_model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)

        processor = TyphoonDataProcessor()
        processor.scaler_lat = scaler_data['scaler_lat']
        processor.scaler_lon = scaler_data['scaler_lon']
        processor.scaler_wind = scaler_data['scaler_wind']
        processor.scaler_dist = scaler_data['scaler_dist']

        model = TyphoonHMM()
        model.model = hmm_model

        print(f"✅ Load thành công!")
        return model, processor

def train_and_save_model(csv_path, model_path, scaler_path):
    """Train model và lưu"""
    print("="*80)
    print("🌀 TRAINING MÔ HÌNH DỰ ĐOÁN BÃO (HMM)")
    print("="*80)

    processor = TyphoonDataProcessor(sequence_length=8)
    df = processor.load_and_clean(csv_path)
    obs_list = processor.create_sequences(df)

    # Normalize: Fit on all data combined
    all_obs = np.vstack(obs_list)
    processor.normalize(all_obs, fit=True)
    # Normalize each sequence
    obs_list_norm = [processor.normalize(obs, fit=False) for obs in obs_list]

    model = TyphoonHMM(n_states=8, n_features=4)
    model.train(obs_list_norm, epochs=50)

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
        model: TyphoonHMM model
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

    X_raw = np.column_stack([lats, lons, winds, dists]).astype(float)

    # Normalize
    X_norm = processor.normalize(X_raw, fit=False)

    # Predict
    pred_norm = model.predict_trajectory(X_norm, n_steps=n_predict_steps)
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

    return result_df

if __name__ == "__main__":
    # ===== CẤU HÌNH ĐƯỜNG DẪN =====
    csv_path = 'ibtracs.WP.list.v04r00.csv'
    model_path = 'typhoon_hmm.pkl'  # ← .pkl
    scaler_path = 'typhoon_scaler.pkl'

    print("\n" + "="*80)
    print("🌀 HỆ THỐNG DỰ ĐOÁN BÃO - WORKFLOW ĐẦY ĐỦ (HMM)")
    print("="*80)

    # ===== BƯỚC 1: TRAIN MODEL (CHẠY 1 LẦN ĐẦU) =====
    # print("\n[Bước 1] TRAINING MODEL")
    # print("-" * 80)

    # model, processor, df = train_and_save_model(csv_path, model_path, scaler_path)

    # ===== BƯỚC 2: LOAD MODEL (CÁC LẦN SAU) =====
    print("\n[Bước 2] LOAD MODEL ĐÃ TRAIN")
    print("-" * 80)
    model, processor = TyphoonHMM.load_model(model_path, scaler_path)
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