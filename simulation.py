# simulation.py
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
            ax.plot(obs_lons, obs_lats, 'orange', linewidth=4, linestyle='-',
                   alpha=0.8, transform=ccrs.PlateCarree(), zorder=4, label='Bão (quan sát)')
            ax.scatter(obs_lons, obs_lats, c='orange', s=150, marker='o',
                      edgecolors='black', linewidths=1, transform=ccrs.PlateCarree(), zorder=5)

        # Bão dự đoán
        pred_until = min(frame, len(pred_typhoons))
        if pred_until > 0:
            pred_lats = [t.lat for t in pred_typhoons[:pred_until]]
            pred_lons = [t.lon for t in pred_typhoons[:pred_until]]

            ax.plot(pred_lons, pred_lats, 'red', linewidth=4,
                   alpha=0.8, transform=ccrs.PlateCarree(), zorder=6, label='Bão (dự báo)')
            ax.scatter(pred_lons, pred_lats, c='red', s=150, marker='o',
                      edgecolors='black', linewidths=2, transform=ccrs.PlateCarree(), zorder=7)

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
            ax.plot(boat_lons, boat_lats, color=color, linewidth=6,
                   transform=ccrs.PlateCarree(), zorder=8, label='Tàu cá')
            ax.scatter(boat_lons, boat_lats, c=color, s=200, marker='s',
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

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚢 HỆ THỐNG ĐIỀU HƯỚNG TÀU CÁ TRÁNH BÃO - PHIÊN BẢN HOÀN THIỆN")
    print("="*80)

    # ===== BƯỚC 1: LOAD DỰ ĐOÁN BÃO =====
    print("\n[Bước 1] LOAD DỰ ĐOÁN BÃO")
    print("-" * 80)

    typhoon_csv = input("📂 Nhập đường dẫn file CSV bão (từ prediction.py): ").strip()
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