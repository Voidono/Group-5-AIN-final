import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import random
from scipy.interpolate import make_interp_spline

# CẤU HÌNH
FPS = 15
SO_KHUNG_HINH = 120
GRID_SIZE = 0.5  
TONG_THOI_GIAN = 48 

def tao_du_lieu_bao_tu_dong(waypoints):
    
    x_points = np.array([p[0] for p in waypoints])
    y_points = np.array([p[1] for p in waypoints])
    
   
    t = np.arange(len(x_points))
    t_new = np.linspace(t.min(), t.max(), SO_KHUNG_HINH)
    
    try:
        spl_x = make_interp_spline(t, x_points, k=3)
        spl_y = make_interp_spline(t, y_points, k=3)
        track_x = spl_x(t_new)
        track_y = spl_y(t_new)
    except:
        track_x = np.linspace(x_points[0], x_points[-1], SO_KHUNG_HINH)
        track_y = np.linspace(y_points[0], y_points[-1], SO_KHUNG_HINH)

    # Tính toán cường độ 
    x_norm = np.linspace(-1, 1, SO_KHUNG_HINH)
    intensity_curve = 1 - x_norm**2
    
    track_s = 0.3 + (intensity_curve * 0.9)  
    track_c = intensity_curve                
    
    return track_x, track_y, track_s, track_c

def get_storm_color(value):
    if value < 0.3: return '#FFD700'   
    elif value < 0.6: return '#FF8C00' 
    elif value < 0.85: return '#FF0000'
    else: return '#800080'             

def run_typhoon_simulation(ten_bao, waypoints, output_file):
    print(f"--- ĐANG KHỞI TẠO: {ten_bao.upper()} ---")
    
    # Map
    url_world = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    try:
        print("⏳ Đang tải dữ liệu bản đồ...")
        world = gpd.read_file(url_world)
    except:
        print("❌ Lỗi tải map!")
        return

    vietnam = world[world['NAME'] == "Vietnam"]
    other_countries = world[world['NAME'] != "Vietnam"]

    
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_facecolor('#a6cce3')

    other_countries.plot(ax=ax, color='#f0f0f0', edgecolor='white')
    vietnam.plot(ax=ax, color='#FFD700', edgecolor='#d32f2f', linewidth=1.2)

    # CẢNG BIỂN
    danh_sach_cang = [
        {"name": "Hải Phòng", "lon": 106.7, "lat": 20.85},
        {"name": "Đà Nẵng",   "lon": 108.22, "lat": 16.1},
        {"name": "Quy Nhơn",  "lon": 109.22, "lat": 13.76},
        {"name": "Cam Ranh",  "lon": 109.15, "lat": 11.9},
        {"name": "Vũng Tàu",  "lon": 107.08, "lat": 10.35}
    ]
    for cang in danh_sach_cang:
        plt.text(cang["lon"], cang["lat"], "⚓", fontsize=12, color='#00008B', ha='center', va='center', zorder=7)
        plt.text(cang["lon"]+0.2, cang["lat"], cang["name"], fontsize=9, weight='bold', color='black', 
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=8)

    # ĐẢO
    plt.text(111.2, 16.5, "★", fontsize=20, color='red', ha='center', va='center', zorder=6)
    plt.text(111.6, 16.5, "Hoàng Sa (VN)", color='red', weight='bold', fontsize=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=7)
    plt.text(114.5, 10.0, "★", fontsize=20, color='red', ha='center', va='center', zorder=6)
    plt.text(114.9, 10.0, "Trường Sa (VN)", color='red', weight='bold', fontsize=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=7)


    ax.set_xlim(102, 120)
    ax.set_ylim(6, 23)
    
    ax.xaxis.set_major_locator(mticker.MultipleLocator(GRID_SIZE))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(GRID_SIZE))
    plt.grid(True, linestyle='--', alpha=0.3, color='black')

    plt.title(f"MÔ PHỎNG: {ten_bao.upper()}", fontsize=18, weight='bold', pad=15, color='#8B0000')
    plt.xlabel("Kinh độ", fontsize=10)
    plt.ylabel("Vĩ độ", fontsize=10)

    # TÍNH TOÁN DỮ LIỆU BÃO 
    track_x, track_y, track_s, track_c = tao_du_lieu_bao_tu_dong(waypoints)

    idx_a = int(len(track_x) * 0.25) 
    ax_pos = track_x[idx_a] + random.uniform(-0.8, 0.8)
    ay_pos = track_y[idx_a] + random.uniform(-0.8, 0.8)

    idx_b = int(len(track_x) * 0.75) 
    bx_pos = track_x[idx_b] + random.uniform(-0.8, 0.8)
    by_pos = track_y[idx_b] + random.uniform(-0.8, 0.8)

    # Vẽ A và B
    plt.plot(ax_pos, ay_pos, marker='s', color='green', markersize=10, markeredgecolor='white', zorder=8, label='Điểm A (Tàu)')
    plt.text(ax_pos + 0.2, ay_pos, "Điểm A", fontsize=11, weight='bold', color='green', 
             path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=9)

    plt.plot(bx_pos, by_pos, marker='s', color='purple', markersize=10, markeredgecolor='white', zorder=8, label='Điểm B (Tàu)')
    plt.text(bx_pos + 0.2, by_pos, "Điểm B", fontsize=11, weight='bold', color='purple', 
             path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=9)

    
    
    
    storm_circle = plt.Circle((0, 0), 0.1, color='red', alpha=0.6, zorder=10)
    ax.add_patch(storm_circle)
    storm_eye, = ax.plot([], [], 'yo', markersize=4, markeredgecolor='black', zorder=11)
    
    # LINE
    ax.plot(track_x, track_y, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=5)

    # TIME
    time_box = ax.text(0.02, 0.95, "00H", transform=ax.transAxes, fontsize=12, weight='bold', color='white',
                       bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'), va='top')

    print(f"🎬 Đang render GIF '{output_file}'...")

    def update(frame):
        idx = frame if frame < len(track_x) else len(track_x) - 1
        
        cx, cy = track_x[idx], track_y[idx]
        size = track_s[idx]
        color_val = track_c[idx]
        
        # Update Bão
        storm_circle.center = (cx, cy)
        storm_circle.set_radius(size)
        storm_circle.set_color(get_storm_color(color_val))
        storm_eye.set_data([cx], [cy])
        
        
        current_hour = int((frame / len(track_x)) * TONG_THOI_GIAN)
        
        if color_val < 0.3: cap_bao = "8-9"
        elif color_val < 0.6: cap_bao = "10-12"
        elif color_val < 0.85: cap_bao = "13-15"
        else: cap_bao = "16+"
        
        time_box.set_text(f"Thời gian: {current_hour:02d}H / {TONG_THOI_GIAN}H\nCấp độ: {cap_bao}")
        time_box.set_bbox(dict(facecolor=get_storm_color(color_val), alpha=0.8, boxstyle='round,pad=0.5'))

        return storm_circle, storm_eye, time_box

    ani = animation.FuncAnimation(fig, update, frames=len(track_x), interval=80, blit=False)
    writer = animation.PillowWriter(fps=FPS)
    ani.save(output_file, writer=writer)
    print(f"✅ ĐÃ XONG! File '{output_file}' đã hoàn thành.")
    plt.close()

# PATH TYPHOON

path_typhoon= [
    (119.0, 11.5), 
    (115.0, 12.8), 
    (112.0, 13.2), 
    (109.2, 13.4),
    (106.5, 13.6)
]
run_typhoon_simulation("typhoon", path_typhoon, "typhoon.gif")
