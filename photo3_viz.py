import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import numpy.ma as ma
from sklearn.cluster import DBSCAN
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

# -------------------------
# 1) Point Cloud 로드 및 전처리
# -------------------------
file_path = r"D:\GEODB\2024 2B\Photogrammetry\FINAL\MeshroomCache\ConvertSfMFormat\a5e345d1be8289fef3b413e4afa8f1743bbc7af5\sfm.ply"
pcd = o3d.io.read_point_cloud(file_path)
print("[INFO] Loaded point cloud with {} points.".format(len(pcd.points)))

# 다운샘플링
voxel_size = 0.05
down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30
))
print("[INFO] Downsampled point cloud has {} points.".format(len(down_pcd.points)))

# -------------------------
# 2) RANSAC으로 바닥면 검출
# -------------------------
def detect_plane_ransac(pcd, dist_threshold=0.02, ransac_n=3, num_iter=1000):
    points = np.asarray(pcd.points)
    best_plane = None
    best_inliers_idx = []
    
    for _ in range(num_iter):
        idx = np.random.choice(len(points), size=ransac_n, replace=False)
        p1, p2, p3 = points[idx]
        
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        
        dist = np.abs(np.dot(points, normal) + d)
        inliers_idx = np.where(dist <= dist_threshold)[0]
        
        if len(inliers_idx) > len(best_inliers_idx):
            best_plane = (normal, d)
            best_inliers_idx = inliers_idx
    
    return best_plane, best_inliers_idx

plane_model, inliers = detect_plane_ransac(down_pcd)
[a, b, c] = plane_model[0]
d = plane_model[1]
print(f"[INFO] Detected plane: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

inlier_cloud = down_pcd.select_by_index(inliers)
print(f"[INFO] Floor inliers: {len(inlier_cloud.points)} points")

# -------------------------
# 3) XY 평면으로 정렬
# -------------------------
normal = np.array([a, b, c])
normal /= np.linalg.norm(normal)
z_axis = np.array([0, 0, 1])
rotation_axis = np.cross(normal, z_axis)
rotation_angle = np.arccos(np.dot(normal, z_axis))

if np.linalg.norm(rotation_axis) > 1e-6:
    rotation_axis /= np.linalg.norm(rotation_axis)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angle * rotation_axis)
else:
    R = np.eye(3)

aligned_inliers = o3d.geometry.PointCloud()
aligned_inliers.points = o3d.utility.Vector3dVector(np.asarray(inlier_cloud.points))
if inlier_cloud.has_colors():
    aligned_inliers.colors = o3d.utility.Vector3dVector(np.asarray(inlier_cloud.colors))
if inlier_cloud.has_normals():
    aligned_inliers.normals = o3d.utility.Vector3dVector(np.asarray(inlier_cloud.normals))
aligned_inliers.rotate(R, center=(0,0,0))

# -------------------------
# 4) 2D 투영 및 Rasterization
# -------------------------
points_3d = np.asarray(aligned_inliers.points)
points_2d = points_3d[:, :2]

def process_points(eps_cm, min_samples, resolution_mm, kernel_size, iterations):
    # DBSCAN으로 노이즈 제거
    points_2d_scaled = points_2d * 100  # 미터를 센티미터로 변환
    clustering = DBSCAN(eps=eps_cm, min_samples=min_samples).fit(points_2d_scaled)
    labels = clustering.labels_

    # 가장 큰 클러스터와 그와 비슷한 크기의 클러스터들 모두 포함
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) > 0:
        max_count = np.max(counts)
        significant_clusters = unique_labels[counts > max_count * 0.2]
        mask = np.isin(labels, significant_clusters)
        points_2d_clean = points_2d[mask]
    else:
        points_2d_clean = points_2d

    # 격자 해상도 설정
    resolution = resolution_mm / 1000.0  # mm를 m로 변환
    x_min, y_min = points_2d_clean.min(axis=0)
    x_max, y_max = points_2d_clean.max(axis=0)
    nx = int((x_max - x_min) / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1

    # 포인트를 격자에 매핑
    grid = np.zeros((ny, nx), dtype=np.uint8)
    for point in points_2d_clean:
        x = int((point[0] - x_min) / resolution)
        y = int((point[1] - y_min) / resolution)
        if 0 <= x < nx and 0 <= y < ny:
            grid[y, x] = 255

    # 모폴로지 연산
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # DILATE로 점들을 확장
    grid = cv2.dilate(grid, kernel, iterations=iterations)
    
    # 가우시안 블러
    grid_smooth = cv2.GaussianBlur(grid, (3,3), 0.5)
    
    # Otsu 이진화
    _, binary = cv2.threshold(grid_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # CLOSE로 구멍 메우기
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # OPEN으로 노이즈 제거
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Contour 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, binary, x_min, y_min, x_max, y_max, resolution

    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Contour 단순화
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 격자 좌표를 실제 좌표로 변환
    real_contour = []
    for point in approx_contour[:, 0, :]:
        x = point[0] * resolution + x_min
        y = point[1] * resolution + y_min
        real_contour.append([x, y])
    real_contour = np.array(real_contour)

    # 첫 점을 마지막에 추가하여 폐곡선 만들기
    if not np.array_equal(real_contour[0], real_contour[-1]):
        real_contour = np.vstack([real_contour, real_contour[0]])

    return real_contour, points_2d_clean, binary, x_min, y_min, x_max, y_max, resolution

# -------------------------
# 5) 인터랙티브 시각화
# -------------------------
plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize=(16, 10))

# 메인 플롯과 서브플롯을 위한 그리드 설정
gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1])  # 상하 비율 조정
gs.update(left=0.08, right=0.92, bottom=0.08, top=0.95, wspace=0.25, hspace=0.2)

# 메인 플롯
ax_main = fig.add_subplot(gs[0, :])
ax_binary = fig.add_subplot(gs[1, 0])
ax_controls = fig.add_subplot(gs[1, 1])
ax_controls.set_visible(False)  # 컨트롤 영역 숨기기

# 초기 파라미터
eps_cm = 5
min_samples = 2
resolution_mm = 2
kernel_size = 3
iterations = 2

# 초기 처리
real_contour, points_2d_clean, binary, x_min, y_min, x_max, y_max, resolution = process_points(
    eps_cm, min_samples, resolution_mm, kernel_size, iterations)

# 메인 플롯 업데이트 함수
def update_plot():
    ax_main.clear()
    ax_binary.clear()
    
    # 배경 스타일 설정
    ax_main.set_facecolor('#f0f0f0')
    ax_binary.set_facecolor('#f0f0f0')
    
    # 포인트와 외곽선 그리기
    scatter = ax_main.scatter(points_2d[:, 0], points_2d[:, 1], 
                            c='#2c3e50', s=1, alpha=0.3, label='Floor Points')
    
    if real_contour is not None:
        line = ax_main.plot(real_contour[:, 0], real_contour[:, 1], 
                          color='#e74c3c', linewidth=2, 
                          label='Floor Boundary', zorder=5)
        
        # 면적 계산
        polygon = Polygon(real_contour)
        area = polygon.area
        perimeter = polygon.length
        
        # 제목 스타일링
        title = f"Floor Plan Analysis\n"
        title += f"Area: {area:.2f}m² | Perimeter: {perimeter:.2f}m"
        ax_main.set_title(title, pad=20, fontsize=12, fontweight='bold')
        
        # 데이터 범위 계산
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 여백 추가 (10%)
        margin = 0.1
        x_margin = x_range * margin
        y_margin = y_range * margin
        
        # 축 범위 설정
        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 치수선 추가
        for i in range(len(real_contour)-1):
            p1 = real_contour[i]
            p2 = real_contour[i+1]
            length = np.sqrt(np.sum((p2 - p1)**2))
            
            if length > 0.3:  # 30cm 이상인 선분만 표시
                direction = p2 - p1
                angle = np.arctan2(direction[1], direction[0])
                
                # 치수선 오프셋과 스타일
                offset = min(x_range, y_range) * 0.05  # 상대적인 오프셋
                normal = np.array([-np.sin(angle), np.cos(angle)])
                offset_p1 = p1 + normal * offset
                offset_p2 = p2 + normal * offset
                
                # 치수선 화살표
                arrow_props = dict(arrowstyle='<->', color='#7f8c8d', 
                                 linewidth=1, shrinkA=0, shrinkB=0)
                ax_main.annotate('', xy=offset_p1, xytext=offset_p2,
                               arrowprops=arrow_props)
                
                # 치수 텍스트
                text_pos = (offset_p1 + offset_p2) / 2 + normal * offset * 0.3
                text_angle = np.degrees(angle)
                if text_angle > 90 or text_angle < -90:
                    text_angle += 180
                ax_main.text(text_pos[0], text_pos[1], f'{length:.2f}m',
                           ha='center', va='center', rotation=text_angle,
                           fontsize=8, color='#34495e',
                           bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.8, pad=1))
    
    # 축 레이블 스타일링
    ax_main.set_xlabel("X (meters)", fontsize=10, labelpad=10)
    ax_main.set_ylabel("Y (meters)", fontsize=10, labelpad=10)
    ax_main.axis('equal')
    ax_main.grid(True, linestyle='--', alpha=0.7)
    ax_main.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # 이진 이미지 표시
    if binary is not None:
        ax_binary.imshow(binary, origin='lower', 
                        extent=[x_min, x_max, y_min, y_max],
                        cmap='Greys', alpha=0.8)
        ax_binary.set_title("Binary Representation", 
                          fontsize=10, fontweight='bold', pad=10)
        ax_binary.axis('equal')
        ax_binary.grid(False)
        
        # 이진 이미지의 축 범위도 동일하게 설정
        ax_binary.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_binary.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 스케일 바 추가 (상대적인 크기로 조정)
    bar_length = min(x_range, y_range) * 0.2  # 도면 크기의 20%
    bar_x = x_min + x_margin
    bar_y = y_min + y_margin
    ax_main.plot([bar_x, bar_x + bar_length], [bar_y, bar_y], 
                'k-', linewidth=2, zorder=10)
    ax_main.text(bar_x + bar_length/2, bar_y - y_range*0.02, f'{bar_length:.1f}m',
                ha='center', va='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none',
                         alpha=0.8, pad=2))
    
    fig.canvas.draw_idle()

# 슬라이더 스타일 설정
slider_props = {
    'color': '#3498db',
    'alpha': 0.6
}
text_color = '#2c3e50'

# 슬라이더 생성 (위치 조정)
slider_width = 0.25
slider_x = 0.62
slider_y_start = 0.05
slider_spacing = 0.045

ax_eps = plt.axes([slider_x, slider_y_start + 4*slider_spacing, slider_width, 0.02])
ax_min_samples = plt.axes([slider_x, slider_y_start + 3*slider_spacing, slider_width, 0.02])
ax_resolution = plt.axes([slider_x, slider_y_start + 2*slider_spacing, slider_width, 0.02])
ax_kernel = plt.axes([slider_x, slider_y_start + slider_spacing, slider_width, 0.02])
ax_iterations = plt.axes([slider_x, slider_y_start, slider_width, 0.02])

# 슬라이더 생성
s_eps = Slider(ax_eps, 'DBSCAN eps (cm)', 1, 20, valinit=eps_cm, **slider_props)
s_min_samples = Slider(ax_min_samples, 'Min. samples', 1, 10, 
                      valinit=min_samples, valstep=1, **slider_props)
s_resolution = Slider(ax_resolution, 'Resolution (mm)', 0.5, 10, 
                     valinit=resolution_mm, **slider_props)
s_kernel = Slider(ax_kernel, 'Kernel size', 3, 11, 
                 valinit=kernel_size, valstep=2, **slider_props)
s_iterations = Slider(ax_iterations, 'Iterations', 1, 5, 
                     valinit=iterations, valstep=1, **slider_props)

# 슬라이더 레이블 색상 설정
for slider in [s_eps, s_min_samples, s_resolution, s_kernel, s_iterations]:
    slider.label.set_color(text_color)
    slider.valtext.set_color(text_color)

# 슬라이더 업데이트 함수
def update(val):
    global real_contour, points_2d_clean, binary, x_min, y_min, x_max, y_max, resolution
    real_contour, points_2d_clean, binary, x_min, y_min, x_max, y_max, resolution = process_points(
        s_eps.val, int(s_min_samples.val), s_resolution.val, 
        int(s_kernel.val), int(s_iterations.val))
    update_plot()

s_eps.on_changed(update)
s_min_samples.on_changed(update)
s_resolution.on_changed(update)
s_kernel.on_changed(update)
s_iterations.on_changed(update)

# 초기 플롯
update_plot()

# 창 제목 설정
fig.canvas.manager.set_window_title('Floor Plan Analysis Tool')

plt.show()
