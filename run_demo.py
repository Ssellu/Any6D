import os
import trimesh
import yaml
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time
from collections import defaultdict

from PIL import Image
from estimater import Any6D

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()

class InferenceTimer:
    """순수 추론 시간 측정을 위한 타이머 클래스"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_section = None
        self.section_start = None

    def start(self, section_name):
        """특정 섹션의 시간 측정 시작"""
        self.current_section = section_name
        self.section_start = time.time()

    def end(self):
        """현재 섹션의 시간 측정 종료"""
        if self.current_section and self.section_start:
            elapsed = time.time() - self.section_start
            self.timings[self.current_section].append(elapsed)
            self.current_section = None
            self.section_start = None
            return elapsed
        return 0

    def get_summary(self):
        """측정된 시간들의 요약 통계 반환"""
        summary = {}
        for section, times in self.timings.items():
            summary[section] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times)
            }
        return summary

    def print_report(self, target_fps=30):
        """상세한 성능 리포트 출력"""
        print("\n" + "="*70)
        print("⏱️  INFERENCE TIME ANALYSIS (Excluding Visualization)")
        print("="*70)

        summary = self.get_summary()
        total_inference_time = 0

        print(f"\n{'Section':<35} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min/Max (ms)'}")
        print("-" * 70)

        for section, stats in summary.items():
            mean_ms = stats['mean'] * 1000
            std_ms = stats['std'] * 1000
            min_ms = stats['min'] * 1000
            max_ms = stats['max'] * 1000
            total_inference_time += stats['mean']

            print(f"{section:<35} {mean_ms:>10.2f}   {std_ms:>10.2f}   {min_ms:>6.2f}/{max_ms:<6.2f}")

        print("-" * 70)
        print(f"{'TOTAL INFERENCE TIME':<35} {total_inference_time*1000:>10.2f} ms")
        print(f"{'ESTIMATED FPS':<35} {1.0/total_inference_time:>10.2f}")

        # 실시간 처리 가능 여부 분석
        target_frame_time = 1.0 / target_fps
        print("\n" + "="*70)
        print(f"🎯 REAL-TIME CAPABILITY ANALYSIS (Target: {target_fps} FPS)")
        print("="*70)
        print(f"Target frame time:     {target_frame_time*1000:>8.2f} ms")
        print(f"Actual inference time: {total_inference_time*1000:>8.2f} ms")

        if total_inference_time < target_frame_time:
            margin = (target_frame_time - total_inference_time) * 1000
            print(f"Status: ✅ REAL-TIME CAPABLE (margin: +{margin:.2f} ms)")
        else:
            deficit = (total_inference_time - target_frame_time) * 1000
            max_achievable_fps = 1.0 / total_inference_time
            print(f"Status: ❌ NOT REAL-TIME (deficit: -{deficit:.2f} ms)")
            print(f"Maximum achievable FPS: {max_achievable_fps:.2f}")

        print("="*70 + "\n")

        return total_inference_time

def draw_3d_bounding_box(image, mesh, pose, K):
    """
    3D bounding box를 이미지에 그리는 함수
    SAM-6D 스타일의 oriented bounding box 시각화

    Args:
        image: RGB 이미지 (numpy array)
        mesh: trimesh 객체
        pose: 4x4 pose matrix (rotation + translation)
        K: 3x3 camera intrinsic matrix

    Returns:
        annotated_image: 3D bounding box가 그려진 이미지
    """
    result_img = image.copy()

    # 1. Mesh의 bounding box 코너 구하기 (local coordinates)
    bbox_min = mesh.bounds[0]  # (x_min, y_min, z_min)
    bbox_max = mesh.bounds[1]  # (x_max, y_max, z_max)

    # 8개의 코너 점 정의 (local coordinate system)
    corners_3d = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0
        [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1
        [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2
        [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3
        [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4
        [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5
        [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6
        [bbox_min[0], bbox_max[1], bbox_max[2]],  # 7
    ])

    # 2. World coordinates로 변환 (pose 적용)
    R = pose[:3, :3]
    t = pose[:3, 3]
    corners_world = np.dot(corners_3d, R.T) + t

    # 3. 카메라 좌표계로 투영
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    corners_2d = []
    for corner in corners_world:
        if corner[2] > 0:  # 카메라 앞에 있는 경우만
            x = int(corner[0] * fx / corner[2] + cx)
            y = int(corner[1] * fy / corner[2] + cy)
            corners_2d.append([x, y])
        else:
            corners_2d.append(None)

    # 4. Bounding box의 선 그리기
    # 정의: 아래 면 (0,1,2,3), 위 면 (4,5,6,7)
    edges = [
        # 아래 면
        (0, 1), (1, 2), (2, 3), (3, 0),
        # 위 면
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 수직 연결선
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # 앞면 (카메라 쪽) 강조를 위한 색상 구분
    front_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 아래 면
    vertical_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    back_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]  # 위 면

    # 선 그리기
    for edge in edges:
        pt1, pt2 = corners_2d[edge[0]], corners_2d[edge[1]]
        if pt1 is not None and pt2 is not None:
            # 이미지 범위 내에 있는지 확인
            if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):

                # 선 종류에 따라 색상과 두께 구분
                if edge in front_edges:
                    color = (0, 255, 0)  # 초록색 - 앞면 (강조)
                    thickness = 3
                elif edge in vertical_edges:
                    color = (255, 255, 0)  # 노란색 - 수직선
                    thickness = 2
                else:  # back_edges
                    color = (0, 255, 255)  # 시안색 - 뒷면
                    thickness = 2

                cv2.line(result_img, tuple(pt1), tuple(pt2), color, thickness)

    # 5. 코너 점 표시 (선택적)
    for i, corner in enumerate(corners_2d):
        if corner is not None:
            if 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]:
                # 앞면 코너는 더 크게
                radius = 5 if i < 4 else 3
                color = (255, 0, 0) if i < 4 else (0, 0, 255)
                cv2.circle(result_img, tuple(corner), radius, color, -1)

    return result_img

def check_scale_mismatch(mesh, depth, mask, K):
    """스케일 불일치 체크 및 자동 수정"""
    print("\n=== 스케일 분석 ===")
    
    # 실제 객체의 depth 범위
    masked_depth = depth[mask]
    if len(masked_depth) == 0:
        print("❌ No valid depth points in mask!")
        return mesh
    
    real_depth_range = np.max(masked_depth) - np.min(masked_depth)
    real_depth_mean = np.mean(masked_depth)
    
    # CAD 모델의 크기
    mesh_bbox = mesh.bounds
    mesh_size = np.max(mesh_bbox[1] - mesh_bbox[0])
    mesh_center = mesh.centroid
    
    print(f"Real depth range: {real_depth_range:.3f}m (mean: {real_depth_mean:.3f}m)")
    print(f"Mesh size: {mesh_size:.3f}m")
    print(f"Mesh center: {mesh_center}")
    print(f"Mesh bounds: {mesh_bbox}")
    
    # 스케일 비율 계산
    if mesh_size > 0:
        scale_ratio = real_depth_range / mesh_size
        print(f"Initial scale ratio: {scale_ratio:.3f}")
        
        # 스케일 조정이 필요한 경우
        if abs(scale_ratio - 1.0) > 0.3:  # 30% 이상 차이
            print(f"⚠️ Scale mismatch detected! Adjusting by factor: {scale_ratio:.3f}")
            
            # 메시 스케일 조정
            mesh_scaled = mesh.copy()
            mesh_scaled.vertices *= scale_ratio
            
            # 조정 후 정보
            new_bbox = mesh_scaled.bounds
            new_size = np.max(new_bbox[1] - new_bbox[0])
            print(f"Adjusted mesh size: {new_size:.3f}m")
            
            return mesh_scaled
        else:
            print("✅ Scale is reasonable")
    
    return mesh

def align_mesh_coordinate_system(mesh, depth, mask, K, save_path):
    """좌표계 정렬 최적화"""
    print("\n=== 좌표계 정렬 분석 ===")
    
    # 실제 3D 포인트 생성
    h, w = depth.shape
    y, x = np.meshgrid(range(h), range(w), indexing='ij')
    
    mask_points = mask & (depth > 0)
    if np.sum(mask_points) < 100:
        print("⚠️ Too few valid points for alignment analysis")
        return mesh
    
    # 3D 포인트 계산
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = depth[mask_points]
    x_3d = (x[mask_points] - cx) * z / fx
    y_3d = (y[mask_points] - cy) * z / fy
    
    real_points = np.column_stack([x_3d, -y_3d, -z])  # 좌표계 변환
    real_center = np.mean(real_points, axis=0)
    real_extent = np.max(real_points, axis=0) - np.min(real_points, axis=0)
    
    print(f"Real point cloud center: {real_center}")
    print(f"Real point cloud extent: {real_extent}")
    
    # 다양한 좌표계 변환 시도
    transformations = [
        ("Original", np.eye(4)),
        ("Rotate_X_90", trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])),
        ("Rotate_X_-90", trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])),
        ("Rotate_Y_90", trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])),
        ("Rotate_Y_-90", trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])),
        ("Rotate_Z_90", trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1])),
        ("Rotate_Z_-90", trimesh.transformations.rotation_matrix(-np.pi/2, [0,0,1])),
        ("Flip_Z", np.diag([1,1,-1,1])),
        ("Flip_Y", np.diag([1,-1,1,1])),
        ("Flip_X", np.diag([-1,1,1,1])),
    ]
    
    best_mesh = mesh
    best_score = float('inf')
    best_name = "Original"
    
    print("\nTesting coordinate alignments:")
    for name, transform in transformations:
        test_mesh = mesh.copy()
        test_mesh.apply_transform(transform)
        
        # 중심점 맞추기
        mesh_center = test_mesh.centroid
        test_mesh.vertices += (real_center - mesh_center)
        
        # 점들 샘플링 (성능을 위해)
        mesh_points = test_mesh.vertices[::max(1, len(test_mesh.vertices)//1000)]
        real_points_sampled = real_points[::max(1, len(real_points)//1000)]
        
        # 가장 가까운 점들 간의 거리 계산
        if len(mesh_points) > 0 and len(real_points_sampled) > 0:
            distances = cdist(mesh_points, real_points_sampled)
            score = np.mean(np.min(distances, axis=1))
            
            print(f"  {name}: score={score:.4f}, center={test_mesh.centroid}")
            
            if score < best_score:
                best_score = score
                best_mesh = test_mesh
                best_name = name
    
    print(f"✅ Best alignment: {best_name} (score: {best_score:.4f})")
    
    # 시각화 저장
    fig = plt.figure(figsize=(15, 5))
    
    # 원본 mesh
    ax1 = fig.add_subplot(131, projection='3d')
    orig_v = mesh.vertices[::100]
    ax1.scatter(orig_v[:, 0], orig_v[:, 1], orig_v[:, 2], c='red', s=1, alpha=0.6, label='Original Mesh')
    ax1.set_title('Original Mesh')
    ax1.legend()
    
    # 실제 포인트클라우드
    ax2 = fig.add_subplot(132, projection='3d')
    real_sampled = real_points[::50]
    ax2.scatter(real_sampled[:, 0], real_sampled[:, 1], real_sampled[:, 2], c='blue', s=1, alpha=0.6, label='Real Points')
    ax2.set_title('Real Point Cloud')
    ax2.legend()
    
    # 정렬된 mesh + 실제 포인트
    ax3 = fig.add_subplot(133, projection='3d')
    best_v = best_mesh.vertices[::100]
    ax3.scatter(best_v[:, 0], best_v[:, 1], best_v[:, 2], c='red', s=1, alpha=0.6, label='Aligned Mesh')
    ax3.scatter(real_sampled[:, 0], real_sampled[:, 1], real_sampled[:, 2], c='blue', s=1, alpha=0.4, label='Real Points')
    ax3.set_title(f'Best Alignment: {best_name}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'coordinate_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_mesh

def validate_mask_quality(color, depth, mask, save_path):
    """마스크 품질 검증 및 개선"""
    print("\n=== 마스크 품질 분석 ===")
    
    mask_area = np.sum(mask)
    image_area = mask.shape[0] * mask.shape[1]
    mask_ratio = mask_area / image_area
    
    print(f"Mask coverage: {mask_ratio:.3f} ({mask_area} pixels)")
    
    # 마스크 연결성 확인
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours: {len(contours)}")
    
    if len(contours) > 1:
        print("⚠️ Multiple contours detected - cleaning mask...")
        # 가장 큰 컨투어만 유지
        largest_contour = max(contours, key=cv2.contourArea)
        mask_cleaned = np.zeros_like(mask)
        cv2.fillPoly(mask_cleaned, [largest_contour], True)
        mask = mask_cleaned.astype(bool)
        print("✅ Mask cleaned")
        
    # 마스크된 영역의 depth 유효성
    masked_depth = depth[mask]
    valid_depth_ratio = np.sum(masked_depth > 0) / len(masked_depth) if len(masked_depth) > 0 else 0
    print(f"Valid depth in mask: {valid_depth_ratio:.3f}")
    
    if valid_depth_ratio < 0.7:
        print("⚠️ Poor depth coverage in masked region")
    
    # 마스크 품질 시각화
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(color)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    mask_overlay = color.copy()
    mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[1].imshow(mask_overlay)
    axes[1].set_title(f'Mask (coverage: {mask_ratio:.3f})')
    axes[1].axis('off')
    
    masked_depth_vis = np.zeros_like(depth)
    masked_depth_vis[mask] = masked_depth
    im = axes[2].imshow(masked_depth_vis, cmap='jet')
    axes[2].set_title(f'Masked Depth (valid: {valid_depth_ratio:.3f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    # 히스토그램
    axes[3].hist(masked_depth[masked_depth > 0], bins=50, alpha=0.7, edgecolor='black')
    axes[3].set_title('Depth Histogram')
    axes[3].set_xlabel('Depth (m)')
    axes[3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mask_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return mask

def validate_camera_calibration(intrinsic, color_shape):
    """카메라 캘리브레이션 검증"""
    print("\n=== 카메라 캘리브레이션 분석 ===")
    
    h, w = color_shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    print(f"Image size: {w}x{h}")
    print(f"Focal length: fx={fx:.1f}, fy={fy:.1f}")
    print(f"Principal point: cx={cx:.1f}, cy={cy:.1f}")
    
    # 합리적인 값인지 체크
    warnings = []
    if abs(fx - fy) / max(fx, fy) > 0.05:
        warnings.append("Significant focal length difference (fx != fy)")
    
    if abs(cx - w/2) > w*0.1 or abs(cy - h/2) > h*0.1:
        warnings.append("Principal point far from image center")
    
    if fx < w*0.5 or fx > w*2:
        warnings.append("Unusual focal length value")
    
    if len(warnings) > 0:
        print("⚠️ Calibration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Camera calibration looks reasonable")
    
    return len(warnings) == 0

def optimize_resolution(color, depth, mask, target_size=640):
    """해상도 최적화"""
    print("\n=== 해상도 최적화 ===")
    
    original_h, original_w = color.shape[:2]
    print(f"Original resolution: {original_w}x{original_h}")
    
    if max(original_h, original_w) != target_size:
        # 비율 유지하면서 리사이즈
        scale = target_size / max(original_h, original_w)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        color_resized = cv2.resize(color, (new_w, new_h))
        depth_resized = cv2.resize(depth, (new_w, new_h))
        mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h)) > 0.5
        
        print(f"Resized to: {new_w}x{new_h} (scale: {scale:.3f})")
        return color_resized, depth_resized, mask_resized, scale
    else:
        print("✅ Resolution is optimal")
    
    return color, depth, mask, 1.0

def visualize_with_depth(color, depth, mask, mesh, pose, K, save_path):
    """Depth 정보를 포함한 완전한 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 원본 RGB 이미지
    axes[0, 0].imshow(color)
    axes[0, 0].set_title('Original RGB Image')
    axes[0, 0].axis('off')

    # 2. Depth 이미지에 3D Bounding Box 오버레이
    depth_colored = cv2.applyColorMap((depth * 255 / depth.max()).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    # 3D bounding box 그리기
    depth_with_bbox = draw_3d_bounding_box(depth_colored, mesh, pose, K)
    axes[0, 1].imshow(depth_with_bbox)
    axes[0, 1].set_title('Depth with 3D Bounding Box')
    axes[0, 1].axis('off')
    
    # 3. RGB 이미지에 3D Bounding Box 오버레이
    rgb_with_bbox = draw_3d_bounding_box(color, mesh, pose, K)
    axes[0, 2].imshow(rgb_with_bbox)
    axes[0, 2].set_title('RGB with 3D Bounding Box')
    axes[0, 2].axis('off')
    
    # 4. Mesh 투영 결과
    vertices_transformed = np.dot(mesh.vertices, pose[:3, :3].T) + pose[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_2d = vertices_transformed[:, 0] * fx / vertices_transformed[:, 2] + cx
    y_2d = vertices_transformed[:, 1] * fy / vertices_transformed[:, 2] + cy
    depth_projected = vertices_transformed[:, 2]
    
    # 유효한 점들만 선택
    valid = (x_2d >= 0) & (x_2d < color.shape[1]) & \
            (y_2d >= 0) & (y_2d < color.shape[0]) & \
            (depth_projected > 0)
    
    mesh_overlay = color.copy()
    if np.sum(valid) > 0:
        # Depth에 따른 색상 변화로 mesh 표시
        for i in range(0, len(x_2d[valid]), 20):
            depth_val = depth_projected[valid][i]
            # Depth에 따른 색상 (가까우면 빨강, 멀면 파랑)
            color_intensity = max(0, min(1, (depth_val - 0.3) / 1.0))
            point_color = (int(255 * (1 - color_intensity)), int(255 * color_intensity), 0)
            cv2.circle(mesh_overlay, (int(x_2d[valid][i]), int(y_2d[valid][i])), 3, point_color, -1)
    
    axes[1, 0].imshow(mesh_overlay)
    axes[1, 0].set_title('Mesh Projection (Depth-colored)')
    axes[1, 0].axis('off')
    
    # 5. 간단한 OpenCV 오버레이
    simple_overlay = color.copy()
    if np.sum(valid) > 0:
        # 점들 표시 (샘플링)
        for i in range(0, len(x_2d[valid]), 50):
            cv2.circle(simple_overlay, (int(x_2d[valid][i]), int(y_2d[valid][i])), 2, (0, 255, 0), -1)
        
        # 바운딩 박스
        min_x, max_x = int(np.min(x_2d[valid])), int(np.max(x_2d[valid]))
        min_y, max_y = int(np.min(y_2d[valid])), int(np.max(y_2d[valid]))
        cv2.rectangle(simple_overlay, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    axes[1, 1].imshow(simple_overlay)
    axes[1, 1].set_title('Simple Mesh Overlay')
    axes[1, 1].axis('off')
    
    # 6. 원본 foundationpose 시각화 (depth 제외)
    try:
        # depth 매개변수 없이 시도
        vis_img = visualize_frame_results(
            color,  # 위치 인수로
            K,
            mesh,
            pose,
            glctx
        )
        axes[1, 2].imshow(vis_img)
        axes[1, 2].set_title('FoundationPose Visualization')
    except Exception as e:
        print(f"⚠️ FoundationPose visualization failed: {e}")
        # 실패시 간단한 텍스트 표시
        axes[1, 2].text(0.5, 0.5, f'Visualization\nFailed:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Visualization Failed')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(save_path, 'complete_visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return vis_path

if __name__=='__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="./dataset/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true", help="Running with InstantMesh+SAM2")
    parser.add_argument("--debug", action="store_true", help="Enable comprehensive debugging")
    args = parser.parse_args()

    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d
    debug_mode = args.debug

    results = []
    demo_path = 'demo_data'
    mesh_path = os.path.join(demo_path, f'bottom_case.obj')

    obj = 'demo_bottom_case'
    save_path = f'results/{obj}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("🚀 Any6D with comprehensive debugging started!")

    # ===== 추론 시간 측정 시작 =====
    timer = InferenceTimer()

    # 1. 데이터 로딩
    timer.start("1. Data Loading (Image/Depth/Mask)")
    depth_scale = 1000.0
    color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    timer.end()
    
    # 이미지 크기 확인 및 출력
    print(f"Original color shape: {color.shape}")
    print(f"Original depth shape: {depth.shape}")
    
    # 크기가 다르면 맞춰주기
    if color.shape[:2] != depth.shape[:2]:
        print("⚠️ Color and depth image sizes don't match. Resizing...")
        depth = cv2.resize(depth, (color.shape[1], color.shape[0]))
        print(f"Resized depth shape: {depth.shape}")

    Image.fromarray(color).save(os.path.join(save_path, 'color.png'))

    timer.start("2. Mask Loading & Preprocessing")
    label = np.load(os.path.join(demo_path, 'labels.npz'))
    obj_num = 5
    mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)
    timer.end()
    
    print(f"Original mask shape: {mask.shape}")
    
    # 마스크 크기도 이미지에 맞추기
    if mask.shape != color.shape[:2]:
        print("⚠️ Mask and image sizes don't match. Resizing mask...")
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_uint8, (color.shape[1], color.shape[0]))
        mask = (mask_resized > 127).astype(np.bool_)
        print(f"Resized mask shape: {mask.shape}")
    
    # 최종 크기 확인
    print(f"\nFinal shapes:")
    print(f"Color: {color.shape}")
    print(f"Depth: {depth.shape}")
    print(f"Mask: {mask.shape}")
    
    # 크기가 일치하는지 체크
    assert color.shape[:2] == depth.shape[:2] == mask.shape, "Image dimensions must match!"

    # === Mesh 로드/생성 ===
    timer.start("3. Mesh Loading/Generation")
    if img_to_3d:
        print("🎨 Using InstantMesh + SAM2 for 3D reconstruction...")
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)

        input_image = preprocess_image(color, mask_refine, save_path, obj)
        images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
        instant_mesh_process(images, save_path, obj)

        mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))
        mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
        print("✅ InstantMesh reconstruction completed!")
    else:
        try:
            mesh = trimesh.load(mesh_path)
            print(f"✅ Loaded existing mesh: {mesh_path}")
        except FileNotFoundError:
            print(f"❌ Mesh file not found: {mesh_path}")
            print("💡 Please use --img_to_3d option to generate mesh with InstantMesh")
            exit(1)
    timer.end()

    # === Camera info ===
    intrinsic_path = f"{demo_path}/realsense_f435_640x480.yml"
    with open(intrinsic_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    intrinsic = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]], 
                         [0.0, data["depth"]["fy"], data["depth"]["ppy"]], 
                         [0.0, 0.0, 1.0]])
    
    # 내재 파라미터 조정
    original_width = 640
    original_height = 480
    current_width = color.shape[1]
    current_height = color.shape[0]
    
    if current_width != original_width or current_height != original_height:
        print(f"Adjusting intrinsic parameters from {original_width}x{original_height} to {current_width}x{current_height}")
        scale_x = current_width / original_width
        scale_y = current_height / original_height
        
        intrinsic[0, 0] *= scale_x  # fx
        intrinsic[1, 1] *= scale_y  # fy
        intrinsic[0, 2] *= scale_x  # ppx
        intrinsic[1, 2] *= scale_y  # ppy
        
        print(f"Adjusted intrinsic matrix:\n{intrinsic}")
    
    np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)

    # === 종합 디버깅 및 전처리 ===
    print("\n" + "="*50)
    print("🔍 COMPREHENSIVE DEBUGGING & PREPROCESSING")
    print("="*50)

    timer.start("4. Preprocessing - Mask Validation")
    # 1. 마스크 품질 검증 및 개선
    mask = validate_mask_quality(color, depth, mask, save_path)
    timer.end()

    timer.start("5. Preprocessing - Camera Calibration")
    # 2. 카메라 캘리브레이션 검증
    validate_camera_calibration(intrinsic, color.shape)
    timer.end()

    # 3. 해상도 최적화 (필요한 경우)
    if debug_mode:
        timer.start("6. Preprocessing - Resolution Opt")
        color, depth, mask, res_scale = optimize_resolution(color, depth, mask)
        if res_scale != 1.0:
            intrinsic *= res_scale
            intrinsic[2, 2] = 1.0
        timer.end()

    timer.start("7. Preprocessing - Scale Check")
    # 4. 스케일 불일치 체크 및 수정
    mesh = check_scale_mismatch(mesh, depth, mask, intrinsic)
    timer.end()

    timer.start("8. Preprocessing - Coordinate Alignment")
    # 5. 좌표계 정렬 최적화
    mesh = align_mesh_coordinate_system(mesh, depth, mask, intrinsic, save_path)
    timer.end()
    
    # 최종 mesh 저장
    optimized_mesh_path = os.path.join(save_path, f'optimized_mesh_{obj}.obj')
    mesh.export(optimized_mesh_path)
    print(f"✅ Optimized mesh saved: {optimized_mesh_path}")
    
    print("\n" + "="*50)
    print("🎯 STARTING ANY6D POSE ESTIMATION")
    print("="*50)

    # === Any6D 추정 (최적화된 데이터로) ===
    timer.start("9. Any6D Model Initialization")
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)
    timer.end()

    try:
        timer.start("10. Any6D Pose Estimation (Core)")
        pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')
        timer.end()
        print("✅ 6D pose estimation completed successfully!")
        
        # === 완전한 시각화 ===
        vis_path = visualize_with_depth(color, depth, mask, est.mesh, pred_pose, intrinsic, save_path)
        print(f"✅ Complete visualization saved: {vis_path}")
        
        # Pose 정보 출력 및 저장
        print(f"\n📍 Estimated Translation: {pred_pose[:3, 3]}")
        print(f"📐 Estimated Rotation:\n{pred_pose[:3, :3]}")
        
        # 결과 저장
        np.savetxt(os.path.join(save_path, f'{obj}_estimated_pose.txt'), pred_pose)
        
        # JSON으로도 저장
        import json
        pose_info = {
            'translation': pred_pose[:3, 3].tolist(),
            'rotation_matrix': pred_pose[:3, :3].tolist(),
            'full_pose_4x4': pred_pose.tolist()
        }
        
        with open(os.path.join(save_path, 'pose_info.json'), 'w') as f:
            json.dump(pose_info, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error during pose estimation: {e}")
        print("Shapes at error:")
        print(f"  Color: {color.shape}")
        print(f"  Depth: {depth.shape}")  
        print(f"  Mask: {mask.shape}")
        import traceback
        traceback.print_exc()
        raise

    # === GT 비교 (옵션) ===
    try:
        pose_list = label['pose_y']
        index_list = np.unique(label['seg'])
        index = (np.where(index_list == obj_num)[0] - 1).tolist()[0]
        tmp = pose_list[index]
        gt_pose = np.eye(4)
        gt_pose[:3, :] = tmp
        
        print(f"\n📍 Ground Truth Translation: {gt_pose[:3, 3]}")
        print(f"📐 Ground Truth Rotation:\n{gt_pose[:3, :3]}")
        
        # Translation error 계산
        translation_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
        print(f"📏 Translation Error: {translation_error:.4f}m")
        
        # Rotation error 계산 (angle)
        R_rel = np.dot(pred_pose[:3, :3], gt_pose[:3, :3].T)
        angle_error = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        print(f"📐 Rotation Error: {np.degrees(angle_error):.2f}°")
        
        # 종합 점수
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"Translation Error: {translation_error:.4f}m")
        print(f"Rotation Error: {np.degrees(angle_error):.2f}°")
        
        if translation_error < 0.05 and np.degrees(angle_error) < 10:
            print("🎉 EXCELLENT PERFORMANCE!")
        elif translation_error < 0.1 and np.degrees(angle_error) < 30:
            print("✅ GOOD PERFORMANCE!")
        else:
            print("⚠️ NEEDS IMPROVEMENT - Check scale/alignment")
        
        np.savetxt(os.path.join(save_path, f'{obj}_gt_pose.txt'), gt_pose)
        
    except Exception as e:
        print(f"⚠️ GT comparison failed: {e}")

    # ===== 추론 시간 분석 리포트 출력 =====
    timer.print_report(target_fps=30)

    # JSON으로도 저장
    import json
    timing_summary = timer.get_summary()
    total_time = sum([v['mean'] for v in timing_summary.values()])
    timing_output = {
        'detailed_timings_ms': {k: {
            'mean': float(v['mean'] * 1000),
            'std': float(v['std'] * 1000),
            'min': float(v['min'] * 1000),
            'max': float(v['max'] * 1000)
        } for k, v in timing_summary.items()},
        'total_inference_time_ms': float(total_time * 1000),
        'estimated_fps': float(1.0 / total_time if total_time > 0 else 0),
        'realtime_30fps_capable': bool(total_time < (1.0/30))
    }

    timing_json_path = os.path.join(save_path, 'timing_analysis.json')
    with open(timing_json_path, 'w') as f:
        json.dump(timing_output, f, indent=2)

    print("\n✅ All processing completed!")
    print(f"📁 Results saved in: {save_path}")
    print(f"📊 Check the debugging visualizations for detailed analysis")
    print(f"⏱️  Timing analysis saved: {timing_json_path}")
