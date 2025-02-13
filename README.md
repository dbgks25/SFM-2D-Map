# photo3_viz.py

**photo3_viz.py**는 포인트 클라우드(.ply) 데이터를 기반으로 바닥면을 추출하고, 이를 2D 평면에 투영하여 상호작용형(Interactive)으로 시각화하는 Python 스크립트입니다.  
DBSCAN 클러스터링과 OpenCV의 모폴로지 연산 등을 활용하여 노이즈를 제거하고, 최종적으로 가장 큰 바닥 윤곽선을 추출한 뒤 면적, 둘레, 각 변의 길이를 시각화합니다.  
슬라이더를 통해 다양한 파라미터를 즉시 조정하면서 결과를 확인할 수 있습니다.

---

## 주요 기능

1. **포인트 클라우드 로드 및 다운샘플링**  
   - `open3d.io.read_point_cloud`를 이용해 `.ply` 파일에서 포인트 클라우드를 읽어옵니다.  
   - `voxel_down_sample`을 사용하여 점 개수를 줄이고, 연산 효율을 높입니다.

2. **RANSAC을 통한 바닥면 검출**  
   - 직접 구현한 `detect_plane_ransac` 함수를 통해 3개의 점으로 평면 방정식을 추정하고, 인라이어(Inlier)가 가장 많은 평면을 바닥면으로 선택합니다.

3. **바닥면 정렬(Orientation)**  
   - 검출된 바닥면의 법선을 Z축 방향과 일치시키도록 회전 행렬을 계산하여, 바닥면이 XY 평면 위에 놓이도록 만듭니다.

4. **2D 투영 및 래스터화**  
   - 정렬된 바닥 포인트를 2D 좌표계로 투영한 뒤, 해상도( mm 단위 )를 설정하여 격자 이미지(배열)를 생성합니다.

5. **DBSCAN 및 이미지 기반 후처리**  
   - `DBSCAN`을 활용해 노이즈 점들을 제거하고, OpenCV 모폴로지 연산으로 외곽선의 폐합 및 노이즈 제거를 수행합니다.  
   - 가장 큰 윤곽(Contour)을 바닥의 다각형으로 간주하며, 이를 폴리라인으로 근사화(PolyDP)합니다.

6. **최종 윤곽 시각화 & 치수 표기**  
   - Matplotlib 상에서 바닥 윤곽을 시각화하고, 다각형 면적( m² ), 둘레 길이( m ), 그리고 윤곽선을 이루는 각 변의 길이를 표시합니다.  
   - 30cm 이상 길이의 변마다 자동으로 치수선을 추가하여 도면 해석을 돕습니다.

7. **슬라이더 기반 인터랙티브 조정**  
   - Matplotlib 슬라이더로 DBSCAN 파라미터(`eps_cm`, `min_samples`), 해상도(`resolution_mm`), 모폴로지 커널 크기(`kernel_size`), 반복 횟수(`iterations`) 등을 조절하면서 실시간으로 결과를 확인할 수 있습니다.

---

## 의존성(Dependencies)

- **Python** 3.7 이상 권장  
- [Open3D](http://www.open3d.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [SciPy](https://www.scipy.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [OpenCV-Python](https://pypi.org/project/opencv-python/)  
- [Shapely](https://pypi.org/project/Shapely/)  

다음과 같이 `pip` 명령어를 통해 설치할 수 있습니다:

```bash
pip install open3d numpy matplotlib scipy scikit-learn opencv-python shapely
