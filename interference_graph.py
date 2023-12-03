import numpy as np
import matplotlib.pyplot as plt

def generate_interference_pdf(num_base_stations, area_size):
    # 무작위로 기지국 위치 생성
    base_stations = np.random.rand(num_base_stations, 2) * area_size

    # 측정 지점 생성
    resolution = 0.1
    x = np.arange(0, area_size[0], resolution)
    y = np.arange(0, area_size[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 거리에 따른 감쇠 모델링
    interference = np.zeros_like(xx)
    for station in base_stations:
        distances = np.sqrt(np.sum((grid_points - station)**2, axis=1))
        interference += (1 / (distances + 1))  # 거리에 따른 감쇠 모델링

    return xx, yy, interference

def plot_interference(xx, yy, interference):
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, interference, shading='auto', cmap='viridis')
    plt.colorbar(label='Interference Level')
    plt.scatter(base_stations[:, 0], base_stations[:, 1], c='red', marker='x', label='Base Stations')
    plt.title('Interference PDF with Randomly Placed Base Stations')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)  # 재현성을 위한 랜덤 시드 설정
    num_base_stations = 10
    area_size = (100, 100)  # 지역의 크기 (가로, 세로)

    xx, yy, interference = generate_interference_pdf(num_base_stations, area_size)
    plot_interference(xx, yy, interference)
