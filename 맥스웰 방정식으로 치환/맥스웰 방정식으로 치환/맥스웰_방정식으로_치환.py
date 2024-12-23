import numpy as np

# 주기율표 크기 설정 (예: 7x18)
rows, cols = 7, 18

# 원소의 전기장 및 자기장 초기화
E = np.random.rand(rows, cols, 3)  # 전기장 (Ex, Ey, Ez)
B = np.random.rand(rows, cols, 3)  # 자기장 (Bx, By, Bz)

# 밀도 및 전류
rho = np.random.rand(rows, cols)  # 전하 밀도
J = np.random.rand(rows, cols, 3)  # 전류 밀도

# 물리 상수
epsilon_0 = 8.85e-12  # 진공 유전율
mu_0 = 4 * np.pi * 1e-7  # 진공 투자율

# 맥스웰 방정식 행렬 계산
div_E = np.gradient(E[..., 0], axis=0) + np.gradient(E[..., 1], axis=1)  # div(E)
div_B = np.gradient(B[..., 0], axis=0) + np.gradient(B[..., 1], axis=1)  # div(B)
curl_E = np.cross(np.gradient(E, axis=1), E)  # curl(E)
curl_B = np.cross(np.gradient(B, axis=1), B)  # curl(B)

# 결과 출력
print("div(E):", div_E)
print("div(B):", div_B)
print("curl(E):", curl_E)
print("curl(B):", curl_B)

