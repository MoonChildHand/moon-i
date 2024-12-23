import numpy as np

# �ֱ���ǥ ũ�� ���� (��: 7x18)
rows, cols = 7, 18

# ������ ������ �� �ڱ��� �ʱ�ȭ
E = np.random.rand(rows, cols, 3)  # ������ (Ex, Ey, Ez)
B = np.random.rand(rows, cols, 3)  # �ڱ��� (Bx, By, Bz)

# �е� �� ����
rho = np.random.rand(rows, cols)  # ���� �е�
J = np.random.rand(rows, cols, 3)  # ���� �е�

# ���� ���
epsilon_0 = 8.85e-12  # ���� ������
mu_0 = 4 * np.pi * 1e-7  # ���� ������

# �ƽ��� ������ ��� ���
div_E = np.gradient(E[..., 0], axis=0) + np.gradient(E[..., 1], axis=1)  # div(E)
div_B = np.gradient(B[..., 0], axis=0) + np.gradient(B[..., 1], axis=1)  # div(B)
curl_E = np.cross(np.gradient(E, axis=1), E)  # curl(E)
curl_B = np.cross(np.gradient(B, axis=1), B)  # curl(B)

# ��� ���
print("div(E):", div_E)
print("div(B):", div_B)
print("curl(E):", curl_E)
print("curl(B):", curl_B)

