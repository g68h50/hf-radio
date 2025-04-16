import itertools

import numpy as np
from scipy.linalg import hadamard


class GenTools:
    barker_list = [
        [1],
        [1, 0],
        [1, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    ]

    def __init__(self, matrix_size, walsh_size):
        self.matrix_size = matrix_size
        self.walsh_size = walsh_size

    def __str__(self):
        return f"generate hadamard matrix (n={self.matrix_size}) and walsh sequence size {self.walsh_size}"

    # генерация матриц
    @staticmethod
    def _hadamard_matrix(self, n):
        """Генерация матрицы Адамара порядка n"""
        if n == 1:
            return np.array([[1]])
        else:
            h = self._hadamard_matrix(n // 2)
            return np.block([
                [h, h],
                [h, -h]
            ])

    @staticmethod
    def _walsh_sequences(self):
        """Генерация последовательностей Уолша длиной n"""
        n = self.walsh_size
        H = self._hadamard_matrix(n)
        H = (H + 1) // 2
        # Перестановка по числу переходов (для Уолша-Хэддема)
        order = np.argsort([bin(i).count('1') for i in range(n)])
        return H[order]

    @staticmethod
    def _hamming_distance(self, A, B):
        """Вычисляет процент совпадений между двумя матрицами."""
        return np.sum(A == B) / A.size

    @staticmethod
    def _check_uniform_distribution(self, A, tolerance=1):
        """Проверяет равномерное распределение 1 и 0 в строках и колонках."""
        row_sums = np.sum(A, axis=1)
        col_sums = np.sum(A, axis=0)
        return np.max(row_sums) - np.min(row_sums) <= tolerance and np.max(col_sums) - np.min(col_sums) <= tolerance

    @staticmethod
    def _calculate_correlation(self, A):
        """Оценивает максимальную корреляцию между строками."""
        n = A.shape[0]
        correlation_values = []
        for i in range(n):
            for j in range(i + 1, n):
                correlation = np.dot(2 * A[i] - 1, (2 * A[j] - 1))
                correlation_values.append(abs(correlation))
        return max(correlation_values)

    @staticmethod
    def _generate_optimized_matrices(self, similarity_threshold=0.5, tolerance=1.0):
        n = self.matrix_size
        if (n & (n - 1)) != 0 or n < 1:
            raise ValueError("Размер матрицы должен быть степенью двойки (2, 4, 8, 16, ...)")
        H = hadamard(n)
        sign_variations = list(itertools.product([-1, 1], repeat=n))
        candidate_matrices = []
        for signs in sign_variations:
            modified_H = H * np.array(signs)[:, np.newaxis]
            binary_matrix = (modified_H + 1) // 2
            inverse_matrix = 1 - binary_matrix
            if not self._check_uniform_distribution(binary_matrix, tolerance):
                continue
            if any(self._hamming_distance(binary_matrix, M) > similarity_threshold for M in candidate_matrices):
                continue
            if any(np.array_equal(inverse_matrix, M) for M in candidate_matrices):
                continue

            candidate_matrices.append(binary_matrix)  # Фильтрация по корреляции
        if candidate_matrices:
            min_correlation = min(self._calculate_correlation(M) for M in candidate_matrices)
            optimal_matrices = [M for M in candidate_matrices if self._calculate_correlation(M) == min_correlation]
        else:
            optimal_matrices = []
        return optimal_matrices

    @staticmethod
    def _check_autocorrelation(A):
        """Проверяет автокорреляцию: A A^T = kI."""
        n = A.shape[0]
        AAT = np.dot(2 * A - 1, (2 * A - 1).T)
        I = np.identity(n) * np.trace(AAT) / n
        return np.allclose(AAT, I)

    @staticmethod
    def _gen_matrix_mul_walsh(self, matrices, walsh_seq):
        _, m_col = matrices[0].shape
        w_len = len(walsh_seq[0])

        # inx, frq, time
        res = np.zeros((m_col * w_len, m_col, m_col * w_len))
        for i_w in range(w_len):
            for i_h in range(m_col):
                ws = walsh_seq[i_w]
                h = matrices[i_h]
                inv_h = 1 - h
                r = h if ws[0] == 1 else inv_h
                for k in range(w_len - 1):
                    t = h if ws[k + 1] == 1 else inv_h
                    r = np.concatenate((r, t), axis=1)
                res[i_w * m_col + i_h] = r
        return res

    @staticmethod
    def _gen_matrix_mul_barker(self, matrices, inx_barker=0):
        n_cnt, n_row, m_col = matrices.shape
        barker = self.barker_list[inx_barker]
        l_b = len(barker)
        res = np.zeros((n_cnt, n_row, m_col * l_b))

        for i in range(n_cnt):
            for j in range(n_row):
                for k in range(m_col):
                    for l in range(l_b):
                        m_i = int(matrices[i, j, k])
                        b_i = barker[(l + j) % l_b]
                        res[i, j, k * l_b + l] = m_i ^ b_i

        return res

    @staticmethod
    def _gen_matrix_preambula(self, matrices, inx_barker=3):
        _, m_col = matrices[0].shape
        barker = self.barker_list[inx_barker]
        # inx, frq, time
        l_b = len(barker)
        res = np.zeros((m_col, m_col * (2 * m_col) * l_b))
        m = np.zeros((m_col, m_col * (2 * m_col)))

        for i_h in range(m_col):
            h = matrices[i_h]
            inv_h = 1 - h
            r = np.concatenate((h, inv_h), axis=1)
            m[0:m_col, i_h * (2 * m_col):(i_h + 1) * (2 * m_col)] = r
        for j in range(m_col):
            for k in range(m_col * (2 * m_col)):
                for l in range(l_b):
                    m_i = int(m[j, k])
                    b_i = barker[(l + j) % l_b]
                    res[j, k * l_b + l] = m_i ^ b_i
        return res

    def get_matrix(self, inx_barker=0):
        similarity_threshold = 0.5
        tolerance = self.matrix_size / 2

        # Вывод последовательностей Уолша длиной 32
        sequences = self._walsh_sequences()
        # варакин
        walsh_seq = []
        for seq in sequences:
            if self.walsh_size == 16:
                key = int("F9A415D8", 16)
            else:
                key = int("FC08629E8E4B766A", 16)
            w_s = [0] * self.walsh_size
            for i in range(self.walsh_size):
                w_s[i] = seq[i] ^ ((key >> i) & 1)
            walsh_seq.append(w_s)

        # Генерация матриц
        matrices = self._generate_optimized_matrices(self, similarity_threshold=similarity_threshold,
                                                     tolerance=tolerance)
        m_w = self._gen_matrix_mul_walsh(self, matrices=matrices, walsh_seq=walsh_seq)
        res = self._gen_matrix_mul_barker(self, matrices=m_w, inx_barker=inx_barker)

        return res

    def get_matrix_preambula(self, inx_barker=3):
        similarity_threshold = 0.5
        tolerance = self.matrix_size / 2

        # Генерация матриц
        matrices = self._generate_optimized_matrices(self, similarity_threshold=similarity_threshold,
                                                     tolerance=tolerance)
        res = self._gen_matrix_preambula(self, matrices=matrices, inx_barker=inx_barker)

        return res
