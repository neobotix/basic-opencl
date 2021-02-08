
float2 mul_22_2(const float* mat, const float2 b) {
	float2 res;
	res.x = mat[0] * b.x + mat[2] * b.y;
	res.y = mat[1] * b.x + mat[3] * b.y;
	return res;
}

float3 mul_33_3(const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z;
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z;
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z;
	return res;
}

float3 mul_34_3(const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z + mat[9];
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z + mat[10];
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z + mat[11];
	return res;
}

float2 gmul_22_2(__global const float* mat, const float2 b) {
	float2 res;
	res.x = mat[0] * b.x + mat[2] * b.y;
	res.y = mat[1] * b.x + mat[3] * b.y;
	return res;
}

float3 gmul_33_3(__global const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z;
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z;
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z;
	return res;
}

float3 gmul_34_3(__global const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z + mat[9];
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z + mat[10];
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z + mat[11];
	return res;
}

/*
 * Matrix multiplication assuming row major storage: Y_NK = A_NM * B_MK
 */
void mul_NM_K(int N, int M, int K, float* Y, const float* A, const float* B) {
	for(int i = 0; i < N*K; ++i) {
		Y[i] = 0;
	}
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int k = 0; k < M; ++k) {
				Y[j*N + i] += A[k*N + i] * B[j*M + k];
			}
		}
	}
}

/*
 * Matrix multiplication assuming row major storage: Y_MK = A_NM^T * B_MK
 */
void mul_NM_T_K(int N, int M, int K, float* Y, const float* A, const float* B) {
	for(int i = 0; i < M*K; ++i) {
		Y[i] = 0;
	}
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int k = 0; k < N; ++k) {
				Y[j*M + i] += A[i*N + k] * B[j*N + k];
			}
		}
	}
}

