#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high)
{
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high)
{
    srand(seed);
    for (int i = 0; i < result->rows; i++)
    {
        for (int j = 0; j < result->cols; j++)
        {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 *Allocate space for matrix, init all the element in the matrix to 0
 */
int allocate_matrix(matrix **mat, int rows, int cols)
{
    if (mat == NULL || rows <= 0 || cols <= 0)
    {
        return -1;
    }

    matrix *m = malloc(sizeof(matrix));
    if (m == NULL)
    {
        free(m);
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->is_1d = rows == 1 || cols == 1;
    m->ref_cnt = 1;
    m->data = malloc(rows * cols * sizeof(double));
    m->parent = NULL;
    m->stride = cols;

    if (m->data == NULL)
    {
        free(m->data);
        free(m);
        return -1;
    }

    int i;

#pragma omp parallel for if (rows * cols > 10000)
    for (i = 0; i < rows * cols; i++)
    {
        m->data[i] = 0.0;
    }

    *mat = m;

    return 0;
}

/*
 *allocate space for ref of one existing matrix
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols)
{
    matrix *m = malloc(sizeof(matrix));
    if (m == NULL || rows < 1 || cols < 1)
    {
        free(m);
        return -1;
    }

    m->rows = rows;
    m->cols = cols;
    m->is_1d = rows == 1 || cols == 1;
    m->ref_cnt = 0;
    m->parent = from;
    m->stride = from->stride;

    m->data = from->data + (row_offset * from->stride + col_offset);

    from->ref_cnt += 1;

    *mat = m;
    return 0;
}

/*
 *deallocate the matrix, also allocate the father of the matrix if it exist
 */
void deallocate_matrix(matrix *mat)
{
    if (mat == NULL)
    {
        return;
    }

    mat->ref_cnt -= 1;

    if (mat->ref_cnt > 0)
    {
        return;
    }

    if (mat->parent == NULL)
    {
        free(mat->data);
    }
    else
    {
        deallocate_matrix(mat->parent);
    }
    free(mat);
}

/*
 * Return the double value of the matrix at the given row and column.
 */
double get(matrix *mat, int row, int col)
{
    return mat->data[row * mat->stride + col];
}

/*
 * Set the value at the given row and column to val.
 */
void set(matrix *mat, int row, int col, double val)
{
    mat->data[row * mat->stride + col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val)
{

    int i, j;
#pragma omp parallel for collapse(2) if (mat->rows * mat->cols > 10000)
    for (i = 0; i < mat->rows; i++)
    {
        for (j = 0; j < mat->cols; j++)
        {
            mat->data[i * mat->stride + j] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols)
    {
        return -1;
    }

    int row = mat1->rows;
    int col = mat2->cols;
    int i, j;

#pragma omp parallel for collapse(2) if (row * col > 10000)
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            result->data[i * result->stride + j] = mat1->data[i * mat1->stride + j] + mat2->data[i * mat2->stride + j];
        }
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols)
    {
        return -1;
    }

    int row = mat1->rows;
    int col = mat2->cols;
    int i, j;

#pragma omp parallel for collapse(2) if (row * col > 10000)
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            result->data[i * result->stride + j] = mat1->data[i * mat1->stride + j] - mat2->data[i * mat2->stride + j];
        }
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    int row1 = mat1->rows;
    int row2 = mat2->rows;
    int col1 = mat1->cols;
    int col2 = mat2->cols;

    if (col1 != row2)
    {
        return -1;
    }

    fill_matrix(result, 0);
    int i, k, j;

#pragma omp parallel for
    for (i = 0; i < row1; i++)
    {
        for (k = 0; k < col1; k++)
        {
            double r = mat1->data[i * mat1->stride + k];
            for (j = 0; j < col2; j++)
            {
                result->data[i * result->stride + j] += r * mat2->data[k * mat2->stride + j];
            }
        }
    }

    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int pow_matrix(matrix *result, matrix *mat, int pow)
{
    if (mat->rows != mat->cols || pow < 0)
    {
        return -1;
    }

    matrix *base;
    allocate_matrix(&base, mat->rows, mat->cols);

    for (int r = 0; r < mat->rows; r++)
    {
        for (int c = 0; c < mat->cols; c++)
        {
            base->data[r * base->stride + c] = mat->data[r * mat->stride + c];
        }
    }

    matrix *base_after;
    allocate_matrix(&base_after, mat->rows, mat->cols);

    for (int r = 0; r < mat->rows; r++)
    {
        for (int c = 0; c < mat->cols; c++)
        {
            base_after->data[r * base_after->stride + c] = mat->data[r * mat->stride + c];
        }
    }
    matrix *template;
    allocate_matrix(&template, mat->rows, mat->cols);

    for (int r = 0; r < mat->rows; r++)
    {
        for (int c = 0; c < mat->cols; c++)
        {
            template->data[r * template->stride + c] = mat->data[r * mat->stride + c];
        }
    }

    fill_matrix(result, 0.0);

    for (int i = 0; i < result->stride; i++)
    {
        result->data[i * result->stride + i] = 1;
    }

    while (pow > 0)
    {
        if (pow & 1)
        {
            mul_matrix(template, result, base);
            for (int r = 0; r < mat->rows; r++)
            {
                for (int c = 0; c < mat->cols; c++)
                {
                    result->data[r * result->stride + c] = template->data[r * template->stride + c];
                }
            }
        }
        mul_matrix(base_after, base, base);

        for (int r = 0; r < mat->rows; r++)
            for (int c = 0; c < mat->cols; c++)
                base->data[r * base->stride + c] = base_after->data[r * base_after->stride + c];
        pow = pow >> 1;
    }

    deallocate_matrix(base);
    deallocate_matrix(template);
    deallocate_matrix(base_after);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat)
{
    int row = mat->rows;
    int col = mat->cols;
    int i, j;

#pragma omp parallel for collapse(2) if (row * col > 10000)
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            result->data[i * result->stride + j] = mat->data[i * mat->stride + j] * -1.0;
        }
    }

    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a -1 value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat)
{
    int row = mat->rows;
    int col = mat->cols;
    int i, j;

#pragma omp parallel for collapse(2) if (row * col > 10000)
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            double num = mat->data[i * mat->stride + j];
            if (num < 0)
            {
                num = num * -1.0;
            }
            result->data[i * result->stride + j] = num;
        }
    }
    return 0;
}
