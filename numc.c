#include "numc.h"
#include <structmember.h>

PyTypeObject MatrixoxType;

/* Helper functions for initalization of matrices and vectors */

/*
 * Return a tuple given rows and cols
 */
PyObject *get_shape(int rows, int cols)
{
    if (rows == 1 || cols == 1)
    {
        return PyTuple_Pack(1, PyLong_FromLong(rows * cols));
    }
    else
    {
        return PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
    }
}
/*
 * Matrix(rows, cols, low, high). Fill a matrix random double values
 */
int init_rand(PyObject *self, int rows, int cols, unsigned int seed, double low,
              double high)
{
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    rand_matrix(new_mat, seed, low, high);
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(rows, cols, val). Fill a matrix of dimension rows * cols with val
 */
int init_fill(PyObject *self, int rows, int cols, double val)
{
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    else
    {
        fill_matrix(new_mat, val);
        ((Matrixox *)self)->mat = new_mat;
        ((Matrixox *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    }
    return 0;
}

/*
 * Matrix(rows, cols, 1d_list). Fill a matrix with dimension rows * cols with 1d_list values
 */
int init_1d(PyObject *self, int rows, int cols, PyObject *lst)
{
    if (rows * cols != PyList_Size(lst))
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of elements in list");
        return -1;
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    int count = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            set(new_mat, i, j, PyFloat_AsDouble(PyList_GetItem(lst, count)));
            count++;
        }
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * Matrix(2d_list). Fill a matrix with dimension len(2d_list) * len(2d_list[0])
 */
int init_2d(PyObject *self, PyObject *lst)
{
    int rows = PyList_Size(lst);
    if (rows == 0)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Cannot initialize numc.Matrix with an empty list");
        return -1;
    }
    int cols;
    if (!PyList_Check(PyList_GetItem(lst, 0)))
    {
        PyErr_SetString(PyExc_ValueError, "List values not valid");
        return -1;
    }
    else
    {
        cols = PyList_Size(PyList_GetItem(lst, 0));
    }
    for (int i = 0; i < rows; i++)
    {
        if (!PyList_Check(PyList_GetItem(lst, i)) ||
            PyList_Size(PyList_GetItem(lst, i)) != cols)
        {
            PyErr_SetString(PyExc_ValueError, "List values not valid");
            return -1;
        }
    }
    matrix *new_mat;
    int alloc_failed = allocate_matrix(&new_mat, rows, cols);
    if (alloc_failed)
        return alloc_failed;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            set(new_mat, i, j,
                PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(lst, i), j)));
        }
    }
    ((Matrixox *)self)->mat = new_mat;
    ((Matrixox *)self)->shape = get_shape(new_mat->rows, new_mat->cols);
    return 0;
}

/*
 * This deallocation function is called when reference count is 0
 */
void Matrixox_dealloc(Matrixox *self)
{
    deallocate_matrix(self->mat);
    Py_TYPE(self)->tp_free(self);
}

/* For immutable types all initializations should take place in tp_new */
PyObject *Matrixox_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwds)
{
    /* size of allocated memory is tp_basicsize + nitems*tp_itemsize*/
    Matrixox *self = (Matrixox *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/*
 * This matrixox type is mutable, so needs init function. Return 0 on success otherwise -1
 */
int Matrixox_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Generate random matrices */
    if (kwds != NULL)
    {
        PyObject *rand = PyDict_GetItemString(kwds, "rand");
        if (!rand)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (!PyBool_Check(rand))
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
        if (rand != Py_True)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        PyObject *low = PyDict_GetItemString(kwds, "low");
        PyObject *high = PyDict_GetItemString(kwds, "high");
        PyObject *seed = PyDict_GetItemString(kwds, "seed");
        double double_low = 0;
        double double_high = 1;
        unsigned int unsigned_seed = 0;

        if (low)
        {
            if (PyFloat_Check(low))
            {
                double_low = PyFloat_AsDouble(low);
            }
            else if (PyLong_Check(low))
            {
                double_low = PyLong_AsLong(low);
            }
        }

        if (high)
        {
            if (PyFloat_Check(high))
            {
                double_high = PyFloat_AsDouble(high);
            }
            else if (PyLong_Check(high))
            {
                double_high = PyLong_AsLong(high);
            }
        }

        if (double_low >= double_high)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }

        // Set seed if argument exists
        if (seed)
        {
            if (PyLong_Check(seed))
            {
                unsigned_seed = PyLong_AsUnsignedLong(seed);
            }
        }

        PyObject *rows = NULL;
        PyObject *cols = NULL;
        if (PyArg_UnpackTuple(args, "args", 2, 2, &rows, &cols))
        {
            if (rows && cols && PyLong_Check(rows) && PyLong_Check(cols))
            {
                return init_rand(self, PyLong_AsLong(rows), PyLong_AsLong(cols), unsigned_seed, double_low,
                                 double_high);
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    PyObject *arg1 = NULL;
    PyObject *arg2 = NULL;
    PyObject *arg3 = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 3, &arg1, &arg2, &arg3))
    {
        /* arguments are (rows, cols, val) */
        if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && (PyLong_Check(arg3) || PyFloat_Check(arg3)))
        {
            if (PyLong_Check(arg3))
            {
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyLong_AsLong(arg3));
            }
            else
                return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), PyFloat_AsDouble(arg3));
        }
        else if (arg1 && arg2 && arg3 && PyLong_Check(arg1) && PyLong_Check(arg2) && PyList_Check(arg3))
        {
            /* Matrix(rows, cols, 1D list) */
            return init_1d(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), arg3);
        }
        else if (arg1 && PyList_Check(arg1) && arg2 == NULL && arg3 == NULL)
        {
            /* Matrix(rows, cols, 1D list) */
            return init_2d(self, arg1);
        }
        else if (arg1 && arg2 && PyLong_Check(arg1) && PyLong_Check(arg2) && arg3 == NULL)
        {
            /* Matrix(rows, cols, 1D list) */
            return init_fill(self, PyLong_AsLong(arg1), PyLong_AsLong(arg2), 0);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments");
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return -1;
    }
}

/*
 * List of lists representations for matrices
 */
PyObject *Matrixox_to_list(Matrixox *self)
{
    int rows = self->mat->rows;
    int cols = self->mat->cols;
    PyObject *py_lst = NULL;
    if (self->mat->is_1d)
    { // If 1D matrix, print as a single list
        py_lst = PyList_New(rows * cols);
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                PyList_SetItem(py_lst, count, PyFloat_FromDouble(get(self->mat, i, j)));
                count++;
            }
        }
    }
    else
    { // if 2D, print as nested list
        py_lst = PyList_New(rows);
        for (int i = 0; i < rows; i++)
        {
            PyList_SetItem(py_lst, i, PyList_New(cols));
            PyObject *curr_row = PyList_GetItem(py_lst, i);
            for (int j = 0; j < cols; j++)
            {
                PyList_SetItem(curr_row, j, PyFloat_FromDouble(get(self->mat, i, j)));
            }
        }
    }
    return py_lst;
}

PyObject *Matrixox_class_to_list(Matrixox *self, PyObject *args)
{
    PyObject *mat = NULL;
    if (PyArg_UnpackTuple(args, "args", 1, 1, &mat))
    {
        if (!PyObject_TypeCheck(mat, &MatrixoxType))
        {
            PyErr_SetString(PyExc_TypeError, "Argument must of type numc.Matrix!");
            return NULL;
        }
        Matrixox *matox = (Matrixox *)mat;
        return Matrixox_to_list(matox);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
}

/*
 * Add class methods
 */
PyMethodDef Matrixox_class_methods[] = {
    {"to_list", (PyCFunction)Matrixox_class_to_list, METH_VARARGS, "Returns a list representation of numc.Matrix"},
    {NULL, NULL, 0, NULL}};

/*
 * Matrixox string representation. For printing purposes.
 */
PyObject *Matrixox_repr(PyObject *self)
{
    PyObject *py_lst = Matrixox_to_list((Matrixox *)self);
    return PyObject_Repr(py_lst);
}

/* NUMBER METHODS */

/*
 * Add the second numc.Matrix (Matrixox) object to the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrixox_add(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimentions do not match");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->rows != other->mat->rows ||
        self->mat->cols != other->mat->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimention do not match");
        return NULL;
    }

    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }

    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);
    add_matrix(ans->mat, other->mat, self->mat);

    ans->shape = get_shape(ans->mat->rows, ans->mat->cols);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * Substract the second numc.Matrix (Matrixox) object from the first one. The first operand is
 * self, and the second operand can be obtained by casting `args`.
 */
PyObject *Matrixox_sub(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimentions do not match");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->rows != other->mat->rows ||
        self->mat->cols != other->mat->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimention do not match");
        return NULL;
    }

    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }

    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);
    sub_matrix(ans->mat, self->mat, other->mat);

    ans->shape = get_shape(ans->mat->rows, ans->mat->cols);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * NOT element-wise multiplication. The first operand is self, and the second operand
 * can be obtained by casting `args`.
 */
PyObject *Matrixox_multiply(Matrixox *self, PyObject *args)
{
    if (!PyObject_TypeCheck(args, &MatrixoxType))
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimentions do not match");
        return NULL;
    }

    Matrixox *other = (Matrixox *)args;

    if (self->mat->rows != other->mat->rows ||
        self->mat->cols != other->mat->cols)
    {
        PyErr_SetString(PyExc_ValueError, "Matrix dimention do not match");
        return NULL;
    }

    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }

    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);
    mul_matrix(ans->mat, other->mat, self->mat);

    ans->shape = get_shape(ans->mat->rows, ans->mat->cols);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * Negates the given numc.Matrix.
 */
PyObject *Matrixox_neg(Matrixox *self)
{
    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }
    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);

    neg_matrix(ans->mat, self->mat);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * Take the element-wise absolute value of this numc.Matrix.
 */
PyObject *Matrixox_abs(Matrixox *self)
{
    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }
    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);

    abs_matrix(ans->mat, self->mat);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * Raise numc.Matrix (Matrixox) to the `pow`th power. You can ignore the argument `optional`.
 */
PyObject *Matrixox_pow(Matrixox *self, PyObject *pow, PyObject *optional)
{
    Matrixox *ans = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
    if (!ans)
    {
        return NULL;
    }
    allocate_matrix(&ans->mat, self->mat->rows, self->mat->cols);

    int p = (int)PyLong_AsLong(pow);
    pow_matrix(ans->mat, self->mat, p);

    PyObject *res = (PyObject *)ans;

    return res;
}

/*
 * Create a PyNumberMethods struct for overloading operators with all the number methods you have
 * define. You might find this link helpful: https://docs.python.org/3.6/c-api/typeobj.html
 */
PyNumberMethods Matrixox_as_number = {
    .nb_add = (binaryfunc)Matrixox_add,
    .nb_subtract = (binaryfunc)Matrixox_sub,
    .nb_multiply = (binaryfunc)Matrixox_multiply,
    .nb_absolute = (binaryfunc)Matrixox_abs,
    .nb_negative = (binaryfunc)Matrixox_neg,
    .nb_power = (binaryfunc)Matrixox_pow,
};

/* INSTANCE METHODS */

/*
 * Given a numc.Matrix self, parse `args` to (int) row, (int) col, and (double/int) val.
 * Return None in Python (this is different from returning null).
 */
PyObject *Matrixox_set_value(Matrixox *self, PyObject *args)
{
    PyObject *r = NULL;
    PyObject *c = NULL;
    PyObject *v = NULL;

    if (!PyArg_UnpackTuple(args, "set", 3, 3, &r, &c, &v))
    {
        return NULL;
    }

    if (!PyLong_Check(r) || !PyLong_Check(c) ||
        !PyLong_Check(v) && !PyFloat_Check(v))
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument types");
        return NULL;
    }

    int row = PyLong_AsLong(r);
    int col = PyLong_AS_LONG(c);
    double val = PyLong_Check(v) ? PyLong_AS_LONG(v) : PyFloat_AS_DOUBLE(v);

    if (row < 0 || col < 0 || row >= self->mat->rows || col >= self->mat->cols)
    {
        PyErr_SetString(PyExc_IndexError, "row or column out of range");
        return NULL;
    }

    set(self->mat, row, col, val);

    Py_RETURN_NONE;
}

/*
 * Given a numc.Matrix `self`, parse `args` to (int) row and (int) col.
 * Return the value at the `row`th row and `col`th column, which is a Python
 * float/int.
 */
PyObject *Matrixox_get_value(Matrixox *self, PyObject *args)
{
    PyObject *r = NULL;
    PyObject *c = NULL;
    PyObject *v = NULL;

    if (!PyArg_UnpackTuple(args, "get", 2, 2, &r, &c))
    {
        return NULL;
    }

    if (!PyLong_Check(r) || !PyLong_Check(c))
    {
        PyErr_SetString(PyExc_TypeError, "Invalid argument types");
        return NULL;
    }

    int row = PyLong_AsLong(r);
    int col = PyLong_AS_LONG(c);

    if (row < 0 || col < 0 || row >= self->mat->rows || col >= self->mat->cols)
    {
        PyErr_SetString(PyExc_IndexError, "row or column out of range");
        return NULL;
    }

    double ans = get(self->mat, row, col);
    PyObject *res = PyFloat_FromDouble(ans);

    return res;
}

/*
 * Create an array of PyMethodDef structs to hold the instance methods.
 * Name the python function corresponding to Matrixox_get_value as "get" and Matrixox_set_value
 * as "set"
 * You might find this link helpful: https://docs.python.org/3.6/c-api/structures.html
 */
PyMethodDef Matrixox_methods[] = {
    {"get", (PyCFunction)Matrixox_get_value, METH_VARARGS, "Gets a value from the matrix at (row, col)"},
    {"set", (PyCFunction)Matrixox_set_value, METH_VARARGS, "Sets a value in the matrix at (row, col, val)"},
    {NULL, NULL, 0, NULL}};

/* INDEXING */

/*
 * Given a numc.Matrix `self`, index into it with `key`. Return the indexed result.
 */
PyObject *Matrixox_subscript(Matrixox *self, PyObject *key)
{
    if (self->mat->is_1d == 1)
    {
        int len = (self->mat->rows == 1) ? self->mat->cols : self->mat->rows;

        if (PyLong_Check(key))
        {
            int num = PyLong_AS_LONG(key);

            if (num < 0 || num >= len)
            {
                PyErr_SetString(PyExc_IndexError, "Index out of range");
                return NULL;
            }
            double val = (self->mat->rows == 1) ? self->mat->data[0][num] : self->mat->data[num][0];
            return PyFloat_FromDouble(val);
        }

        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (PySlice_GetIndicesEx(key, self->mat->rows, &start, &stop, &step, &slicelength) < 0)
            {
                return NULL;
            }

            if (step != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return NULL;
            }

            if (slicelength < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return NULL;
            }

            Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
            if (!result)
            {
                return NULL;
            }

            int row_offset = (self->mat->rows == 1) ? 0 : start;
            int col_offset = (self->mat->rows == 1) ? start : 0;
            int new_rows = (self->mat->rows == 1) ? 1 : slicelength;
            int new_cols = (self->mat->rows == 1) ? slicelength : 1;

            int err = allocate_matrix_ref(&(result->mat), self->mat, row_offset, col_offset, new_rows, new_cols);

            if (err != 0)
            {
                Py_DECREF(result);
                return NULL;
            }

            result->shape = get_shape(result->mat->rows, result->mat->cols);
            return (PyObject *)result;
        }

        PyErr_SetString(PyExc_TypeError, "Invalid index type");
        return NULL;
    }
    else
    {
        int row_len = self->mat->rows;
        int col_len = self->mat->cols;

        if (PyLong_Check(key))
        {
            int num = PyLong_AS_LONG(key);

            if (num < 0 || num >= row_len)
            {
                PyErr_SetString(PyExc_IndexError, "Index out of range");
                return NULL;
            }
            Matrixox *val = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);

            int err = allocate_matrix_ref(&(val->mat), self->mat, num, 0, 1, self->mat->cols);

            if (err != 0)
            {
                Py_DECREF(val);
                return NULL;
            }

            val->shape = get_shape(val->mat->rows, val->mat->cols);
            return (PyObject *)val;
        }
        if (PySlice_Check(key))
        {
            Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;

            if (PySlice_GetIndicesEx(key, row_len, &start, &stop, &step, &slicelength) < 0)
            {
                return NULL;
            }
            if (step != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return NULL;
            }
            if (slicelength < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return NULL;
            }

            Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
            if (!result)
            {
                return NULL;
            }

            int err = allocate_matrix_ref(&(result->mat), self->mat, start, 0, slicelength, col_len);

            if (err != 0)
            {
                Py_DECREF(result);
                return NULL;
            }

            result->shape = get_shape(result->mat->rows, result->mat->cols);
            return (PyObject *)result;
        }

        if (PyTuple_Check(key))
        {
            if (PyTuple_Size(key) != 2)
            {
                PyErr_SetString(PyExc_TypeError, "2D matrix require 2 slice/index arguments");
                return NULL;
            }

            PyObject *row_key = PyTuple_GetItem(key, 0);
            PyObject *col_key = PyTuple_GetItem(key, 1);

            row_len = self->mat->rows;
            col_len = self->mat->cols;

            if (PyLong_Check(row_key) && PyLong_Check(col_key))
            {
                double num = get(self->mat, PyLong_AS_LONG(row_key), PyLong_AS_LONG(col_key));
                return PyFloat_FromDouble(num);
            }
            if (PyLong_Check(row_key) && PySlice_Check(col_key))
            {
                Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
                Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
                if (PySlice_GetIndicesEx(col_key, col_len, &start, &stop, &step, &slicelength) < 0)
                {
                    return NULL;
                }
                if (step != 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                    return NULL;
                }
                if (slicelength < 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                    return NULL;
                }
                int err = allocate_matrix_ref(&(result->mat), self->mat, PyLong_AS_LONG(row_key), start, 1, slicelength);

                if (err != 0)
                {
                    Py_DECREF(result);
                    return NULL;
                }

                return (PyObject *)result;
            }

            if (PyLong_Check(col_key) && PySlice_Check(row_key))
            {
                Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
                Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
                if (PySlice_GetIndicesEx(row_key, row_len, &start, &stop, &step, &slicelength) < 0)
                {
                    return NULL;
                }
                if (step != 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                    return NULL;
                }
                if (slicelength < 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                    return NULL;
                }
                int err = allocate_matrix_ref(&(result->mat), self->mat, start, PyLong_AS_LONG(col_key), slicelength, 1);

                if (err != 0)
                {
                    Py_DECREF(result);
                    return NULL;
                }

                return (PyObject *)result;
            }

            if (PySlice_Check(col_key) && PySlice_Check(row_key))
            {
                int row_len = self->mat->rows;
                int col_len = self->mat->cols;

                Matrixox *result = (Matrixox *)MatrixoxType.tp_alloc(&MatrixoxType, 0);
                Py_ssize_t start1 = 0, stop1 = 0, step1 = 0, slicelength1 = 0;
                if (PySlice_GetIndicesEx(row_key, row_len, &start1, &stop1, &step1, &slicelength1) < 0)
                {
                    return NULL;
                }
                if (step1 != 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                    return NULL;
                }
                if (slicelength1 < 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                    return NULL;
                }

                Py_ssize_t start2 = 0, stop2 = 0, step2 = 0, slicelength2 = 0;
                if (PySlice_GetIndicesEx(col_key, col_len, &start2, &stop2, &step2, &slicelength2) < 0)
                {
                    return NULL;
                }
                if (step2 != 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                    return NULL;
                }
                if (slicelength2 < 1)
                {
                    PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                    return NULL;
                }

                int err = allocate_matrix_ref(&(result->mat), self->mat, start1, start2, slicelength1, slicelength2);

                if (err != 0)
                {
                    Py_DECREF(result);
                    return NULL;
                }

                result->shape = get_shape(result->mat->rows, result->mat->cols);
                return (PyObject *)result;
            }

            PyErr_SetString(PyExc_TypeError, "Invalid index type");
            return NULL;
        }
        PyErr_SetString(PyExc_TypeError, "Invalid key type");
        return NULL;
    }
}

/*
 * Given a numc.Matrix `self`, index into it with `key`, and set the indexed result to `v`.
 */
int Matrixox_set_subscript(Matrixox *self, PyObject *key, PyObject *v)
{
    if (!PyLong_Check(v) || !PyFloat_AS_DOUBLE(v))
    {
        PyErr_SetString(PyExc_TypeError, "value should be an int or float");
    }
    if (self->mat->is_1d)
    {
        if (PyLong_Check(key))
        {
            int idx = PyLong_AS_LONG(key);
            int len = self->mat->rows == 1 ? self->mat->cols : self->mat->rows;

            if (idx < 0 || idx > len)
            {
                PyErr_SetString(PyExc_IndexError, "Index out of range");
                return -1;
            }
            if (!PyLong_Check(v) && !PyFloat_Check(v))
            {
                PyErr_SetString(PyExc_IndexError, "Value must be an in or float");
            }

            double val = PyFloat_AS_DOUBLE(v);
            if (self->mat->rows == 1)
            {
                set(self->mat, 0, idx, val);
            }
            else
            {
                set(self->mat, idx, 0, val);
            }
        }
        else if (PySlice_Check(key))
        {
            int len = self->mat->rows == 1 ? self->mat->cols : self->mat->rows;

            if (!PyList_Check(v))
            {
                PyErr_SetString(PyExc_TypeError, "value should be a list here");
                return -1;
            }
            Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (PySlice_GetIndicesEx(key, len, &start, &stop, &step, &slicelength) < 0)
            {
                return -1;
            }
            if (step != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return -1;
            }
            if (slicelength < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return -1;
            }
            if (slicelength > len)
            {
                PyErr_SetString(PyExc_ValueError, "the length of the slice is too long");
                return -1;
            }
            if (PyList_Size(v) != slicelength)
            {
                PyErr_SetString(PyExc_ValueError, "The size of the value should suit the matrix");
                return -1;
            }

            for (int i = 0; i < slicelength; i++)
            {
                if (self->mat->rows == 1)
                {
                    set(self->mat, 0, i + start, PyFloat_AS_DOUBLE(PyList_GetItem(v, (Py_ssize_t)i)));
                }
                else
                {
                    set(self->mat, i + start, 0, PyFloat_AS_DOUBLE(PyList_GetItem(v, (Py_ssize_t)i)));
                }
            }
        }
        return 0;
    }
    else
    {
        int row_len = self->mat->rows;
        int col_len = self->mat->cols;

        if (PyLong_Check(key))
        {
            if (!PyList_Check(v))
            {
                PyErr_SetString(PyExc_TypeError, "Type didn't match");
                return -1;
            }
            if (PyList_GET_SIZE(v) != col_len)
            {
                PyErr_SetString(PyExc_ValueError, "the size of key and v didn't match");
                return -1;
            }

            for (int i = 0; i < PyList_GET_SIZE(v); i++)
            {
                set(self->mat, PyLong_AS_LONG(key), i, PyFloat_AS_DOUBLE(PyList_GET_ITEM(v, i)));
            }

            return 0;
        }
        if (PySlice_Check(key))
        {
            if (!PyTuple_Check(v))
            {
                PyErr_SetString(PyExc_TypeError, "Type didn't match");
                return -1;
            }

            Py_ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
            Py_ssize_t n = PyTuple_Size(v);
            if (PySlice_GetIndicesEx(key, row_len, &start, &stop, &step, &slicelength) < 0)
            {
                return -1;
            }
            if (step != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return -1;
            }
            if (slicelength < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return -1;
            }
            if (slicelength != n)
            {
                PyErr_SetString(PyExc_ValueError, "the size of the key and the v didn't match");
                return -1;
            }
            if (col_len != PyList_Size(PyTuple_GET_ITEM(v, 0)))
            {
                PyErr_SetString(PyExc_ValueError, "the size of the key and the v didn't match");
                return -1;
            }

            for (int i = 0; i < n; i++)
            {
                PyObject *l = PyTuple_GET_ITEM(v, i);
                for (int j = 0; j < col_len; j++)
                {
                    set(self->mat, start + i, j, PyLong_AS_LONG(PyList_GET_ITEM(l, j)));
                }
            }

            return 0;
        }
        if (PyTuple_Check(key))
        {
            if (!PyList_Check(v))
            {
                PyErr_SetString(PyExc_TypeError, "Type didn't match");
            }
            if (PyTuple_Size(key) != 2)
            {
                PyErr_SetString(PyExc_ValueError, "only support two dimentional matrix");
                return -1;
            }
            PyObject *key1 = PyTuple_GetItem(key, 0);
            PyObject *key2 = PyTuple_GetItem(key, 1);

            Py_ssize_t start1 = 0, stop1 = 0, step1 = 0, slicelength1 = 0;
            if (PySlice_GetIndicesEx(key1, row_len, &start1, &stop1, &step1, &slicelength1) < 0)
            {
                return -1;
            }
            if (step1 != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return -1;
            }
            if (slicelength1 < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return -1;
            }

            Py_ssize_t start2 = 0, stop2 = 0, step2 = 0, slicelength2 = 0;
            if (PySlice_GetIndicesEx(key2, col_len, &start2, &stop2, &step2, &slicelength2) < 0)
            {
                return -1;
            }
            if (step2 != 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice step must be 1");
                return -1;
            }
            if (slicelength2 < 1)
            {
                PyErr_SetString(PyExc_ValueError, "Slice cannot result in empty dimension");
                return -1;
            }

            Py_ssize_t rn = PyTuple_Size(v);
            Py_ssize_t cn = PyList_Size(PyTuple_GET_ITEM(v, 0));

            if (rn != slicelength1 || cn != slicelength2)
            {
                PyErr_SetString(PyExc_ValueError, "the size of key and v didn't match");
                return -1;
            }

            for (int i = 0; i < rn; i++)
            {
                for (int j = 0; j < cn; j++)
                {
                    set(self->mat, start1 + i, start2 + j, PyLong_AS_LONG(PyList_GET_ITEM(PyTuple_GET_ITEM(v, i), j)));
                }
            }

            return 0;
        }

        PyErr_SetString(PyExc_TypeError, "The type of key is not suitable");
        return -1;
    }
}

PyMappingMethods Matrixox_mapping = {
    NULL,
    (binaryfunc)Matrixox_subscript,
    (objobjargproc)Matrixox_set_subscript,
};

/* INSTANCE ATTRIBUTES*/
PyMemberDef Matrixox_members[] = {
    {"shape", T_OBJECT_EX, offsetof(Matrixox, shape), 0,
     "(rows, cols)"},
    {NULL} /* Sentinel */
};

PyTypeObject MatrixoxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numc.Matrix",
    .tp_basicsize = sizeof(Matrixox),
    .tp_dealloc = (destructor)Matrixox_dealloc,
    .tp_repr = (reprfunc)Matrixox_repr,
    .tp_as_number = &Matrixox_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT |
                Py_TPFLAGS_BASETYPE,
    .tp_doc = "numc.Matrix objects",
    .tp_methods = Matrixox_methods,
    .tp_members = Matrixox_members,
    .tp_as_mapping = &Matrixox_mapping,
    .tp_init = (initproc)Matrixox_init,
    .tp_new = Matrixox_new};

struct PyModuleDef numcmodule = {
    PyModuleDef_HEAD_INIT,
    "numc",
    "Numc matrix operations",
    -1,
    Matrixox_class_methods};

/* Initialize the numc module */
PyMODINIT_FUNC PyInit_numc(void)
{
    PyObject *m;

    if (PyType_Ready(&MatrixoxType) < 0)
        return NULL;

    m = PyModule_Create(&numcmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MatrixoxType);
    PyModule_AddObject(m, "Matrix", (PyObject *)&MatrixoxType);
    fflush(stdout);
    return m;
}