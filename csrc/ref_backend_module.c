#include <Python.h>

static struct PyModuleDef ref_backend_module = {
    PyModuleDef_HEAD_INIT,
    "_ref_backend",
    "Reference backend C extension",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit__ref_backend(void) {
    return PyModule_Create(&ref_backend_module);
}
