#define Py_LIMITED_API 0x030A0000
#include <Python.h>

extern "C" PyObject* PyInit__C() {
    static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
    return PyModule_Create(&module);
}
