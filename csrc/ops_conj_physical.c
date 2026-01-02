#include "ops_unary.h"
#include "ops_scalar.h"

int ref_run_conj_physical(const RefOpCall *call, char *err_msg, size_t err_cap) {
    return ref_run_unary_f32(call, err_msg, err_cap, ref_scalar_conj_physical, "conj_physical");
}
