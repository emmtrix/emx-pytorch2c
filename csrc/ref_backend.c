#include "ref_backend.h"

#include <stdio.h>
#include <string.h>

static void write_error(char *err_msg, size_t err_cap, const char *msg) {
    if (err_msg == NULL || err_cap == 0) {
        return;
    }
    strncpy(err_msg, msg, err_cap - 1);
    err_msg[err_cap - 1] = '\0';
}

int ref_run_add(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sub(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_mul(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_div(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_maximum(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_minimum(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_neg(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_exp(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_abs(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sqrt(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_log(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_sin(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_cos(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_tanh(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_floor(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_ceil(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_reciprocal(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_relu(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_matmul(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_bmm(const RefOpCall *call, char *err_msg, size_t err_cap);
int ref_run_broadcast_in_dim(const RefOpCall *call, char *err_msg, size_t err_cap);

int ref_run_op(int32_t op_kind, const RefOpCall *call, char *err_msg, size_t err_cap) {
    if (call == NULL) {
        write_error(err_msg, err_cap, "RefOpCall is NULL");
        return 1;
    }
    switch (op_kind) {
        case REF_OP_ADD:
            return ref_run_add(call, err_msg, err_cap);
        case REF_OP_SUB:
            return ref_run_sub(call, err_msg, err_cap);
        case REF_OP_MUL:
            return ref_run_mul(call, err_msg, err_cap);
        case REF_OP_DIV:
            return ref_run_div(call, err_msg, err_cap);
        case REF_OP_MAXIMUM:
            return ref_run_maximum(call, err_msg, err_cap);
        case REF_OP_MINIMUM:
            return ref_run_minimum(call, err_msg, err_cap);
        case REF_OP_NEG:
            return ref_run_neg(call, err_msg, err_cap);
        case REF_OP_EXP:
            return ref_run_exp(call, err_msg, err_cap);
        case REF_OP_ABS:
            return ref_run_abs(call, err_msg, err_cap);
        case REF_OP_SQRT:
            return ref_run_sqrt(call, err_msg, err_cap);
        case REF_OP_LOG:
            return ref_run_log(call, err_msg, err_cap);
        case REF_OP_SIN:
            return ref_run_sin(call, err_msg, err_cap);
        case REF_OP_COS:
            return ref_run_cos(call, err_msg, err_cap);
        case REF_OP_TANH:
            return ref_run_tanh(call, err_msg, err_cap);
        case REF_OP_FLOOR:
            return ref_run_floor(call, err_msg, err_cap);
        case REF_OP_CEIL:
            return ref_run_ceil(call, err_msg, err_cap);
        case REF_OP_RECIPROCAL:
            return ref_run_reciprocal(call, err_msg, err_cap);
        case REF_OP_RELU:
            return ref_run_relu(call, err_msg, err_cap);
        case REF_OP_MATMUL:
            return ref_run_matmul(call, err_msg, err_cap);
        case REF_OP_BMM:
            return ref_run_bmm(call, err_msg, err_cap);
        case REF_OP_BROADCAST_IN_DIM:
            return ref_run_broadcast_in_dim(call, err_msg, err_cap);
        default:
            write_error(err_msg, err_cap, "Unsupported op kind");
            return 2;
    }
}
