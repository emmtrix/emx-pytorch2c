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
