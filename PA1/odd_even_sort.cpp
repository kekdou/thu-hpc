#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "worker.h"

void Worker::sort() {
  if (out_of_range) {
    return;
  }
  if (block_len > 0) {
    std::sort(data, data + block_len);
  }
  size_t b_size = (n + nprocs - 1) / nprocs;
  int active_procs = (n + b_size - 1) / b_size;
  float* recv_data = nullptr;
  float* temp = nullptr;
  recv_data = new float[b_size];
  temp = new float[b_size];
  for (int i = 0; i < nprocs; ++i) {
    int local_changed = 0;
    int neighbor = -1;
    if (i % 2 == 0) {
      neighbor = (rank % 2 == 0) ? rank + 1 : rank - 1;
    } else {
      neighbor = (rank % 2 != 0) ? rank + 1 : rank - 1;
    }
    if (neighbor >= 0 && neighbor < active_procs) {
      size_t neighbor_len = (neighbor == active_procs - 1) ? (n - neighbor * b_size) : b_size;
      MPI_Sendrecv(data, block_len, MPI_FLOAT, neighbor, 0, recv_data,
                    neighbor_len, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
      if (rank < neighbor) {
        if (data[block_len - 1] > recv_data[0]) {
          local_changed = 1;
          long long i = 0, j = 0, k = 0;
          while (k < (long long)block_len) {
            if (j == (long long)neighbor_len ||
                (i < (long long)block_len && data[i] <= recv_data[j])) {
              temp[k++] = data[i++];
            } else {
              temp[k++] = recv_data[j++];
            }
          }
          for (size_t idx = 0; idx < block_len; ++idx) data[idx] = temp[idx];
        }
      } else {
        if (data[0] < recv_data[neighbor_len - 1]) {
          local_changed = 1;
          long long i = block_len - 1, j = neighbor_len - 1,
                    k = block_len - 1;
          while (k >= 0) {
            if (j < 0 || (i >= 0 && data[i] >= recv_data[j])) {
              temp[k--] = data[i--];
            } else {
              temp[k--] = recv_data[j--];
            }
          }
          for (size_t idx = 0; idx < block_len; ++idx) data[idx] = temp[idx];
        }
      }
    }
    int global_changed = 0;
    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (global_changed == 0) {
      break;
    }
  }
  delete[] recv_data;
  delete[] temp;
}