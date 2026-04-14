#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "worker.h"

static void radix_sort(float* arr, float* temp, size_t len) {
  uint32_t* src = reinterpret_cast<uint32_t*>(arr);
  uint32_t* dst = reinterpret_cast<uint32_t*>(temp);
  for (size_t i = 0; i < len; ++i) {
    uint32_t u = src[i];
    src[i] = (u >> 31) ? ~u : (u | 0x80000000);
  }
  for (int shift = 0; shift < 32; shift += 8) {
    uint32_t count[256] = {0};
    for (size_t i = 0; i < len; ++i) {
      count[(src[i] >> shift) & 0xFF]++;
    }
    uint32_t pos[256];
    pos[0] = 0;
    for (int i = 1; i < 256; ++i) {
        pos[i] = pos[i - 1] + count[i - 1];
    }
    for (size_t i = 0; i < len; ++i) {
      dst[pos[(src[i] >> shift) & 0xFF]++] = src[i];
    }
    uint32_t* tmp = src;
    src = dst;
    dst = tmp;
  }
  for (size_t i = 0; i < len; ++i) {
    uint32_t u = src[i];
    src[i] = (u >> 31) ? (u & 0x7FFFFFFF) : ~u;
  }
}

void Worker::sort() {
  if (out_of_range || block_len == 0) {
    return;
  }
  size_t b_size = (n + nprocs - 1) / nprocs;
  int active_procs = (n + b_size - 1) / b_size;
  float* bufferB = new float[b_size];
  float* recv_data = new float[b_size];
  if (block_len > 10000) {
    radix_sort(data, bufferB, block_len);
  } else {
    std::sort(data, data + block_len);
  }
  float* src = data;
  float* dst = bufferB;
  for (int shift = 0; shift < nprocs; shift++) {
    int neighbor = (shift & 1) ? (rank & 1 ? rank + 1 : rank - 1) : (rank & 1 ? rank - 1 : rank + 1);
    if (neighbor >= 0 && neighbor < active_procs) {
      size_t neighbor_len = (neighbor == active_procs - 1) ? (n - neighbor * b_size) : b_size;
      float my_pivot, recv_pivot;
      my_pivot = (rank < neighbor) ? src[block_len - 1] : src[0];
      MPI_Sendrecv(&my_pivot, 1, MPI_FLOAT, neighbor, 0,
                   &recv_pivot, 1, MPI_FLOAT, neighbor, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      bool need_exchange = 0;
      if (rank < neighbor) {
        if (my_pivot > recv_pivot) {
          need_exchange = 1; 
        }
      } else {
        if (my_pivot < recv_pivot) {
          need_exchange = 1;
        }
      }
      if (need_exchange) {
        MPI_Sendrecv(src, block_len, MPI_FLOAT, neighbor, 1, 
                     recv_data, neighbor_len, MPI_FLOAT, neighbor, 1, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        size_t k = 0;
        if (rank < neighbor) {
          size_t skip_i = std::upper_bound(src, src + block_len, recv_data[0]) - src;
          if (skip_i > 0) {
            memcpy(dst, src, skip_i * sizeof(float));
          }
          size_t i = skip_i, j = 0, k = skip_i;
          while (k < block_len && i < block_len && j < neighbor_len) {
            bool cmp = src[i] <= recv_data[j];
            dst[k] = cmp ? src[i] : recv_data[j];
            i += cmp;
            j += !cmp;
            k++;
          }
          while (k < block_len && i < block_len) dst[k++] = src[i++];
          while (k < block_len && j < neighbor_len) dst[k++] = recv_data[j++];
        } else {
          size_t skip_count = 0;
          if (neighbor_len > 0) {
            float max_recv = recv_data[neighbor_len - 1];
            size_t safe_start = std::lower_bound(src, src + block_len, max_recv) - src;
            skip_count = block_len - safe_start;
            if (skip_count > 0) {
              memcpy(dst + block_len - skip_count, src + block_len - skip_count, skip_count * sizeof(float));
            }
          }
          long long i = block_len - skip_count - 1;
          long long j = neighbor_len - 1;
          long long k = block_len - skip_count - 1;
          while (k >= 0 && i >= 0 && j >= 0) {
            bool cmp = src[i] >= recv_data[j];
            dst[k] = cmp ? src[i] : recv_data[j];
            i -= cmp;
            j -= !cmp;
            k--;
          }
          while (k >= 0 && i >= 0) dst[k--] = src[i--];
          while (k >= 0 && j >= 0) dst[k--] = recv_data[j--];
        }
        std::swap(src, dst);
      }
    }
  }
  if (src != data) {
    memcpy(data, src, block_len * sizeof(float));
  }
  delete[] recv_data;
  delete[] bufferB;
}