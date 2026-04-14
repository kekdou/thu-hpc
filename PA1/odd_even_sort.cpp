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
      bool need_exchange = ((rank < neighbor) && (my_pivot > recv_pivot)) || ((rank > neighbor) && (my_pivot < recv_pivot))
      if (need_exchange) {
        MPI_Sendrecv(src, block_len, MPI_FLOAT, neighbor, 1, 
                     recv_data, neighbor_len, MPI_FLOAT, neighbor, 1, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < neighbor) {
          size_t low = (block_len > neighbor_len) ? (block_len - neighbor_len) : 0;
          size_t high = block_len;
          while (low < high) {
            size_t mid = (low + high + 1) >> 1;
            size_t j_idx = block_len - mid;
            float a_left = (mid == 0) ? -1e38f : src[mid - 1];
            float b_right = (j_idx == neighbor_len) ? 1e38f : recv_data[j_idx];
            if (a_left <= b_right) {
              low = mid;
            } else {
              high = mid - 1;
            }
          }
          size_t i_len = low;
          size_t j_len = block_len - low;
          size_t i = 0, j = 0, k = 0;
          while (i < i_len && j < j_len) {
            if (src[i] <= recv_data[j]) {
              dst[k++] = src[i++];
            } else {
              dst[k++] = recv_data[j++];
            }
          }
          while (i < i_len) dst[k++] = src[i++];
          while (j < j_len) dst[k++] = recv_data[j++];
        } else {
          size_t low = 0;
          size_t high = std::min(block_len, neighbor_len);
          while (low < high) {
            size_t mid = (low + high + 1) >> 1;
            size_t j_idx = neighbor_len - mid;
            float a_left = (mid == 0) ? -1e38f : src[mid - 1];
            float b_right = (j_idx == neighbor_len) ? 1e38f : recv_data[j_idx];
            if (a_left <= b_right) {
              low = mid;
            } else {
              high = mid - 1;
            }
          }
          size_t drop_i = low;
          size_t drop_j = neighbor_len - low;
          size_t i = drop_i;
          size_t j = drop_j;
          size_t k = 0;
          while (i < block_len && j < neighbor_len) {
            if (src[i] <= recv_data[j]) {
              dst[k++] = src[i++];
            } else {
              dst[k++] = recv_data[j++];
            }
          }
          while (i < block_len) dst[k++] = src[i++];
          while (j < neighbor_len) dst[k++] = recv_data[j++];
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