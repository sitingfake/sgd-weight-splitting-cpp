/**
 * @file csr.hpp
 * @brief Binary Compressed Sparse Row (CSR) matrix loader and representation.
 *
 * Implements little-endian loading of SGDCSR01 binary files.
 * Header format:
 *   offset 0  : magic (8 bytes: "SGDCSR01")
 *   offset 8  : m (int64) - number of rows (samples)
 *   offset 16 : n (int64) - number of columns (features)
 *   offset 24 : nnz (int64) - number of non-zero entries
 *   offset 32 : indptr (int32[m+1])
 *             : indices (int32[nnz]) - sorted ascending in each row
 *             : data (float64[nnz])
 *             : y (float64[m])
 */

#ifndef SGD_CSR_HPP
#define SGD_CSR_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <iostream>

struct CsrMatrix {
    int64_t m = 0;       ///< Number of samples (rows)
    int64_t n = 0;       ///< Number of features (columns)
    int64_t nnz = 0;     ///< Total non-zero elements

    std::vector<int32_t> indptr;   ///< Row index pointers of size m + 1
    std::vector<int32_t> indices;  ///< Column indices of size nnz
    std::vector<double> data;      ///< Non-zero values of size nnz
    std::vector<double> y;         ///< Target labels/values of size m

    /**
     * @brief Load CSR matrix from a binary file in SGDCSR01 format.
     * @param filepath Path to the binary CSR file.
     * @param max_samples If > 0, truncates dataset to the first max_samples rows.
     * @return Loaded CsrMatrix structure.
     */
    static CsrMatrix load(const std::string& filepath, int64_t max_samples = -1) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open binary data file: " + filepath);
        }

        char magic[8];
        file.read(magic, 8);
        if (std::memcmp(magic, "SGDCSR01", 8) != 0) {
            throw std::runtime_error("Invalid file magic in " + filepath + "; expected SGDCSR01");
        }

        int64_t total_m = 0, total_n = 0, total_nnz = 0;
        file.read(reinterpret_cast<char*>(&total_m), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&total_n), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&total_nnz), sizeof(int64_t));

        if (total_nnz >= std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("nnz exceeds int32 limit (2^31 - 1)");
        }

        CsrMatrix mat;
        mat.n = total_n;

        int64_t load_m = total_m;
        if (max_samples > 0 && max_samples < total_m) {
            load_m = max_samples;
        }
        mat.m = load_m;

        // Read full indptr to compute subset bounds if truncated
        std::vector<int32_t> full_indptr(total_m + 1);
        file.read(reinterpret_cast<char*>(full_indptr.data()), sizeof(int32_t) * (total_m + 1));

        int64_t load_nnz = full_indptr[load_m];
        mat.nnz = load_nnz;

        // Truncate or copy indptr
        mat.indptr.assign(full_indptr.begin(), full_indptr.begin() + (load_m + 1));

        // Read indices
        mat.indices.resize(load_nnz);
        file.read(reinterpret_cast<char*>(mat.indices.data()), sizeof(int32_t) * load_nnz);
        if (load_nnz < total_nnz) {
            // Seek past remaining indices
            file.seekg(sizeof(int32_t) * (total_nnz - load_nnz), std::ios::cur);
        }

        // Read data values
        mat.data.resize(load_nnz);
        file.read(reinterpret_cast<char*>(mat.data.data()), sizeof(double) * load_nnz);
        if (load_nnz < total_nnz) {
            // Seek past remaining data
            file.seekg(sizeof(double) * (total_nnz - load_nnz), std::ios::cur);
        }

        // Read targets y
        mat.y.resize(load_m);
        file.read(reinterpret_cast<char*>(mat.y.data()), sizeof(double) * load_m);

        if (!file) {
            throw std::runtime_error("Unexpected EOF or read error while loading: " + filepath);
        }

        return mat;
    }
};

#endif // SGD_CSR_HPP
