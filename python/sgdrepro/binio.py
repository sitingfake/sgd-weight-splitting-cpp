"""
Binary I/O for Compressed Sparse Row (CSR) Datasets.

Implements reading and writing for the SGDCSR01 binary specification:
  - offset 0  : 8-byte magic b"SGDCSR01"
  - offset 8  : m (int64) - number of samples (rows)
  - offset 16 : n (int64) - number of features (columns)
  - offset 24 : nnz (int64) - total non-zero entries
  - offset 32 : indptr (int32[m + 1])
              : indices (int32[nnz]) - sorted ascending per row
              : data (float64[nnz])
              : y (float64[m])
"""

import hashlib
import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp

MAGIC = b"SGDCSR01"
HEADER_FORMAT = "<8sqqq"  # magic (8s), m (q), n (q), nnz (q)
HEADER_SIZE = 32


def sha256_file(filepath: Path | str) -> str:
    """Compute SHA256 hexadecimal hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_csr_binary(
    filepath: Path | str,
    X: sp.csr_matrix,
    y: np.ndarray,
) -> str:
    """
    Serialize a SciPy CSR matrix and target array into SGDCSR01 binary format.

    Parameters
    ----------
    filepath : Path or str
        Destination binary file path.
    X : sp.csr_matrix
        Sparse feature matrix.
    y : np.ndarray
        1D target vector.

    Returns
    -------
    sha256 : str
        SHA256 hex digest of the written file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    m, n = X.shape
    nnz = X.nnz

    if nnz >= (1 << 31) - 1:
        raise ValueError(f"nnz ({nnz}) exceeds 2^31 - 1 limit for int32 indexing")

    indptr = np.ascontiguousarray(X.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(X.indices, dtype=np.int32)
    data = np.ascontiguousarray(X.data, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)

    if len(y_arr) != m:
        raise ValueError(f"Target length ({len(y_arr)}) does not match sample count m ({m})")

    with open(filepath, "wb") as f:
        # Write 32-byte aligned header
        header = struct.pack(HEADER_FORMAT, MAGIC, m, n, nnz)
        f.write(header)

        # Write CSR buffers
        f.write(indptr.tobytes())
        f.write(indices.tobytes())
        f.write(data.tobytes())
        f.write(y_arr.tobytes())

    return sha256_file(filepath)


def read_csr_binary(
    filepath: Path | str,
    max_samples: Optional[int] = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Deserialize an SGDCSR01 binary file into a SciPy CSR matrix and target array.

    Parameters
    ----------
    filepath : Path or str
        Source binary file path.
    max_samples : int, optional
        If specified, loads only the first max_samples rows.

    Returns
    -------
    X : sp.csr_matrix
        Loaded CSR matrix.
    y : np.ndarray
        Loaded target vector.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"File {filepath} is smaller than minimum header size")

        magic, m, n, nnz = struct.unpack(HEADER_FORMAT, header_bytes)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic {magic!r}; expected {MAGIC!r}")

        load_m = m if max_samples is None or max_samples <= 0 else min(m, max_samples)

        # Read indptr
        full_indptr = np.fromfile(f, dtype=np.int32, count=m + 1)
        load_nnz = int(full_indptr[load_m])

        # Read indices
        indices = np.fromfile(f, dtype=np.int32, count=load_nnz)
        if load_nnz < nnz:
            f.seek((nnz - load_nnz) * 4, 1)

        # Read data
        data = np.fromfile(f, dtype=np.float64, count=load_nnz)
        if load_nnz < nnz:
            f.seek((nnz - load_nnz) * 8, 1)

        # Read targets y
        y = np.fromfile(f, dtype=np.float64, count=load_m)

    indptr = full_indptr[: load_m + 1]
    X = sp.csr_matrix((data, indices, indptr), shape=(load_m, n))
    return X, y
