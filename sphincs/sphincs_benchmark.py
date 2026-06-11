import time
import cupy as cp
from sphincs_cpu import SPHINCS_CPU
from sphincs_gpu import SPHINCS_GPU

cuda_source = r'''
extern "C" {
    __constant__ unsigned int K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    #define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
    #define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
    #define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
    #define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
    #define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
    #define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

    __device__ void sha256_compress(unsigned int* state, const unsigned char* block) {
        unsigned int a, b, c, d, e, f, g, h, t1, t2;
        unsigned int W[64];

        for (int i = 0; i < 16; ++i) {
            W[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) | (block[i * 4 + 2] << 8) | (block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
        }

        a = state[0]; b = state[1]; c = state[2]; d = state[3];
        e = state[4]; f = state[5]; g = state[6]; h = state[7];

        for (int i = 0; i < 64; ++i) {
            t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
            t2 = EP0(a) + MAJ(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    __device__ void sha256_96bytes(const unsigned char* pub_seed, const unsigned char* addr, const unsigned char* in_hash, unsigned char* out_hash) {
        unsigned int state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        // Block 1: 32 bytes pub_seed + 32 bytes addr
        unsigned char block1[64];
        for (int i = 0; i < 32; i++) block1[i] = pub_seed[i];
        for (int i = 0; i < 32; i++) block1[32 + i] = addr[i];
        sha256_compress(state, block1);

        // Block 2: 32 bytes in_hash + padding
        unsigned char block2[64] = {0};
        for (int i = 0; i < 32; i++) block2[i] = in_hash[i];

        block2[32] = 0x80; // Append '1' bit
        // Length of 96 bytes = 768 bits = 0x0300
        block2[62] = 0x03;
        block2[63] = 0x00;

        sha256_compress(state, block2);

        // Extract final state to out_hash
        for (int i = 0; i < 8; i++) {
            out_hash[i * 4]     = (state[i] >> 24) & 0xFF;
            out_hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
            out_hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
            out_hash[i * 4 + 3] = (state[i]) & 0xFF;
        }
    }

    // Fully generalized variable-length SHA-256 device function
    __device__ void sha256_device(const unsigned char* msg, int msg_len, unsigned char* out_hash) {
        // Initialize standard SHA-256 state
        unsigned int state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        int num_full_blocks = msg_len / 64;
        int remainder = msg_len % 64;

        // Process all full 64-byte blocks
        for (int i = 0; i < num_full_blocks; i++) {
            sha256_compress(state, msg + (i * 64));
        }

        // Handle padding
        unsigned char buffer[64] = {0}; // Initialize with zeros

        // Copy the remaining bytes into the buffer
        for (int i = 0; i < remainder; i++) {
            buffer[i] = msg[(num_full_blocks * 64) + i];
        }

        // Append the '1' bit
        buffer[remainder] = 0x80;

        // Check the block spill edge case
        // If remainder >= 56, we don't have 8 bytes left for the length integer
        if (remainder >= 56) {
            sha256_compress(state, buffer);

            // Clear buffer for the final length block
            for (int i = 0; i < 64; i++) buffer[i] = 0;
        }

        // Append message length in bits as a 64-bit big-endian integer
        unsigned long long bit_len = (unsigned long long)msg_len * 8;
        buffer[56] = (bit_len >> 56) & 0xFF;
        buffer[57] = (bit_len >> 48) & 0xFF;
        buffer[58] = (bit_len >> 40) & 0xFF;
        buffer[59] = (bit_len >> 32) & 0xFF;
        buffer[60] = (bit_len >> 24) & 0xFF;
        buffer[61] = (bit_len >> 16) & 0xFF;
        buffer[62] = (bit_len >> 8) & 0xFF;
        buffer[63] = (bit_len) & 0xFF;

        // Compress the final block
        sha256_compress(state, buffer);

        // Extract output
        for (int i = 0; i < 8; i++) {
            out_hash[i * 4]     = (state[i] >> 24) & 0xFF;
            out_hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
            out_hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
            out_hash[i * 4 + 3] = (state[i]) & 0xFF;
        }
    }

    // Simple test kernel wrapper
    __global__ void test_variable_sha(const unsigned char* msg, int msg_len, unsigned char* out_hash) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) {
            sha256_device(msg, msg_len, out_hash);
        }
    }

    // PRF: SHA-256(secret_seed || ADDR)
    __device__ void prf_device(const unsigned char* secret_seed, const unsigned char* addr, unsigned char* out_sk) {
        unsigned char prf_buffer[64];
        for(int i=0; i<32; i++) prf_buffer[i] = secret_seed[i];
        for(int i=0; i<32; i++) prf_buffer[32+i] = addr[i];

        // Use our variable-length hasher for the 64-byte block
        sha256_device(prf_buffer, 64, out_sk);
    }

    __global__ void wots_leaves_gen_kernel(
        const unsigned char* secret_seed,
        const unsigned char* pub_seed,
        const unsigned char* base_addr,
        unsigned char* out_leaves,  // array of 256 * 32 bytes
        int w,
        int len_0,
        int num_leaves              // 256
    ) {
        int leaf_idx = blockIdx.x;  // identifies the WOTS+ key this block is making
        int tid = threadIdx.x;      // identifies the hash chain (0 to 66)

        if (leaf_idx >= num_leaves) return;

        // Fast shared memory for this specific block's 67 top nodes
        __shared__ unsigned char top_nodes[2144];

        if (tid < len_0) {
            unsigned char current_hash[32];
            unsigned char local_addr[32];

            // Thread address setup
            for(int i = 0; i < 32; i++) {
                local_addr[i] = base_addr[i];
            }

            // Set key_pair_addr (word1 -> bytes 20-23) to the current block/leaf index
            local_addr[20] = (unsigned char)(leaf_idx >> 24);
            local_addr[21] = (unsigned char)(leaf_idx >> 16);
            local_addr[22] = (unsigned char)(leaf_idx >> 8);
            local_addr[23] = (unsigned char)(leaf_idx);

            // Set chain_addr (word2 -> bytes 24-27)
            local_addr[24] = 0; local_addr[25] = 0; local_addr[26] = 0;
            local_addr[27] = (unsigned char)tid;

            // Reset hash_addr (word3 -> bytes 28-31)
            local_addr[28] = 0; local_addr[29] = 0; local_addr[30] = 0; local_addr[31] = 0;

            // Generate starting secret key
            prf_device(secret_seed, local_addr, current_hash);

            // Parallel Winternitz hash chain
            for (int step = 0; step < w - 1; step++) {
                local_addr[31] = (unsigned char)step;
                sha256_96bytes(pub_seed, local_addr, current_hash, current_hash);
            }

            // Store in shared memory
            for(int i = 0; i < 32; i++) {
                top_nodes[tid * 32 + i] = current_hash[i];
            }
        }

        // Wait for all chains in block
        __syncthreads();

        // Final reduction: thread 0 of each block computes its own WOTS_PK
        if (tid == 0) {
            unsigned char final_addr[32];
            for(int i = 0; i < 32; i++) final_addr[i] = base_addr[i];

            // Set key_pair_addr for final compression
            final_addr[20] = (unsigned char)(leaf_idx >> 24);
            final_addr[21] = (unsigned char)(leaf_idx >> 16);
            final_addr[22] = (unsigned char)(leaf_idx >> 8);
            final_addr[23] = (unsigned char)(leaf_idx);

            // Set ADDR type to WOTS_PK
            final_addr[16] = 0; final_addr[17] = 0; final_addr[18] = 0; final_addr[19] = 1;

            int total_len = 64 + (len_0 * 32);
            unsigned char pk_buffer[2500];

            for(int i = 0; i < 32; i++) pk_buffer[i] = pub_seed[i];
            for(int i = 0; i < 32; i++) pk_buffer[32+i] = final_addr[i];
            for(int i = 0; i < (len_0 * 32); i++) pk_buffer[64+i] = top_nodes[i];

            unsigned char final_wots_pk[32];
            sha256_device(pk_buffer, total_len, final_wots_pk);

            // Write the final 32 bytes directly into the global leaves array
            for(int i = 0; i < 32; i++) {
                out_leaves[leaf_idx * 32 + i] = final_wots_pk[i];
            }
        }
    }

    __global__ void xmss_tree_reduction_kernel(
        const unsigned char* in_leaves, // 8192-byte array of 256 WOTS+ Public Keys
        const unsigned char* pub_seed,
        const unsigned char* base_addr,
        int target_leaf,                // the specific leaf index (0-255) to authenticate
        unsigned char* out_root,        // the final 32-byte root
        unsigned char* out_auth         // array for 8 authentication nodes (8 * 32 bytes)
    ) {
        int tid = threadIdx.x;

        // 8KB shared memory
        __shared__ unsigned char tree_buffer[8192];

        // Load leaves into shared memory
        for(int i = 0; i < 32; i++) {
            tree_buffer[(tid * 2) * 32 + i] = in_leaves[(tid * 2) * 32 + i];
            tree_buffer[(tid * 2 + 1) * 32 + i] = in_leaves[(tid * 2 + 1) * 32 + i];
        }
        __syncthreads();

        unsigned char local_addr[32];
        for (int i = 0; i < 32; i++) local_addr[i] = base_addr[i];
        local_addr[16] = 0; local_addr[17] = 0; local_addr[18] = 0; local_addr[19] = 2; // tree type

        int num_nodes = 256;
        int h = 0; // current layer height (0 to 7)

        // In-place parallel reduction
        while (num_nodes > 1) {

            // Thread 0 extracts  required authentication sibling before layer is overwritten
            if (tid == 0) {
                // To find sibling, bitwise XOR parent-relative index with 1
                int sibling_idx_in_layer = (target_leaf >> h) ^ 1;

                // Copy 32-byte sibling from shared memory to global output array
                for(int i = 0; i < 32; i++) {
                    out_auth[h * 32 + i] = tree_buffer[sibling_idx_in_layer * 32 + i];
                }
            }

            int active_threads = num_nodes / 2;

            if (tid < active_threads) {
                local_addr[24] = 0; local_addr[25] = 0; local_addr[26] = 0;
                local_addr[27] = (unsigned char)(h + 1);

                local_addr[28] = 0; local_addr[29] = 0;
                local_addr[30] = (unsigned char)(tid >> 8);
                local_addr[31] = (unsigned char)(tid);

                unsigned char left_child[32];
                unsigned char right_child[32];
                for (int i = 0; i < 32; i++) {
                    left_child[i]  = tree_buffer[(tid * 2) * 32 + i];
                    right_child[i] = tree_buffer[(tid * 2 + 1) * 32 + i];
                }

                unsigned char hash_buffer[128];
                for (int i = 0; i < 32; i++) hash_buffer[i]      = pub_seed[i];
                for (int i = 0; i < 32; i++) hash_buffer[32+i]   = local_addr[i];
                for (int i = 0; i < 32; i++) hash_buffer[64+i]   = left_child[i];
                for (int i = 0; i < 32; i++) hash_buffer[96+i]   = right_child[i];

                sha256_device(hash_buffer, 128, &tree_buffer[tid * 32]);
            }

            __syncthreads();
            num_nodes /= 2;
            h++;
        }

        // Extract root
        if (tid == 0) {
            for(int i = 0; i < 32; i++) {
                out_root[i] = tree_buffer[i];
            }
        }
    }

    __global__ void fors_leaves_kernel(
        const unsigned char* secret_seed,
        const unsigned char* pub_seed,
        const unsigned char* base_addr,
        const int* target_indices,     // array of 10 target leaf indices from message digest
        unsigned char* out_tree_nodes, // massive array: 10 trees * 65536 nodes * 32 bytes
        unsigned char* out_sks,        // array of 10 target secret keys (10 * 32 bytes)
        int t                          // 32768
    ) {
        // 2D grid: Y=tree index (0-9), X=leaf index chunks
        int tree_idx = blockIdx.y;
        int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (tree_idx >= 10 || leaf_idx >= t) return;

        unsigned char local_addr[32];
        for (int i = 0; i < 32; i++) local_addr[i] = base_addr[i];

        // Setup ADDR for FORS tree
        // Set type to FORS_TREE (3) -> bytes 16-19
        local_addr[16] = 0; local_addr[17] = 0; local_addr[18] = 0; local_addr[19] = 3;

        // Set tree_idx (word3 -> bytes 28-31) = tree_idx * t + leaf_idx
        int global_leaf_idx = tree_idx * t + leaf_idx;
        local_addr[28] = (unsigned char)(global_leaf_idx >> 24);
        local_addr[29] = (unsigned char)(global_leaf_idx >> 16);
        local_addr[30] = (unsigned char)(global_leaf_idx >> 8);
        local_addr[31] = (unsigned char)(global_leaf_idx);

        // Generate secret key (SK) for this leaf
        unsigned char sk[32];

        // Ensure tree height (word2 -> bytes 24-27) is 0 for PRF
        local_addr[24] = 0; local_addr[25] = 0; local_addr[26] = 0; local_addr[27] = 0;
        prf_device(secret_seed, local_addr, sk);

        // If this is the leaf message digest targets, save SK to global memory
        if (leaf_idx == target_indices[tree_idx]) {
            for(int i = 0; i < 32; i++) out_sks[tree_idx * 32 + i] = sk[i];
        }

        // Hash SK to create Merkle leaf node
        unsigned char node[32];
        unsigned char hash_buffer[64];
        for(int i=0; i<32; i++) hash_buffer[i]    = pub_seed[i];
        for(int i=0; i<32; i++) hash_buffer[32+i] = local_addr[i];

        // sha256_96bytes handles pub_seed + addr + data structure
        sha256_96bytes(pub_seed, local_addr, sk, node);

        // Write to binary heap layout in global memory
        // Leaves for a tree start at index 't' (32768) in heap
        int heap_offset = t + leaf_idx;

        // Calculate absolute byte offset in 20MB buffer
        int out_idx = (tree_idx * 65536 * 32) + (heap_offset * 32);

        for(int i = 0; i < 32; i++) {
            out_tree_nodes[out_idx + i] = node[i];
        }
    }

    __global__ void fors_reduction_kernel(
        unsigned char* tree_nodes,     // 20MB array from previous step
        const unsigned char* pub_seed,
        const unsigned char* base_addr,
        const int* target_indices,     // the 10 leaf indices
        unsigned char* out_roots,      // array for 10 final tree roots (10 * 32 bytes)
        unsigned char* out_auths,      // array for auth paths (10 trees * 15 nodes * 32 bytes)
        int t,                         // 32768
        int a                          // 15
    ) {
        int tree_idx = blockIdx.x; // block ID matches the tree ID (0 to 9)
        int tid = threadIdx.x;     // 0 to 511

        if (tree_idx >= 10) return;

        int target_leaf = target_indices[tree_idx];
        int tree_base_idx = tree_idx * 65536 * 32;

        unsigned char local_addr[32];
        for (int i = 0; i < 32; i++) local_addr[i] = base_addr[i];

        // Set type to FORS_TREE (3) -> bytes 16-19
        local_addr[16] = 0; local_addr[17] = 0; local_addr[18] = 0; local_addr[19] = 3;

        // Traverse up tree layer by layer
        for (int h = 0; h < a; h++) {

            // Thread 0 extracts required authentication sibling
            if (tid == 0) {
                // Find sibling index in current layer
                int sibling_idx_in_layer = (target_leaf >> h) ^ 1;

                // Calculate where this is in global binary heap
                int heap_idx = (1 << (a - h)) + sibling_idx_in_layer;
                int node_byte_idx = tree_base_idx + heap_idx * 32;

                // Save to output auth array
                int auth_byte_idx = (tree_idx * a * 32) + (h * 32);
                for(int i = 0; i < 32; i++) {
                    out_auths[auth_byte_idx + i] = tree_nodes[node_byte_idx + i];
                }
            }

            // Parallel reduction of current layer
            int num_parents = 1 << (a - h - 1);
            int child_layer_start = 1 << (a - h);
            int parent_layer_start = 1 << (a - h - 1);

            // 512 threads stride through layer computing parent hashes
            for (int i = tid; i < num_parents; i += blockDim.x) {

                // Update tree height
                local_addr[24] = 0; local_addr[25] = 0; local_addr[26] = 0;
                local_addr[27] = (unsigned char)(h + 1);

                // Update tree index
                int global_idx = tree_idx * t + i;
                local_addr[28] = (unsigned char)(global_idx >> 24);
                local_addr[29] = (unsigned char)(global_idx >> 16);
                local_addr[30] = (unsigned char)(global_idx >> 8);
                local_addr[31] = (unsigned char)(global_idx);

                int left_child_idx = tree_base_idx + (child_layer_start + 2 * i) * 32;
                int right_child_idx = tree_base_idx + (child_layer_start + 2 * i + 1) * 32;
                int parent_idx = tree_base_idx + (parent_layer_start + i) * 32;

                unsigned char hash_buffer[128];
                for (int j = 0; j < 32; j++) hash_buffer[j]      = pub_seed[j];
                for (int j = 0; j < 32; j++) hash_buffer[32+j]   = local_addr[j];
                for (int j = 0; j < 32; j++) hash_buffer[64+j]   = tree_nodes[left_child_idx + j];
                for (int j = 0; j < 32; j++) hash_buffer[96+j]   = tree_nodes[right_child_idx + j];

                sha256_device(hash_buffer, 128, &tree_nodes[parent_idx]);
            }

            // Ensure all parents are computed before next layer begins
            __syncthreads();
        }

        // Thread 0 extracts final root
        if (tid == 0) {
            int root_idx = tree_base_idx + 1 * 32; // root of a binary heap is always at index 1
            for (int i = 0; i < 32; i++) {
                out_roots[tree_idx * 32 + i] = tree_nodes[root_idx + i];
            }
        }
    }

    __global__ void wots_pk_from_sig_kernel(
        const unsigned char* sig,       // WOTS+ signature (67 * 32 bytes)
        const unsigned char* msg,       // message hash (base_w format)
        const unsigned char* pub_seed,
        const unsigned char* base_addr,
        unsigned char* out_pk,
        int w,
        int len_0
    ) {
        int tid = threadIdx.x; // 0 to 66
        __shared__ unsigned char top_nodes[2144];

        if (tid < len_0) {
            unsigned char current_hash[32];
            unsigned char local_addr[32];
            for(int i = 0; i < 32; i++) local_addr[i] = base_addr[i];

            // Setup chain_addr
            local_addr[24] = 0; local_addr[25] = 0; local_addr[26] = 0;
            local_addr[27] = (unsigned char)tid;

            // Load signature node for chain
            for(int i = 0; i < 32; i++) current_hash[i] = sig[tid * 32 + i];

            // Verification - hash from signature's state up to top node
            int start_step = msg[tid];
            for (int step = start_step; step < w - 1; step++) {
                local_addr[31] = (unsigned char)step;
                sha256_96bytes(pub_seed, local_addr, current_hash, current_hash);
            }

            for(int i = 0; i < 32; i++) top_nodes[tid * 32 + i] = current_hash[i];
        }

        __syncthreads();

        // Thread 0 compresses recovered nodes into WOTS+ PK
        if (tid == 0) {
            unsigned char final_addr[32];
            for(int i = 0; i < 32; i++) final_addr[i] = base_addr[i];

            // Set ADDR type to WOTS_PK (1) at bytes 16-19
            final_addr[16] = 0; final_addr[17] = 0; final_addr[18] = 0; final_addr[19] = 1;

            // Calculate buffer size: pub_seed (32) + final_addr (32) + top_nodes (len_0 * 32)
            int total_len = 64 + (len_0 * 32);
            unsigned char pk_buffer[2500];

            for(int i = 0; i < 32; i++) pk_buffer[i] = pub_seed[i];
            for(int i = 0; i < 32; i++) pk_buffer[32+i] = final_addr[i];
            for(int i = 0; i < (len_0 * 32); i++) pk_buffer[64+i] = top_nodes[i];

            // Compress down to final 32-byte PK
            sha256_device(pk_buffer, total_len, out_pk);
        }
    }
}
'''


def benchmark_sphincs(msg: str):
    """
    Benchmarks KeyGen, Sign, and Verify for CPU vs GPU.
    """
    print("Initializing SPHINCS+ classes...")

    # Initialize both classes
    sphincs_cpu = SPHINCS_CPU() # Adjust initialization if your CPU class requires params
    sphincs_gpu = SPHINCS_GPU(cuda_source)

    message = msg.encode()

    # CPU KeyGen benchmark
    start_cpu = time.perf_counter()
    sk_cpu, pk_cpu = sphincs_cpu.keygen()
    cpu_keygen_ms = (time.perf_counter() - start_cpu) * 1000

    # GPU warmup run to compile kernels and allocate context
    _, _ = sphincs_gpu.keygen()
    cp.cuda.Stream.null.synchronize()

    # GPU benchmark
    start_gpu = time.perf_counter()
    sk_gpu, pk_gpu = sphincs_gpu.keygen()
    cp.cuda.Stream.null.synchronize() # Wait for GPU to finish
    gpu_keygen_ms = (time.perf_counter() - start_gpu) * 1000

    # CPU Sign benchmark
    start_cpu = time.perf_counter()
    sig_cpu = sphincs_cpu.sign(message, sk_cpu)
    cpu_sign_ms = (time.perf_counter() - start_cpu) * 1000

    # GPU warmup
    _ = sphincs_gpu.sign(message, sk_gpu)
    cp.cuda.Stream.null.synchronize()

    # GPU benchmark
    start_gpu = time.perf_counter()
    sig_gpu = sphincs_gpu.sign(message, sk_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_sign_ms = (time.perf_counter() - start_gpu) * 1000

    # CPU Verify benchmark
    start_cpu = time.perf_counter()
    valid_cpu = sphincs_cpu.verify(message, sig_cpu, pk_cpu)
    cpu_verify_ms = (time.perf_counter() - start_cpu) * 1000

    # GPU warmup
    _ = sphincs_gpu.verify(message, sig_gpu, pk_gpu)
    cp.cuda.Stream.null.synchronize()

    # GPU benchmark
    start_gpu = time.perf_counter()
    valid_gpu = sphincs_gpu.verify(message, sig_gpu, pk_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_verify_ms = (time.perf_counter() - start_gpu) * 1000

    # Results table
    if not (valid_cpu and valid_gpu):
        print("\nWARNING: A verification failed during the benchmark run!")

    print("\nOperation      CPU (ms)    GPU (ms)   Speedup")
    print("---------    ---------  ----------  ---------")

    print(f"KeyGen       {cpu_keygen_ms:>9.3f}   {gpu_keygen_ms:>9.3f}   {cpu_keygen_ms/gpu_keygen_ms:>6.1f}x")
    print(f"Sign         {cpu_sign_ms:>9.3f}   {gpu_sign_ms:>9.3f}   {cpu_sign_ms/gpu_sign_ms:>6.1f}x")
    print(f"Verify       {cpu_verify_ms:>9.3f}   {gpu_verify_ms:>9.3f}   {cpu_verify_ms/gpu_verify_ms:>6.1f}x")


def main():
    print('Running SPHINCS+ CPU vs GPU Benchmark...\n')
    benchmark_sphincs('Test message for SPHINCS+ implementation.')

if __name__ == "__main__":
    main()