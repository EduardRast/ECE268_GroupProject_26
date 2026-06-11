import os
import math
import time
import hashlib
import cupy as cp

import timeit

from sphincs_cpu import prf_message, hash_message


def base_w(x, w, out_len):
    """
    Converts a byte array into an array of base-w integers.
    """
    vin = 0
    vout = 0
    total = 0
    bits = 0
    basew = []

    for consumed in range(out_len):
        if bits == 0:
            total = x[vin]
            vin += 1
            bits += 8
        bits -= math.floor(math.log(w, 2))
        basew.append((total >> bits) % w)
        vout += 1

    return basew

def cpu_golden_wots_sign(message, secret_seed, pub_seed, base_addr, leaf_idx, w, len_0):
    """
    Computes a WOTS+ signature for a specific message and leaf index.
    """
    # Calculate SPHINCS+ length parameters based on security size
    n = len(message)
    len_1 = math.ceil(8 * n / math.log(w, 2))
    len_2 = math.floor(math.log(len_1 * (w - 1), 2) / math.log(w, 2)) + 1

    # Convert message to base-w
    msg_base_w = base_w(message, w, len_1)

    # Compute Winternitz checksum
    csum = 0
    for i in range(len_1):
        csum += w - 1 - msg_base_w[i]

    padding = (len_2 * math.floor(math.log(w, 2))) % 8
    if padding == 0:
        padding = 8

    csum = csum << (8 - padding)
    csumb = csum.to_bytes(math.ceil((len_2 * math.floor(math.log(w, 2))) / 8), byteorder='big')

    # Append checksum to base_w message array
    csumw = base_w(csumb, w, len_2)
    msg_base_w.extend(csumw)

    # Generate signature chains
    sig = bytearray()

    # Configure ADDR object for WOTS+ key pair
    leaf_addr = bytearray(base_addr)
    leaf_addr[20:24] = leaf_idx.to_bytes(4, 'big') # key_pair_addr

    for i in range(len_0):
        addr = bytearray(leaf_addr)
        addr[27] = i  # chain_addr

        # PRF: generate starting secret key for this chain
        hasher = hashlib.sha256()
        hasher.update(secret_seed + addr)
        current_hash = hasher.digest()

        # Signature loop: hash msg_base_w[i] times
        for step in range(msg_base_w[i]):
            addr[31] = step # hash_addr

            hasher = hashlib.sha256()
            hasher.update(pub_seed + addr + current_hash)
            current_hash = hasher.digest()

        sig.extend(current_hash)

    return bytes(sig)

def cpu_golden_fors_pk_from_sig(sk, auth_path, pub_seed, base_addr, tree_idx, target_leaf, t, a):
    """
    Python reference to rebuild a FORS root from a signature
    """
    node = sk

    addr = bytearray(base_addr)
    addr[19] = 3 # FORS_TREE

    # Hash SK into leaf node
    addr[24:28] = (0).to_bytes(4, 'big') # height = 0
    addr[28:32] = (tree_idx * t + target_leaf).to_bytes(4, 'big')

    hasher = hashlib.sha256()
    hasher.update(pub_seed + addr + node)
    node = hasher.digest()

    # Climb tree using the authentication path
    for j in range(a):
        addr[24:28] = (j + 1).to_bytes(4, 'big') # current height

        parent_idx_in_layer = target_leaf // (2 ** (j + 1))
        addr[28:32] = (tree_idx * t + parent_idx_in_layer).to_bytes(4, 'big')

        auth_node = auth_path[j]
        hasher = hashlib.sha256()

        if (target_leaf // (2 ** j)) % 2 == 0:
            hasher.update(pub_seed + addr + node + auth_node) # node is left child
        else:
            hasher.update(pub_seed + addr + auth_node + node) # node is right child

        node = hasher.digest()

    return node

class SPHINCS_GPU:
    def __init__(self, cuda_source_string):
        """
        Initializes the GPU kernels and SPHINCS+ 128s parameters.
        """

        print("Compiling SPHINCS+ GPU Hardware Architecture...")
        self.module = cp.RawModule(code=cuda_source_string)

        # Load all execution units
        self.fors_leaves_k = self.module.get_function('fors_leaves_kernel')
        self.fors_reduce_k = self.module.get_function('fors_reduction_kernel')
        self.wots_gen_k    = self.module.get_function('wots_leaves_gen_kernel')
        self.xmss_reduce_k = self.module.get_function('xmss_tree_reduction_kernel')
        self.wots_verify_k = self.module.get_function('wots_pk_from_sig_kernel')

        # Update 128s parameters to match 32-byte hardware hashes
        self.n = 32
        self.h = 64
        self.d = 8
        self.a = 15
        self.k = 10
        self.w = 16
        self.t = 2 ** self.a

        # WOTS+ lengths for n=32 and w=16
        self.len_1 = 64
        self.len_2 = 3
        self.len_0 = 67
        self.h_prime = self.h // self.d


    def keygen(self) -> bytearray:
        """
        Executes KeyGen entirely in VRAM. Returns (SK, PK).
        """
        print("\n--- Executing GPU Key Generation ---")
        secret_seed = os.urandom(self.n)
        secret_prf = os.urandom(self.n)
        pub_seed = os.urandom(self.n)

        d_secret_seed = cp.array(bytearray(secret_seed), dtype=cp.uint8)
        d_pub_seed = cp.array(bytearray(pub_seed), dtype=cp.uint8)

        base_addr = bytearray(32)
        base_addr[0:4] = (self.d - 1).to_bytes(4, 'big')
        d_base_addr = cp.array(base_addr, dtype=cp.uint8)

        d_xmss_leaves = cp.zeros(256 * 32, dtype=cp.uint8)
        d_pub_root = cp.zeros(32, dtype=cp.uint8)

        dummy_target_leaf = 0
        d_dummy_auth = cp.zeros(self.h_prime * 32, dtype=cp.uint8)

        self.wots_gen_k((256,), (self.len_0,),
            (d_secret_seed, d_pub_seed, d_base_addr, d_xmss_leaves, cp.int32(self.w), cp.int32(self.len_0), cp.int32(256)))

        self.xmss_reduce_k((1,), (128,),
            (d_xmss_leaves, d_pub_seed, d_base_addr, cp.int32(dummy_target_leaf), d_pub_root, d_dummy_auth))

        cp.cuda.Stream.null.synchronize()
        pub_root = d_pub_root.get().tobytes()

        sk = secret_seed + secret_prf + pub_seed + pub_root
        pk = pub_seed + pub_root
        return sk, pk


    def sign(self, message, sk) -> bytes:
        """
        Computes FORS and Hypertree signatures using parallel grids.
        """
        print("\n--- Executing GPU Signature Generation ---")
        sec_seed = sk[:self.n]
        sec_prf = sk[self.n : 2*self.n]
        pub_seed = sk[2*self.n : 3*self.n]
        pub_root = sk[3*self.n:]

        opt = os.urandom(self.n)
        r = prf_message(sec_prf, opt, message, self.n)

        size_md = math.floor((self.k * self.a + 7) / 8)
        size_idx_tree = math.floor((self.h - self.h_prime + 7) / 8)
        size_idx_leaf = math.floor((self.h_prime + 7) / 8)

        digest = hash_message(r, pub_seed, pub_root, message, size_md + size_idx_tree + size_idx_leaf)

        tmp_md = digest[:size_md]
        tmp_idx_tree = digest[size_md : size_md + size_idx_tree]
        tmp_idx_leaf = digest[size_md + size_idx_tree : len(digest)]

        md_int = int.from_bytes(tmp_md, 'big') >> (len(tmp_md) * 8 - self.k * self.a)
        idx_tree = int.from_bytes(tmp_idx_tree, 'big') >> (len(tmp_idx_tree) * 8 - (self.h - self.h_prime))
        idx_leaf = int.from_bytes(tmp_idx_leaf, 'big') >> (len(tmp_idx_leaf) * 8 - self.h_prime)

        target_indices = []
        for i in range(self.k):
            target_indices.append((md_int >> (self.k - 1 - i) * self.a) % self.t)

        d_secret_seed = cp.array(bytearray(sec_seed), dtype=cp.uint8)
        d_pub_seed = cp.array(bytearray(pub_seed), dtype=cp.uint8)

        d_target_indices = cp.array(target_indices, dtype=cp.int32)
        d_fors_tree_nodes = cp.zeros(self.k * 65536 * 32, dtype=cp.uint8)
        d_fors_sks = cp.zeros(self.k * 32, dtype=cp.uint8)
        d_fors_auths = cp.zeros(self.k * self.a * 32, dtype=cp.uint8)
        d_fors_roots = cp.zeros(self.k * 32, dtype=cp.uint8)

        base_addr = bytearray(32)
        base_addr[4:16] = idx_tree.to_bytes(12, 'big')
        base_addr[16:20] = (0).to_bytes(4, 'big')
        base_addr[20:24] = idx_leaf.to_bytes(4, 'big')
        d_base_addr = cp.array(base_addr, dtype=cp.uint8)

        self.fors_leaves_k(
            (128, self.k), (256,),
            (d_secret_seed, d_pub_seed, d_base_addr, d_target_indices, d_fors_tree_nodes, d_fors_sks, cp.int32(self.t))
        )
        self.fors_reduce_k(
            (self.k,), (512,),
            (d_fors_tree_nodes, d_pub_seed, d_base_addr, d_target_indices, d_fors_roots, d_fors_auths, cp.int32(self.t), cp.int32(self.a))
        )
        cp.cuda.Stream.null.synchronize()

        fors_roots = d_fors_roots.get().tobytes()
        pk_fors_addr = bytearray(base_addr)
        pk_fors_addr[19] = 4

        hasher = hashlib.sha256()
        hasher.update(pub_seed + pk_fors_addr + fors_roots)
        pk_fors = hasher.digest()

        # Interleave the FORS signature chunks
        sks = d_fors_sks.get().tobytes()
        auths = d_fors_auths.get().tobytes()
        sig_fors_array = bytearray()
        for i in range(self.k):
            sig_fors_array.extend(sks[i * self.n : (i + 1) * self.n])
            sig_fors_array.extend(auths[i * self.a * self.n : (i + 1) * self.a * self.n])
        sig_fors = bytes(sig_fors_array)

        sig_ht = bytearray()
        current_message = pk_fors

        d_xmss_leaves = cp.zeros(256 * 32, dtype=cp.uint8)
        d_xmss_root = cp.zeros(32, dtype=cp.uint8)
        d_xmss_auth = cp.zeros(self.h_prime * 32, dtype=cp.uint8)

        current_idx_tree = idx_tree
        current_idx_leaf = idx_leaf

        for j in range(self.d):
            layer_addr = bytearray(32)
            layer_addr[0:4] = j.to_bytes(4, 'big')
            layer_addr[4:16] = current_idx_tree.to_bytes(12, 'big')
            d_layer_addr = cp.array(layer_addr, dtype=cp.uint8)

            self.wots_gen_k(
                (256,), (self.len_0,),
                (d_secret_seed, d_pub_seed, d_layer_addr, d_xmss_leaves, cp.int32(self.w), cp.int32(self.len_0), cp.int32(256))
            )

            self.xmss_reduce_k(
                (1,), (128,),
                (d_xmss_leaves, d_pub_seed, d_layer_addr, cp.int32(current_idx_leaf), d_xmss_root, d_xmss_auth)
            )
            cp.cuda.Stream.null.synchronize()

            wots_sig_bytes = cpu_golden_wots_sign(current_message, sec_seed, pub_seed, layer_addr, current_idx_leaf, self.w, self.len_0)

            extracted_auth_path = d_xmss_auth.get().tobytes()
            sig_ht.extend(wots_sig_bytes + extracted_auth_path)

            current_message = d_xmss_root.get().tobytes()

            current_idx_leaf = current_idx_tree % (2 ** self.h_prime)
            current_idx_tree = current_idx_tree >> self.h_prime

        return r + sig_fors + bytes(sig_ht)


    def verify(self, message, signature, pk) -> bool:
        """
        Reconstructs the root nodes layer-by-layer to verify the signature.
        """
        print("\n--- Executing GPU Signature Verification ---")
        pub_seed = pk[:self.n]
        pub_root = pk[self.n:]

        r = signature[:self.n]

        fors_sig_len = self.k * (1 + self.a) * self.n
        sig_fors = signature[self.n : self.n + fors_sig_len]
        sig_ht = signature[self.n + fors_sig_len :]

        size_md = math.floor((self.k * self.a + 7) / 8)
        size_idx_tree = math.floor((self.h - self.h_prime + 7) / 8)
        size_idx_leaf = math.floor((self.h_prime + 7) / 8)

        digest = hash_message(r, pub_seed, pub_root, message, size_md + size_idx_tree + size_idx_leaf)

        tmp_md = digest[:size_md]
        tmp_idx_tree = digest[size_md : size_md + size_idx_tree]
        tmp_idx_leaf = digest[size_md + size_idx_tree : len(digest)]

        md_int = int.from_bytes(tmp_md, 'big') >> (len(tmp_md) * 8 - self.k * self.a)
        idx_tree = int.from_bytes(tmp_idx_tree, 'big') >> (len(tmp_idx_tree) * 8 - (self.h - self.h_prime))
        idx_leaf = int.from_bytes(tmp_idx_leaf, 'big') >> (len(tmp_idx_leaf) * 8 - self.h_prime)

        target_indices = []
        for i in range(self.k):
            target_indices.append((md_int >> (self.k - 1 - i) * self.a) % self.t)

        d_pub_seed = cp.array(bytearray(pub_seed), dtype=cp.uint8)

        recovered_fors_roots = bytearray()

        for i in range(self.k):
            sk_offset = i * (self.a + 1) * self.n
            sk = sig_fors[sk_offset : sk_offset + self.n]

            auth_path = []
            for j in range(self.a):
                auth_offset = sk_offset + self.n + (j * self.n)
                auth_path.append(sig_fors[auth_offset : auth_offset + self.n])

            base_addr = bytearray(32)
            base_addr[4:16] = idx_tree.to_bytes(12, 'big')
            base_addr[16:20] = (3).to_bytes(4, 'big')
            base_addr[20:24] = idx_leaf.to_bytes(4, 'big')

            root = cpu_golden_fors_pk_from_sig(sk, auth_path, pub_seed, base_addr, i, target_indices[i], self.t, self.a)
            recovered_fors_roots.extend(root)

        pk_fors_addr = bytearray(32)
        pk_fors_addr[4:16] = idx_tree.to_bytes(12, 'big')
        pk_fors_addr[16:20] = (4).to_bytes(4, 'big')
        pk_fors_addr[20:24] = idx_leaf.to_bytes(4, 'big')

        hasher = hashlib.sha256()
        hasher.update(pub_seed + pk_fors_addr + recovered_fors_roots)
        current_message = hasher.digest()

        wots_sig_len = self.len_0 * self.n
        xmss_auth_len = self.h_prime * self.n
        layer_sig_len = wots_sig_len + xmss_auth_len

        current_idx_tree = idx_tree
        current_idx_leaf = idx_leaf

        for j in range(self.d):
            layer_addr = bytearray(32)
            layer_addr[0:4] = j.to_bytes(4, 'big')
            layer_addr[4:16] = current_idx_tree.to_bytes(12, 'big')

            sig_offset = j * layer_sig_len
            wots_sig = sig_ht[sig_offset : sig_offset + wots_sig_len]
            xmss_auth_path = sig_ht[sig_offset + wots_sig_len : sig_offset + layer_sig_len]

            d_wots_sig = cp.array(bytearray(wots_sig), dtype=cp.uint8)

            msg_base_w = base_w(current_message, self.w, self.len_1)
            csum = sum([self.w - 1 - x for x in msg_base_w])
            padding = (self.len_2 * math.floor(math.log(self.w, 2))) % 8 or 8
            csum = csum << (8 - padding)
            csumb = csum.to_bytes(math.ceil((self.len_2 * math.floor(math.log(self.w, 2))) / 8), 'big')
            msg_base_w.extend(base_w(csumb, self.w, self.len_2))

            d_msg_base_w = cp.array(msg_base_w, dtype=cp.uint8)
            d_wots_pk = cp.zeros(32, dtype=cp.uint8)

            layer_addr[20:24] = current_idx_leaf.to_bytes(4, 'big')
            d_layer_addr = cp.array(layer_addr, dtype=cp.uint8)

            self.wots_verify_k(
                (1,), (self.len_0,),
                (d_wots_sig, d_msg_base_w, d_pub_seed, d_layer_addr, d_wots_pk, cp.int32(self.w), cp.int32(self.len_0))
            )
            cp.cuda.Stream.null.synchronize()
            wots_pk = d_wots_pk.get().tobytes()

            node = wots_pk
            layer_addr[19] = 2
            layer_addr[20:24] = (0).to_bytes(4, 'big')
            layer_addr[28:32] = current_idx_leaf.to_bytes(4, 'big')

            for height in range(self.h_prime):
                layer_addr[27] = height + 1
                parent_idx = current_idx_leaf // (2 ** (height + 1))
                layer_addr[28:32] = parent_idx.to_bytes(4, 'big')

                auth_node = xmss_auth_path[height * self.n : (height + 1) * self.n]
                hasher = hashlib.sha256()

                if (current_idx_leaf // (2 ** height)) % 2 == 0:
                    hasher.update(pub_seed + layer_addr + node + auth_node)
                else:
                    hasher.update(pub_seed + layer_addr + auth_node + node)
                node = hasher.digest()

            current_message = node

            current_idx_leaf = current_idx_tree % (2 ** self.h_prime)
            current_idx_tree = current_idx_tree >> self.h_prime

        is_valid = (current_message == pub_root)

        if is_valid:
            print("Verification PASS: data paths matched the expected root perfectly.")
        else:
            print("Verification FAIL: mathematical desynchronization detected.")

        return is_valid

def sphincs_gpu_test(msg: str):
    """
    Tests the basic functionality
    """

    sphincs = SPHINCS_GPU(cuda_source)

    # Generate key pair (secret key and public key)
    sk, pk = sphincs.keygen()

    # Sign the message and get the signature
    message = msg.encode()
    signature = sphincs.sign(message, sk)

    # Verify the signature
    if sphincs.verify(message, signature, pk):
        print('Verification PASSED.')
    else:
        print('Verification FAILED.')

