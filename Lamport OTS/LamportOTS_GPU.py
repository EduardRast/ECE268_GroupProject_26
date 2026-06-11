import hashlib
import os
import cupy as cp
import numpy as np

HASH_SIZE = 32
NUM_BITS = 256

def sha256(data):
    return hashlib.sha256(data).digest()

def keygen():
    """
    Generates keys securely on the CPU, then pushes them to the GPU.
    """
    # 1. Generate on CPU (Must use os.urandom for actual security)
    private_key_cpu = np.empty((NUM_BITS, 2, HASH_SIZE), dtype=np.uint8)
    public_key_cpu = np.empty((NUM_BITS, 2, HASH_SIZE), dtype=np.uint8)

    for i in range(NUM_BITS):
        sk0 = os.urandom(HASH_SIZE)
        sk1 = os.urandom(HASH_SIZE)

        private_key_cpu[i, 0] = np.frombuffer(sk0, dtype=np.uint8)
        private_key_cpu[i, 1] = np.frombuffer(sk1, dtype=np.uint8)

        public_key_cpu[i, 0] = np.frombuffer(sha256(sk0), dtype=np.uint8)
        public_key_cpu[i, 1] = np.frombuffer(sha256(sk1), dtype=np.uint8)

    # 2. Transfer to GPU VRAM
    public_key_gpu = cp.asarray(public_key_cpu)
    private_key_gpu = cp.asarray(private_key_cpu)

    return public_key_gpu, private_key_gpu

def sign(message, private_key_gpu):
    """
    Executes the signature matrix indexing entirely on the GPU.
    """
    digest = sha256(message)
    
    # Push the hashed message to the GPU and unpack it into bits instantly
    byte_arr_gpu = cp.array(bytearray(digest), dtype=cp.uint8)
    bits_gpu = cp.unpackbits(byte_arr_gpu)

    # Vectorized selection executed on the GPU
    # This picks the correct 32-byte hash for all 256 bits simultaneously
    signature_gpu = private_key_gpu[cp.arange(NUM_BITS), bits_gpu]

    return signature_gpu

def verify(message, signature_gpu, public_key_gpu):
    """
    Verifies the signature by coordinating CPU hashing and GPU comparisons.
    """
    digest = sha256(message)
    
    # Unpack message bits on the GPU
    byte_arr_gpu = cp.array(bytearray(digest), dtype=cp.uint8)
    bits_gpu = cp.unpackbits(byte_arr_gpu)

    print("Message hash:")
    print(digest.hex())
    print()

    # Pull the signature back to CPU RAM to hash it
    # (Again, because hashlib cannot read GPU memory)
    signature_cpu = signature_gpu.get()
    hashed_sig_cpu = np.empty((NUM_BITS, HASH_SIZE), dtype=np.uint8)
    
    for i in range(NUM_BITS):
        sig_bytes = signature_cpu[i].tobytes()
        hashed_sig_cpu[i] = np.frombuffer(sha256(sig_bytes), dtype=np.uint8)

    # Push the hashed signature back to the GPU for matrix comparison
    hashed_sig_gpu = cp.asarray(hashed_sig_cpu)
    
    # Ask the GPU what the expected public key values are
    expected_gpu = public_key_gpu[cp.arange(NUM_BITS), bits_gpu]

    # Perform a massive parallel comparison on the GPU
    if not cp.array_equal(hashed_sig_gpu, expected_gpu):
        print("Verification FAILED.")
        return False

    print("Verification SUCCESS")
    return True

def main():
    message = b"hello world"

    print("Generating keys (Transferring to GPU)...")
    public_key_gpu, private_key_gpu = keygen()

    print("Signing message (Executing on GPU)...")
    signature_gpu = sign(message, private_key_gpu)

    print("\n--- VERIFY CORRECT MESSAGE ---")
    valid = verify(message, signature_gpu, public_key_gpu)
    print("Valid:", valid)

    tampered_message = b"hello world!"

    print("\n--- VERIFY TAMPERED MESSAGE ---")
    valid = verify(tampered_message, signature_gpu, public_key_gpu)
    print("Valid:", valid)

