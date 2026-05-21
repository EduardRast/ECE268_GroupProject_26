import hashlib
import os


HASH_SIZE = 32
NUM_BITS = 256


def sha256(data):
    return hashlib.sha256(data).digest()


def bytes_to_bits(data):
    bits = []

    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)

    return bits


def keygen():
    private_key = []
    public_key = []

    for _ in range(NUM_BITS):
        sk0 = os.urandom(HASH_SIZE)
        sk1 = os.urandom(HASH_SIZE)

        pk0 = sha256(sk0)
        pk1 = sha256(sk1)

        private_key.append((sk0, sk1))
        public_key.append((pk0, pk1))

    return public_key, private_key


def sign(message, private_key):
    digest = sha256(message)
    bits = bytes_to_bits(digest)

    signature = []

    for i in range(NUM_BITS):
        bit = bits[i]

        signature.append(private_key[i][bit])

    return signature


def verify(message, signature, public_key):
    digest = sha256(message)
    bits = bytes_to_bits(digest)

    print("Message hash:")
    print(digest.hex())
    print()

    for i in range(NUM_BITS):
        bit = bits[i]

        hashed_sig = sha256(signature[i])
        expected = public_key[i][bit]

        if hashed_sig != expected:
            print("Verification FAILED at bit:", i)
            return False

    print("Verification SUCCESS")
    return True


def main():
    message = b"hello world"

    print("Original message:")
    print(message)
    print()

    print("Generating keys...")
    public_key, private_key = keygen()

    print("Signing message...")
    signature = sign(message, private_key)

    print("\n--- VERIFY CORRECT MESSAGE ---")
    print("Message used:")
    print(message)

    print("Message hash:")
    print(sha256(message).hex())

    valid = verify(message, signature, public_key)
    print("Valid:", valid)

    tampered_message = b"hello world!"

    print("\n--- VERIFY TAMPERED MESSAGE ---")
    print("Original signed message:")
    print(message)

    print("Tampered message:")
    print(tampered_message)

    print("Original hash:")
    print(sha256(message).hex())

    print("Tampered hash:")
    print(sha256(tampered_message).hex())

    valid = verify(tampered_message, signature, public_key)
    print("Valid:", valid)


if __name__ == "__main__":
    main()