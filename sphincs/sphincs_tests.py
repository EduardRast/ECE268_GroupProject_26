import timeit

from sphincs_cpu import SPHINCS_CPU


def sphincs_cpu_test(msg: str):
    """
    Tests the basic functionality 
    """

    sphincs = SPHINCS_CPU()

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


def main():
    """
    The main function that executes the entire script
    """

    print('Running basic SPHINCS CPU test:')
    cpu_runtime = timeit.timeit(lambda: sphincs_cpu_test('Test message for SPHINCS+ implementation.'), number=1)
    print('Runtime: {:.5f} seconds'.format(cpu_runtime))


if __name__ == "__main__":
    main()