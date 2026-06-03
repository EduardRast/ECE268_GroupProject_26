from time import perf_counter
import timeit
from functools import wraps

from sphincs_cpu import SPHINCS




def sphincs_test(msg: str):
    """
    Tests the basic functionality 
    """

    sphincs = SPHINCS()

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

    print('Running basic SPHINCS test:')
    # sphincs_test('Test message for SPHINCS+ implementation.')
    execution_time = timeit.timeit(lambda: sphincs_test('Test message for SPHINCS+ implementation.'), number=1)
    print('Runtime: {:.5f} seconds'.format(execution_time))


if __name__ == "__main__":
    main()