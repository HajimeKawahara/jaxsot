""" test for io

"""

import numpy as np
import pytest
from jaxsot.io.earth import binarymap, multibandmap

def test_binarymap():
    mmap=binarymap(nside=16,show=False)
    assert len(mmap[mmap==1.0])==994
    
def test_multibandmap():
    mmap, A, X = multibandmap(show=False)
    assert np.sum(mmap) == pytest.approx(17147.237680965998)
    assert np.sum(A) == pytest.approx(12288.0)
    assert np.sum(X) == pytest.approx(6.5126960630726485)

if __name__=="__main__":
    test_binarymap()
    test_multibandmap()
