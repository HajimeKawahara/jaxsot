""" test for mockdata

"""

from jaxsot.mockdata.earth import binarymap

def test_binarymap():
    mmap=binarymap(nside=16,show=False)
    assert len(mmap[mmap==1.0])==994

if __name__=="__main__":
    test_binarymap()
