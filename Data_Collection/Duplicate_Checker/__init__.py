import hashlib


def calc_file_hash(path):
    with open(path, 'rb') as f:
        data = f.read()
    hash = hashlib.md5(data).hexdigest()
    return hash
