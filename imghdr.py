# Minimal replacement for removed imghdr (Python 3.13+)

def what(file, h=None):
    if h is None:
        try:
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    h = f.read(32)
            else:
                h = file.read(32)
        except Exception:
            return None

    if h.startswith(b'\xff\xd8'):
        return 'jpeg'
    if h.startswith(b'\x89PNG'):
        return 'png'
    if h.startswith(b'GIF'):
        return 'gif'
    if h.startswith(b'BM'):
        return 'bmp'

    return None
