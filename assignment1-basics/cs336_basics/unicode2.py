# 2.2 Unicode Encodings
# Problem (unicode2): Unicode Encodings (3 points)
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(decode_utf8_bytes_to_str_wrong(test_string.encode("utf-8")))

b = bytes([255, 255])
# print(b.decode("utf-32"))