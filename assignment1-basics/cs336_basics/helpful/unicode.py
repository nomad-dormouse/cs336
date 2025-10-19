# 2.1 The Unicode Standard
# Problem (unicode1): Understanding Unicode (1 point)
print(chr(0))
print(repr(chr(0)))
print("this is a test" + chr(0) + "string")

# 2.2 Unicode Encodings
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)

print(type(utf8_encoded))

# Get the byte values for the encoded string (integers from 0 to 255).
list(utf8_encoded)

# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))

print(test_string)
print(test_string.encode("utf-8"))
print(test_string.encode("utf-16"))
print(test_string.encode("utf-32"))

# Problem (unicode2): Unicode Encodings (3 points)
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(decode_utf8_bytes_to_str_wrong(test_string.encode("utf-8")))

b = bytes([255, 255])
# print(b.decode("utf-32"))