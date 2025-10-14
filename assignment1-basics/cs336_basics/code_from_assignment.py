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