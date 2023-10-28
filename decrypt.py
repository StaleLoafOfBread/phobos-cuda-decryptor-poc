import sys
from Crypto.Cipher import AES
import struct
import os

# Encrypted file format:
#     data: N bytes
#     footer (metadata): 178 bytes
# Footer (metadata) format:
#      padded w/ zeros: 0-20               (length: 20 bytes)
#      iv: 20-36 bytes                     (length: 16 bytes)
#      padded_size: 36-40 bytes            (length: 4 bytes)
#      encrypted_key: 40-168 bytes         (length: 128 bytes)
#      additional_data_size: 168-172 bytes (length: 4 bytes)
#      attacker_id: 172-178 bytes          (length: 6 bytes)


def decrypt_file(input_path, output_path, key):
    with open(input_path, "rb") as f:
        # Step 1: Seek to the metadata and read it
        f.seek(-178, os.SEEK_END)
        metadata = f.read(178)

        # Extract necessary details from metadata
        iv = metadata[20:36]
        padded_size = struct.unpack("<L", metadata[36:40])[0]
        additional_data_size = struct.unpack("<I", metadata[168:172])[0]

        print(f"IV: {iv}")
        print(f"padded_size: {padded_size}")
        print(f"additional_data_size: {additional_data_size}")

        # Step 2: Calculate the total size of the file
        total_size = f.tell()  # Because we've just read the metadata, our current position is the total size

        # Step 3: Calculate encrypted data size
        data_size = total_size - 178
        print(f"data_size: {data_size}")

        # Reset to the start of the file and read the encrypted data
        f.seek(0)
        enc_data = f.read(data_size)

        # Decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data_with_padding = cipher.decrypt(enc_data)

        # Remove padded_size bytes from the end
        decrypted_data = decrypted_data_with_padding[:-padded_size]

        # Remove the extra metadata added at the end
        # TODO: extract the filename and compare it to the encrypted file's name
        decrypted_data = decrypted_data[:-64]

        # Write decrypted data to the output file.
        with open(output_path, "wb") as fout:
            fout.write(decrypted_data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("You must provide a 128 bit key in hex format!")

    # Remove the 0x prefix if provided
    key = sys.argv[1]
    if key[0:2] == "0x":
        key = key[2:]

    # Convert the key into bytes
    key = bytes.fromhex(key)

    # Actually run the decryption
    decrypt_file("./sample_data/tofu.enc", "./test.jpg", key)
