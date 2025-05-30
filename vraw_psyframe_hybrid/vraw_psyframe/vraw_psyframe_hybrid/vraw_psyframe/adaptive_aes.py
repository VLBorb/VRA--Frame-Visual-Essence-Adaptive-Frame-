# vraw_psyframe_hybrid/vraw_psyframe/adaptive_aes.py

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import json # For serializing dictionaries if needed

class AdaptiveAES:
    """
    Derives a 256-bit AES key by iterating the chaotic logistic map:
        x_{n+1} = 3.99 * x_n * (1 - x_n)
    Encrypts and decrypts data (bytes or serializable Python objects).
    """
    def __init__(self, freq_identity: float):
        """
        Initializes AES with a key derived from freq_identity.
        :param freq_identity: A float value (e.g., 0.0 to 1.0) to seed key generation.
                              Should be kept secret and consistent for decryption.
        """
        if not (0.0 <= freq_identity <= 1.0):
            # print("Warning: freq_identity is typically within [0,1] for logistic map stability.")
            # Allow it, but it might produce less chaotic keys if outside common range.
            pass
        self.key = self._generate_key(freq_identity)

    def _generate_key(self, freq_identity: float) -> bytes:
        x = freq_identity
        key_bytes = bytearray()
        for _ in range(32): # 32 bytes = 256 bits
            x = 3.99 * x * (1.0 - x)
            # Clamp to [0,1] to ensure stability if initial x is outside,
            # or if slight floating point errors occur.
            x = max(0.0, min(1.0, x))
            key_bytes.append(int(x * 255)) # Use 255 to stay within byte range properly
        return bytes(key_bytes)

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts byte data using AES CBC mode.
        :param data: Data to encrypt (bytes).
        :return: Encrypted data (IV prepended).
        """
        cipher = AES.new(self.key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data, AES.block_size))
        return cipher.iv + ct_bytes # Prepend IV

    def decrypt(self, enc_data: bytes) -> bytes:
        """
        Decrypts data encrypted with AES CBC mode.
        Assumes IV is prepended to the ciphertext.
        :param enc_data: Encrypted data (IV prepended).
        :return: Decrypted data (bytes).
        """
        iv = enc_data[:AES.block_size]
        ct = enc_data[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt

    def encrypt_dict(self, data_dict: dict) -> bytes:
        """
        Serializes a dictionary to JSON, encodes to UTF-8, then encrypts.
        :param data_dict: Dictionary to encrypt.
        :return: Encrypted data (bytes).
        """
        json_data = json.dumps(data_dict, sort_keys=True) # sort_keys for consistent serialization
        byte_data = json_data.encode('utf-8')
        return self.encrypt(byte_data)

    def decrypt_to_dict(self, enc_data: bytes) -> dict:
        """
        Decrypts data and deserializes from JSON UTF-8 to a dictionary.
        :param enc_data: Encrypted data (bytes).
        :return: Decrypted dictionary.
        """
        decrypted_bytes = self.decrypt(enc_data)
        json_data = decrypted_bytes.decode('utf-8')
        return json.loads(json_data)

if __name__ == '__main__':
    aes_cipher = AdaptiveAES(freq_identity=0.123456789)
    
    # Test with bytes
    secret_msg_bytes = b"This is a secret message in bytes!"
    encrypted_bytes = aes_cipher.encrypt(secret_msg_bytes)
    decrypted_bytes = aes_cipher.decrypt(encrypted_bytes)
    print(f"Original Bytes: {secret_msg_bytes}")
    print(f"Encrypted Bytes (sample): {encrypted_bytes[:32]}...")
    print(f"Decrypted Bytes: {decrypted_bytes}")
    assert secret_msg_bytes == decrypted_bytes, "Byte encryption/decryption failed!"
    print("Byte encryption/decryption successful.\n")

    # Test with dictionary
    secret_dict = {"user_id": 123, "data": "sensitive info", "value": 3.14159}
    encrypted_dict_data = aes_cipher.encrypt_dict(secret_dict)
    decrypted_dict = aes_cipher.decrypt_to_dict(encrypted_dict_data)
    print(f"Original Dict: {secret_dict}")
    print(f"Encrypted Dict Data (sample): {encrypted_dict_data[:32]}...")
    print(f"Decrypted Dict: {decrypted_dict}")
    assert secret_dict == decrypted_dict, "Dictionary encryption/decryption failed!"
    print("Dictionary encryption/decryption successful.")
