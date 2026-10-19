# link.py - framed UART link to the Nucleo: A5 | cmd | len(LE16) | payload | crc8
# Same frame format as UREX-CMOS-002 Rev B, Listing 1.
import struct
from machine import UART

SOF = 0xA5


def crc8(data, crc=0):
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Link:
    def __init__(self, baud=115200):
        # UART 3 on the OpenMV Cam H7 Plus: P4 = TX, P5 = RX
        self.u = UART(3, baudrate=baud, timeout=50, timeout_char=10)

    def send(self, cmd, payload=b""):
        hdr = struct.pack("<BH", cmd, len(payload))
        self.u.write(bytes([SOF]) + hdr)
        if payload:
            self.u.write(payload)
        self.u.write(bytes([crc8(payload, crc8(hdr))]))

    def recv(self):
        """Return (cmd, payload) for one good frame, or None."""
        b = self.u.read(1)
        if not b or b[0] != SOF:
            return None
        hdr = self.u.read(3)
        if not hdr or len(hdr) != 3:
            return None
        cmd, n = struct.unpack("<BH", hdr)
        body = self.u.read(n + 1)
        if not body or len(body) != n + 1 or crc8(body[:n], crc8(hdr)) != body[n]:
            return None
        return cmd, body[:n]
