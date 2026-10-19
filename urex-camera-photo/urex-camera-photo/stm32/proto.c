/* proto.c - frame codec for the camera link (no HAL calls here). */
#include "proto.h"
#include "bsp.h"

#define SOF 0xA5

static uint8_t crc8(const uint8_t *p, uint32_t n, uint8_t c)
{
    while (n--) {
        c ^= *p++;
        for (int i = 0; i < 8; i++)
            c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
    }
    return c;
}

int proto_send(uint8_t cmd, const void *payload, uint16_t len)
{
    uint8_t h[4] = { SOF, cmd, (uint8_t)len, (uint8_t)(len >> 8) };
    uint8_t c = crc8((const uint8_t *)payload, len, crc8(&h[1], 3, 0));
    int err = bsp_cam_tx(h, 4);
    if (!err) err = bsp_cam_tx(payload, len);
    if (!err) err = bsp_cam_tx(&c, 1);
    return err;
}

int proto_recv(frame_t *f, uint16_t cap, uint32_t timeout_ms)
{
    uint8_t h[3], c;

    do {                                            /* hunt for the start byte */
        if (bsp_cam_rx(h, 1, timeout_ms)) return -1;
    } while (h[0] != SOF);

    if (bsp_cam_rx(h, 3, timeout_ms)) return -1;
    f->cmd = h[0];
    f->len = (uint16_t)(h[1] | (h[2] << 8));
    if (f->len > cap) return -2;

    if (bsp_cam_rx(f->buf, f->len, timeout_ms)) return -1;
    if (bsp_cam_rx(&c, 1, timeout_ms)) return -1;

    return crc8(f->buf, f->len, crc8(h, 3, 0)) == c ? 0 : -3;
}
