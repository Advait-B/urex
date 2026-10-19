/* main.c - UREx CMOS branch: press B1 on the NUCLEO-C542RC to take a photo.
 *
 *   B1 pressed -> PHOTO to the OpenMV camera (it captures + JPEG-compresses)
 *              -> READ the JPEG back in 512-byte pieces
 *              -> pass each piece straight on to the laptop over the ST-LINK
 *                 virtual COM port, where pc/get_photo.py saves a .jpg file.
 *
 * Replace the main.c that STM32CubeMX2 generated with this file.
 *
 * Text sent to the laptop (each line ends in \n):
 *   PHOTO <size> <width> <height>            start of a photo
 *   DATA <offset> <n>  followed by n raw bytes   one piece of the JPEG
 *   DONE <crc32 hex>                         all bytes sent (zlib CRC-32)
 *   FAIL <reason>                            photo abandoned
 *   anything else                            log message for a human
 */
#include <stdio.h>
#include <string.h>
#include "mx_system.h"
#include "bsp.h"
#include "proto.h"

#define JPEG_QUALITY     80        /* 1-100, higher = bigger file             */
#define CHUNK            PROTO_CHUNK_MAX
#define MAX_JPEG_BYTES   (600u * 1024u)
#define READ_RETRIES     3

/* BJT on D2 (PA10). Change these to suit what the BJT switches. */
#define BJT_DURING_PHOTO 1         /* 1 = BJT on only while the photo is taken  */
#define BJT_SETTLE_MS    5         /* wait after switching on, before capturing */
#define BJT_SELF_TEST    1         /* 1 = blink the BJT 3 times at start-up     */

static uint8_t rx[4 + CHUNK];      /* largest response: offset + one chunk    */
static frame_t f = { .buf = rx };
static char line[64];

static uint32_t get_u32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t get_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static void     put_u32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

/* Standard CRC-32 (same as Python zlib.crc32), so the laptop can check the file. */
static uint32_t crc32_update(uint32_t crc, const uint8_t *p, uint32_t n)
{
    crc = ~crc;
    while (n--) {
        crc ^= *p++;
        for (int i = 0; i < 8; i++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

/* Send one request and wait for its matching response. 0 = ok. */
static int transact(uint8_t cmd, const void *req, uint16_t len, uint32_t timeout_ms)
{
    bsp_cam_flush();                               /* forget leftovers */
    if (proto_send(cmd, req, len)) return -10;
    int r = proto_recv(&f, sizeof rx, timeout_ms);
    if (r) return r;
    return f.cmd == RSP(cmd) ? 0 : -4;
}

static int camera_ping(void)
{
    return transact(CMD_PING, NULL, 0, 300) == 0 && f.len == 4 && memcmp(rx, "OMV1", 4) == 0;
}

static void fail(const char *why)
{
    snprintf(line, sizeof line, "FAIL %s\n", why);
    bsp_vcp_print(line);
    for (int i = 0; i < 3; i++) {                  /* three quick blinks */
        bsp_led(1); bsp_delay_ms(80);
        bsp_led(0); bsp_delay_ms(80);
    }
}

static void take_photo(void)
{
    bsp_led(1);

    /* 1. Ask the camera to capture and compress one frame.
     *    The BJT (e.g. light or laser) is on only for the capture itself. */
    if (BJT_DURING_PHOTO) {
        bsp_bjt(1);
        bsp_delay_ms(BJT_SETTLE_MS);
    }
    uint8_t q = JPEG_QUALITY;
    int r = transact(CMD_PHOTO, &q, 1, 3000);
    bsp_bjt(0);                                    /* never leave it on */
    if (r || f.len < 10) {
        fail(r == -3 ? "crc error on PHOTO" : "camera did not answer PHOTO");
        return;
    }
    uint32_t size = get_u32(&rx[0]);
    uint16_t w = get_u16(&rx[4]), h = get_u16(&rx[6]), shot = get_u16(&rx[8]);
    if (size == 0 || size > MAX_JPEG_BYTES) { fail("bad photo size"); return; }

    snprintf(line, sizeof line, "PHOTO %lu %u %u\n", (unsigned long)size, w, h);
    bsp_vcp_print(line);

    /* 2. Pull the JPEG across piece by piece and hand each piece to the laptop. */
    uint32_t crc = 0;
    for (uint32_t off = 0; off < size; ) {
        uint16_t n = (uint16_t)((size - off) < CHUNK ? (size - off) : CHUNK);
        uint8_t req[6];
        put_u32(req, off);
        req[4] = (uint8_t)n; req[5] = (uint8_t)(n >> 8);

        int ok = 0;
        for (int t = 0; t < READ_RETRIES && !ok; t++)   /* a piece can be asked for again */
            ok = transact(CMD_READ, req, sizeof req, 500) == 0
                 && f.len == 4u + n && get_u32(rx) == off;
        if (!ok) { fail("lost a piece of the photo"); return; }

        snprintf(line, sizeof line, "DATA %lu %u\n", (unsigned long)off, n);
        bsp_vcp_print(line);
        bsp_vcp_write(&rx[4], n);
        crc = crc32_update(crc, &rx[4], n);
        off += n;
    }

    snprintf(line, sizeof line, "DONE %08lx\n", (unsigned long)crc);
    bsp_vcp_print(line);
    if (shot != 0xFFFF) {
        snprintf(line, sizeof line, "camera also saved photo_%04u.jpg to its SD card\n", shot);
        bsp_vcp_print(line);
    }
    bsp_led(0);
}

int main(void)
{
    /* Clocks, HAL, and the peripherals configured in STM32CubeMX2. */
    if (mx_system_init() != SYSTEM_OK) return -1;
    if (bsp_init() != 0) return -1;

    bsp_vcp_print("\nUREx camera link - press B1 (blue button) to take a photo\n");

    if (BJT_SELF_TEST) {                           /* check the BJT wiring */
        bsp_vcp_print("BJT self-test: D2 on/off 3 times\n");
        for (int i = 0; i < 3; i++) {
            bsp_bjt(1); bsp_delay_ms(300);
            bsp_bjt(0); bsp_delay_ms(300);
        }
    }

    /* The OpenMV board needs a couple of seconds to boot and run main.py. */
    int online = 0;
    for (int i = 0; i < 20 && !online; i++) {
        online = camera_ping();
        if (!online) bsp_delay_ms(250);
    }
    bsp_vcp_print(online ? "camera online\n"
                         : "camera not answering - check wiring, SB48/SB49, main.py\n");

    for (;;) {
        if (!bsp_button()) continue;
        bsp_delay_ms(20);                          /* debounce */
        if (!bsp_button()) continue;

        if (!camera_ping()) {
            fail("camera not answering");
        } else {
            take_photo();
        }
        while (bsp_button()) { }                   /* wait for release */
        bsp_delay_ms(20);
    }
}
