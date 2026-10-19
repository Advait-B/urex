/* proto.h - framed UART link between the Nucleo (U1) and the OpenMV camera (U2).
 *
 * Frame:  0xA5 | CMD | LEN (u16, little-endian) | PAYLOAD | CRC-8
 * CRC-8:  polynomial 0x07, initial value 0x00, over CMD, LEN and PAYLOAD.
 * The camera answers every request with CMD | 0x80.
 *
 * 0x01-0x05 are unchanged from UREX-CMOS-002 Rev B (Table 4).
 * 0x06 and 0x07 are new: they move a full JPEG photo in small pieces so the
 * STM32 (64 KB RAM) never has to hold the whole picture.
 */
#ifndef PROTO_H
#define PROTO_H

#include <stdint.h>

enum {
    CMD_PING    = 0x01,  /* -> 0x81 "OMV1"                                    */
    CMD_SNAP    = 0x02,  /* -> 0x82 u16 w, u16 h, u8 mean, 80x60 thumbnail     */
    CMD_GRAB    = 0x03,  /* stage 2                                            */
    CMD_PROFILE = 0x04,  /* stage 2-3                                          */
    CMD_EXPOSE  = 0x05,  /* stage 2-3                                          */
    CMD_PHOTO   = 0x06,  /* req: u8 jpeg quality -> 0x86 u32 size, u16 w, u16 h, u16 shot */
    CMD_READ    = 0x07   /* req: u32 offset, u16 n -> 0x87 u32 offset, n bytes */
};
#define RSP(c) ((uint8_t)((c) | 0x80))

#define PROTO_CHUNK_MAX 512u          /* largest READ the camera will answer */

typedef struct { uint8_t cmd; uint16_t len; uint8_t *buf; } frame_t;

int proto_send(uint8_t cmd, const void *payload, uint16_t len);   /* 0 = ok */

/* 0 = ok, -1 = timeout, -2 = frame larger than cap, -3 = CRC error */
int proto_recv(frame_t *f, uint16_t cap, uint32_t timeout_ms);

#endif /* PROTO_H */
